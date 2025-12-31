import config
from data_loader import subsequent_mask
import math
import copy
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE=config.device
class LabelSmoothing(nn.Module):
    def __init__(self,size,padding_idx,smoothing=0.0):
        super(LabelSmoothing,self).__init__()
        self.criterion=nn.KLDivLoss(reduction='sum')#所有loss求和
        self.padding_idx=padding_idx
        self.confidence=1.0-smoothing
        self.smoothing=smoothing
        self.size=size
        self.true_dist=None
    def forward(self,x,target):
        """
        x:[b*seq_len,vocab_size]
        target:[b*seq_len]

        """
        assert x.size(1)==self.size
        true_dist=x.data.clone()
        true_dist.fill_(self.smoothing/(self.size-2))
        true_dist.scatter_(1,target.data.unsqueeze(1),self.confidence)
        """
        scatter_(dim, index, src):
        在指定维度 dim 上，根据 index 的位置，把值 src 写入对应位置
        unsqueeze(1) 变成 [batch_size*seq_len, 1]，方便 scatter_ 对列索引定位
        """
        true_dist[:,self.padding_idx]=0

        mask=torch.nonzero(target.data==self.padding_idx)
        # 把 padding 位置的 true_dist 设为 0
        if mask.dim>0:
            true_dist.index_fill_(0,mask.squeeze(),0.0)
            """
            index_fill_(dim, index, value)：在指定维度 dim 的指定 index 位置，
            将值 value 原地覆盖
            把 padding 行的概率分布全部设为 0
            """
        self.true_dist=true_dist
        return self.criterion(x,Variable(true_dist,requires_grad=False))



def clones(module,n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])
class Embeddings(nn.Module):
    def __init__(self,d_model,vocab):
        super(Embeddings,self).__init__()
        self.lut=nn.Embedding(vocab,d_model)
        self.d_model=d_model
    def forward(self,x):
        return self.lut(x)*math.sqrt(self.d_model)
        """
        乘 math.sqrt(d_model) 是为了 让 embedding 和位置编码在数值量级上相近，
        """
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout,max_len=5000):
        super(PositionalEncoding,self).__init__()
        self.dropout=nn.Dropout(dropout)

        pe=torch.zeros(max_len,d_model)
        position=torch.arange(0,max_len,device=DEVICE).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,d_model,2,device=DEVICE)*-(math.log(10000.0)/d_model))

        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)

        pe=pe.unsqueeze(0)#[1,max_len,d_model]
        self.register_buffer('pe',pe)
    def forward(self,x):
        x=x+Variable(self.pe[:,:x.size(1)],requires_grad=False)
        """
        [b,seq_len,d_model]
        seq_len:这批语句中最长的序列长度
        """
        return self.dropout(x)
    

def attention(query,key,value,mask=None,droput=None):
    d_k=query.size(-1)

    scores=query@key.transpose(-2,-1)/math.sqrt(d_k)
    #[b,num_heads,seq_len,seq_len]
    
    if mask is not None:
        scores=scores.masked_fill(mask==0,-1e9)
    p_attn=F.softmax(scores,dim=-1)

    if droput is not None:
        p_attn=droput(p_attn)
    return p_attn@value,p_attn    

class MultiHeadedAttention(nn.Module):
    def __init__(self,num_heads,d_model,dropout=0.1):
        super(MultiHeadedAttention,self).__init__()
        self.num_heads=num_heads
        assert d_model%num_heads==0
        self.d_k=d_model//num_heads
        self.linears=clones(nn.Linear(d_model,d_model),4)
        #定义4个全连接层，用于Q,K,V和最后concat之后输出
        self.attn=None
        self.dropout=nn.Dropout(p=dropout)
    def forward(self,query,key,value,mask=None):
        if mask is not None:
            mask=mask.unsqueeze(1)#[b,1,len_seq,len_seq]
        nbatches=query.size(0)
        query,key,value=[l(x).view(nbatches,-1,self.num_heads,self.d_k).transpose(1,2)
                         for l,x in zip(self.linears,(query,key,value))
                         ]#[b,num_heads,seq_len,d_k] 
        x,self.attn=attention(query,key,value,mask=mask,droput=self.dropout)

        x=x.transpose(1,2).contiguous().view(nbatches,-1,self.num_heads*self.d_k)
        #[b,seq_len,d_model]
        return self.linears[-1](x)
           

class SublayerConnection(nn.Module):
    def __init__(self,d_model,dropout):
        super(SublayerConnection,self).__init__()
        self.norm=LayerNorm(d_model)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x,sublayer):
        return x+self.dropout(sublayer(self.norm(x)))
        """
        原始写法(Post-LN)
        x = LayerNorm(x + Sublayer(x)) 
        Pre-LN:
        x = Sublayer(LayerNorm(x)) + x
        残差路径 x → x 是一个 恒等映射
        即使 sublayer 的梯度变小或震荡，也不会“断掉梯度链”
        """
class LayerNorm(nn.Module):
    def __init__(self,features,eps=1e-6):
        super(LayerNorm,self).__init__()
        self.a_2=nn.Parameter(torch.ones(features))#[d_model]
        self.b_2=nn.Parameter(torch.zeros(features))
        self.eps=eps
    def forward(self,x):
        mean=x.mean(-1,keepdim=True)
        std=x.std(-1,keepdim=True)
        return self.a_2*(x-mean)/torch.sqrt(std**2+self.eps)+self.b_2

class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super(PositionwiseFeedForward,self).__init__()
        self.w1=nn.Linear(d_model,d_ff)
        self.w2=nn.Linear(d_ff,d_model)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        return self.w2(self.dropout(F.relu(self.w1(x))))    


class EncoderLayer(nn.Module):
    def __init__(self,d_model,self_attn,feed_forward,dropout):
        super(EncoderLayer,self).__init__()
        self.self_attn=self_attn
        self.feed_forward=feed_forward
        self.d_model=d_model
        self.sublayer=clones(SublayerConnection(d_model,dropout),2)
        self.size=d_model
    def forward(self,x,mask):
        x=self.sublayer[0](x,lambda x: self.self_attn(x,x,x,mask))
        return self.sublayer[1](x,lambda x:self.feed_forward(x))

class Encoder(nn.Module):
    def __init__(self,layer,n_layers):
        super(Encoder,self).__init__()
        self.layers=nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])
        self.norm=LayerNorm(layer.size)
    def forward(self,x,mask):
        for layer in self.layers:
            x=layer(x,mask)
        return self.norm(x)
class Transformer(nn.Module):
    def __init__(self,encoder,decoder,src_embed,tgt_embed,generator):
        super(Transformer,self).__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.src_embed=src_embed
        self.tgt_embed=tgt_embed
        self.generator=generator
    def encode(self,src,src_mask):
        if src_mask is not None:
            src_mask = src_mask.to(src.device)
        return self.encoder(self.src_embed(src),src_mask)
    def decode(self,memory,src_mask,tgt,tgt_mask):
        if src_mask is not None:
            src_mask = src_mask.to(memory.device)
        if tgt_mask is not None:
            tgt_mask = tgt_mask.to(tgt.device)
        return self.decoder(self.tgt_embed(tgt),memory,src_mask,tgt_mask)
    def forward(self,src,tgt,src_mask,tgt_mask):
        return self.decode(self.encode(src,src_mask),src_mask,tgt,tgt_mask)

class DecoderLayer(nn.Module):
    def __init__(self,d_model,self_attn,src_attn,feed_forward,dropout):
        super(DecoderLayer,self).__init__()
        self.d_model=d_model
        self.self_attn=self_attn
        self.src_attn=src_attn
        self.feed_forward=feed_forward
        self.feed_forward=feed_forward
        self.sublayer=clones(SublayerConnection(d_model,dropout),3)
        self.size=d_model
    def forward(self,x,memory,src_mask,tgt_mask):
        #用m来存放encoder最终hiddenn表示结果
        m=memory
        # Self-Attention：注意self-attention的q，k和v均为decoder hidden
        x=self.sublayer[0](x,lambda x:self.self_attn(x,x,x,tgt_mask))
        #Context-Attention：注意context-attention的q为decoder hidden，而k和v为encoder hidden
        x=self.sublayer[1](x,lambda x:self.src_attn(x,m,m,src_mask))

        return self.sublayer[2](x,self.feed_forward)
        


class Decoder(nn.Module):
    def __init__(self,layer,n_layers):
        super(Decoder, self).__init__()
        self.layers=clones(layer,n_layers)
        self.norm=LayerNorm(layer.size)
    def forward(self,x,memory,src_mask,tgt_mask):

        for layer in self.layers:
            x=layer(x,memory,src_mask,tgt_mask)
        return self.norm(x)




def make_model(config):
    c=copy.deepcopy
    """
    deepcopy(obj) 会创建一个 对象的完整独立副本，包括：
    该对象本身；
    它内部引用的所有子对象（递归复制）；
    每个副本都有独立的内存空间。
    """
    attn=MultiHeadedAttention(config.n_heads,config.d_model).to(DEVICE)
    ff=PositionwiseFeedForward(config.d_model,config.d_ff,config.dropout).to(DEVICE)    
    position=PositionalEncoding(config.d_model,config.dropout).to(DEVICE)

    model=Transformer(
        Encoder(EncoderLayer(config.d_model,c(attn),c(ff),config.dropout)
                .to(DEVICE),config.n_layers).to(DEVICE),
        Decoder(DecoderLayer(config.d_model,c(attn),c(attn),c(ff),config.dropout)
                .to(DEVICE),config.n_layers).to(DEVICE),
        nn.Sequential(Embeddings(config.d_model,config.src_vocab_size).to(DEVICE),c(position)),
        nn.Sequential(Embeddings(config.d_model,config.tgt_vocab_size).to(DEVICE),c(position)),
        Generator(config.d_model,config.tgt_vocab_size).to(DEVICE)
    )
    for p in model.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
    return model.to(DEVICE)
class Generator(nn.Module):
    def __init__(self,d_model,vocab_size):
        super(Generator,self).__init__()
        self.proj=nn.Linear(d_model,vocab_size)

    def forward(self,x):
        return F.log_softmax(self.proj(x),dim=-1)
def batch_greedy_decode(model,src,src_mask,max_len=64,start_symbol=2,end_symbol=3):
    batch_size,src_seq_len=src.size()
    results=[[] for _ in range(batch_size)]
    stop_flag=[False for _ in range(batch_size)]
    count=0

    memory=model.encode(src,src_mask)
    tgt=torch.Tensor(batch_size,1).fill_(start_symbol).type_as(src.data)

    for s in range(max_len):
        tgt_mask=subsequent_mask(tgt.size(1)).expand(batch_size,-1,-1).type_as(src.data)
        #[b,t,t]
        out=model.decode(memory,src_mask,Variable(tgt),Variable(tgt_mask))
        
        prob=model.generator(out[:,-1,:])#[b,vocab_size]
        pred=torch.argmax(prob,dim=-1)#[b]
        tgt=torch.cat(tgt,pred.unsqueeze(1),dim=1)
        pred=pred.cpu().numpy()

        for i in range(batch_size):
            if stop_flag[i] is False:
                if pred[i]==end_symbol:
                    count+=1
                    stop_flag[i]=True
                else:
                    results[i].append(pred[i].item())
            if count==batch_size:
                break
        if count==batch_size:
            break
    return results

def greedy_decode(model,src,src_mask,max_len=64,start_symbol=2,end_symbol=3):

    memory=model.encode(src,src_mask)

    ys=torch.ones(1,1).fill_(start_symbol).type_as(src.data)

    for i in range(max_len-1):
        out=model.decode(memory,src_mask,Variable(ys)
                         ,Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        prob=model.generator(out[:,-1])#[1,vocab_size]
        _,next_word=torch.max(prob,dim=-1)
        """
        返回两个张量：
        1.values,每个位置的最大值
        2.indices,每个位置最大值的索引
        """
        next_word=next_word[0]
        if next_word==end_symbol:
            break
        ys=torch.cat([ys,torch.ones(1,1).type_as(src.data).fill_(next_word)],dim=1)
    return ys    

