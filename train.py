import torch
import torch.nn as nn
from torch.autograd import Variable

import logging
import sacrebleu

from tqdm import tqdm

import config
from beam_decoder import beam_search

from model import batch_greedy_decode

from utils import chinese_tokenizer_load
def run_epoch(data,model,loss_compute):
    total_tokens=0.
    total_loss=0
    for batch in tqdm(data):
        out=model(batch.src,batch.trg,batch.src_mask,batch.trg_mask)
        loss=loss_compute(out,batch.trg_y,batch.ntokens)
        total_loss+=loss
        total_tokens+=batch.ntokens
    return total_loss/total_tokens


def train(train_data,dev_data,model,model_par,criterion,optimizer):
    """
    model:原始模型,用于保存、评估。
    model_par:多 GPU 训练模型包装，用于 forward/backward。
    两者指向同一组参数，但 model_par 支持多 GPU 并行计算,model 更适合单 GPU 或保存。
    """
    best_bleu_score=0.0
    early_stop=config.early_stop
    """
    早停机制
    如果验证集指标在连续若干个 epoch 内没有改善，就停止训练，节省时间并避免过拟合。
    """
    for epoch in range(1,config.epoch_num+1):
        model.train()
        train_loss=run_epoch(train_data,model_par,MultiGPULossCompute(model.generator,
                                                                      criterion,config.device_id,optimizer))
        logging.info(f"Epoch {epoch} Train Loss: {train_loss:.4f}")

        model.eval()
        dev_loss=run_epoch(dev_data,model_par,MultiGPULossCompute(model.generator,
                                                                criterion,config.device_id,None))
        bleu_score=evaluate(dev_data,model)
        logging.info(f"Epoch {epoch} Dev Loss: {dev_loss:.4f} Bleu Score: {bleu_score:.4f}")

        if bleu_score>best_bleu_score:
            torch.save(model.state_dict(),config.model_path)
            best_bleu_score=bleu_score
            early_stop=config.early_stop
            logging.info("-------- Save Best Model! --------")
        else:
            early_stop-=1
            logging.info(f"Early Stop Left: {early_stop}")
        if early_stop==0:
            logging.info("Early Stop!")
            break
class lossCompute:
    def __init__(self,generator,criterion,opt=None):
        self.generator=generator
        self.criterion=criterion
        self.opt=opt
    def __call__(self,x,y,norm):
        x=self.generator(x)
        loss=self.criterion(x.contiguous().view(-1,x.size(-1)),y.contiguous().view(-1))
        loss=loss/norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            if config.use_noamopt:
                self.opt.optimizer.zero_grad()
            else:
                self.opt.zero_grad()
        return loss.data.item()*norm.float()    
class   MultiGPULossCompute:
    def __init__(self,generator,criterion,devices,opt,chunk_size=5):
        self.generator=generator
        self.criterion=nn.parallel.replicate(criterion,devices=devices)
        #把同一个损失函数对象（criterion）复制到多个 GPU 上
        self.devices=devices
        self.opt=opt
        self.chunk_size=chunk_size
        """
        控制序列分块的大小（沿时间步/序列长度方向）
        序列长度是 50,chunk_size=5 → 分 10 次计算。
        """
    def __call__(self,out,targets,normalize):
        total=0.0

        generator=nn.parallel.replicate(self.generator,devices=self.devices)
        out_scatter=nn.parallel.scatter(out,target_gpus=self.devices)
        #把模型输出切分并分发（scatter）到多张 GPU 上,沿batch方向
        out_grad=[[] for _ in out_scatter ]
        targets=nn.parallel.scatter(targets,target_gpus=self.devices)
        #把真实目标序列切分并分发（scatter）到多张 GPU 上,沿batch方向
        chun_size=self.chunk_size
        for i in range(0,out_scatter[0].size(1),chun_size):
        #每次处理一个时间步长（chunk_size）的输出
            out_column=[[Variable(o[:,i:i+chun_size].data,requires_grad=self.opt is not None)]
                        for o in out_scatter
                        ]   
            """
            out_column = [
            [Variable(out_gpu0[:, i:i+chunk_size].data, requires_grad=...)],
            [Variable(out_gpu1[:, i:i+chunk_size].data, requires_grad=...)],
            ...
            ]

            """
            #取出每个GPU上的目标序列的对应部分
            gen=nn.parallel.parallel_apply(generator,out_column)
            #[gpu_batch_size,chunk_size,vocab_size]
            y=[(g.contiguous().view(-1,g.size(-1)),t[:,i:i+chun_size].contiguous().view(-1))
               for g,t in zip(gen,targets)]
            loss=nn.parallel.parallel_apply(self.criterion,y)
            #在多张 GPU 上并行计算损失函数的结果，每张 GPU 对自己那部分数据单独求 loss，
            # 最后会再汇总。
            l=nn.parallel.gather(loss,target_device=self.devices[0])
            #把各张 GPU 上计算得到的 loss（一个标量张量）
            # 收集到主 GPU（通常是 cuda:0）上，形成一个新的张量。
            l=l.sum()/normalize
            #normalize 一般是总样本数或总 token 数，用来归一化 loss
            total+=l.data

            if self.opt is not None:
                l.backward()
                for j,l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())
                    #j表示 GPU 编号,0表示当前 GPU 上的一组 chunk 数据
                
        if self.opt is not None:
            out_grad=[torch.cat(g,dim=1) for g in out_grad]
            #把每个 GPU 上的梯度拼接起来，形成一个完整的梯度张量,
            # out_grad每个元素[per_gpu_batch_size,seq_len,vocab_size]。
            o1=out
            o2=nn.parallel.gather(out_grad,target_device=self.devices[0])
            #所有 GPU 的 batch 被沿 dim=0 拼接到一起,[b,seq_len,vocab_size]
            o1.backward(gradient=o2)
            #把拼接好的输出梯度沿完整计算图回传到 Transformer 模型参数
            self.opt.step()
            if config.use_noamopt:
                self.opt.optimizer.zero_grad()
            else:
                self.opt.zero_grad()
        return total*normalize    

def evaluate(data,model,mode='dev',use_beam=True):
    sp_chn=chinese_tokenizer_load()
    trg=[]
    res=[]
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data):
            cn_sent=batch.trg_text
            src=batch.src
            src_mask=(src!=0).unsqueeze(-2)
            if use_beam:
                decode_result,_=beam_search(model,src,src_mask,config.max_len,
                                            config.padding_idx,config.bos_idx,
                                            config.eos_idx,config.beam_size,config.device)
            else:
                decode_result=batch_greedy_decode(model,src,src_mask,max_len=config.max_len)
            
            decode_result=[h[0] for h in decode_result]
            translation=[sp_chn.DecodeIds(_s) for _s in decode_result]
            trg.extend(cn_sent)
            res.extend(translation)
    if mode=='test':
        with open(config.output_path,"w",encoding='utf-8') as f:
            for i in range(len(trg)):
                line="idx:" +str(i)+trg[i]+'|||' +res[i]+'\n'
                f.write(line)
    trg=[trg]
    bleu=sacrebleu.corpus_bleu(res,trg,tokenize='zh')
    """
    这里的 tokenize 参数用于指定 如何对文本进行分词/标记化
    表示使用 中文分词 的方式进行标记化
    """
    return float(bleu.score)

def test(data,model,criterion):
    with torch.no_grad():
        model.load_state_dict(torch.load(config.model_path))
        model_par=torch.nn.DataParallel(model)
        model.eval()
        test_loss=run_epoch(data,model_par,MultiGPULossCompute(model.generator,criterion,config.device_id,None))
        bleu_score=evaluate(data,model,'test')
        logging.info(f"test loss:{test_loss:.4f},bleu score:{bleu_score:.4f}")

def translate(src,model,use_beam=True):
    sp_chn=chinese_tokenizer_load()
    model.eval()
    with torch.no_grad():
        model.load_state_dict(torch.load(config.model_path))
        src_mask=(src!=0).unsqueeze(-2)
        if use_beam:
            decode_result,_=beam_search(model,src,src_mask,config.max_len,
                                        config.padding_idx,config.bos_idx,
                                        config.eos_idx,config.beam_size,config.device)
        else:
            decode_result=batch_greedy_decode(model,src,src_mask,max_len=config.max_len)
        decode_result=[h[0] for h in decode_result]
        translation=[sp_chn.DecodeIds(_s) for _s in decode_result]
        return translation


            




