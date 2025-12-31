import torch
import json
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils import english_tokenizer_load,chinese_tokenizer_load
import config
device=config.device
class Batch:
    def __init__(self,src_text,trg_text,src,trg=None,pad=0):
        self.src_text=src_text
        self.trg_text=trg_text
        
        self.src=src.to(device)
        self.src_mask=(src!=pad).unsqueeze(-2)#[b,1,seq_len]
        if trg is not None:
            trg=trg.to(device)

            self.trg=trg[:,:-1]

            self.trg_y=trg[:,1:]

            self.trg_mask=self.make_std_mask(self.trg,pad)
            # 将应输出的target结果中实际的词数进行统计
            self.ntokens=(self.trg_y!=pad).data.sum()
    """
    @staticmethod 表示这个函数不依赖类或实例的状态，
    它只是逻辑上属于这个类的一个“工具函数”。
    """
    @staticmethod
    def make_std_mask(tgt,pad):
        tgt_mask=(tgt!=pad).unsqueeze(-2)#[b,1,seq_len]
        tgt_mask=tgt_mask&Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask#[b,seq_len,seq_len]

def subsequent_mask(size):
    attn_shape=(1,size,size)
    #生成一个右上角(不含主对角线)为全1，左下角(含主对角线)为全0的矩阵
    subsequent_mask=np.triu(np.ones(attn_shape),k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask)==0

class MTDataset(Dataset):
    def __init__(self,data_path):
        self.out_en_sent,self.out_cn_sent=self.get_dataset(data_path,sort=True)
        self.sp_eng=english_tokenizer_load()
        self.sp_ch=chinese_tokenizer_load()
        self.PAD=self.sp_eng.pad_id()
        self.BOS=self.sp_eng.bos_id()
        self.EOS=self.sp_eng.eos_id()
    def len_argsort(self,seq):
        return sorted(range(len(seq)),key=lambda x:len(seq[x]))
    def get_dataset(self,data_path,sort=False):
        dataset=json.load(open(data_path,'r'))
        out_en_sent=[]
        out_cn_sent=[]
        for idx,_ in enumerate(dataset):
            out_en_sent.append(dataset[idx][0])
            out_cn_sent.append(dataset[idx][1])
        if sort:
            sorted_idx=self.len_argsort(out_en_sent)
            out_en_sent=[out_en_sent[idx] for idx in sorted_idx]
            out_cn_sent=[out_cn_sent[idx] for idx in sorted_idx]
        return out_en_sent,out_cn_sent
    def __getitem__(self,idx):
        eng_text=self.out_en_sent[idx]
        chn_text=self.out_cn_sent[idx]
        return [eng_text,chn_text]

    def __len__(self):
        return len(self.out_en_sent)
    def collate_fn(self,batch):
        src_text=[x[0] for x in batch]
        tgt_text=[x[1] for x in batch]

        src_tokens=[[self.BOS]+self.sp_eng.EncodeAsIds(sent)+[self.EOS] for sent in src_text]
        tgt_tokens=[[self.BOS]+self.sp_ch.EncodeAsIds(sent)+[self.EOS] for sent in tgt_text]

        batch_input=pad_sequence([torch.LongTensor(np.array(token)) for token in src_tokens],batch_first=True,padding_value=self.PAD)
        #batch_input = batch_input[:, :config.max_len]
        batch_target=pad_sequence([torch.LongTensor(np.array(token)) for token in tgt_tokens],batch_first=True,padding_value=self.PAD)
    

        return Batch(src_text,tgt_text,batch_input,batch_target,self.PAD)
        
