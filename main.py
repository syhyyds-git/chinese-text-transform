import logging
import utils
import config
from data_loader import MTDataset
from torch.utils.data import DataLoader
from model import make_model,LabelSmoothing
import torch
from train import train,test,translate
from utils import english_tokenizer_load
import numpy as np
class NoamOpt:
    def __init__(self,model_size,factor,warmup,optimizer):
        self.optimizer=optimizer
        self._step=0
        self.warmup=warmup
        self.factor=factor
        self.model_size=model_size
        self._rate=0
    def step(self):
        self._step+=1
        rate=self.rate()
        for p in self.optimizer.param_groups:
            p['lr']=rate
        self._rate=rate
        self.optimizer.step()
    def rate(self,step=None):
        if step is None:
            step=self._step
        return self.factor*(self.model_size**(-0.5)*min(step**(-0.5),step*self.warmup**(-1.5)))
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model,1,10000,
                       torch.optim.Adam(model.parameters(),lr=0,betas=(0.9,0.98),eps=1e-9))



def run():
    utils.chinese_tokenizer_load()
    utils.english_tokenizer_load()
    utils.set_logger(config.log_path)

    train_dataset=MTDataset(config.train_data_path)
    dev_dataset=MTDataset(config.dec_data_path)
    test_dataset=MTDataset(config.test_data_path)

    logging.info("--------Dataset Loaded!--------")
    
    train_dataloader=DataLoader(train_dataset,batch_size=config.batch_size,shuffle=True,
                                collate_fn=train_dataset.collate_fn)
    dev_dataloader=DataLoader(dev_dataset,batch_size=config.batch_size,shuffle=False,
                                collate_fn=dev_dataset.collate_fn)
    test_dataloader=DataLoader(test_dataset,batch_size=config.batch_size,shuffle=False,
                                collate_fn=test_dataset.collate_fn)
    
    logging.info("--------Dataloader Loaded!--------")

    model=make_model(config).to(config.device)

    if config.use_smoothing:
        criterion=LabelSmoothing(size=config.vocab_size,
                                 padding_idx=config.padding_idx,
                                 smoothing=config.label_smoothing).to(config.device)
    else:
        criterion=torch.nn.CrossEntropyLoss(ignore_index=config.padding_idx,
                                            reduction='sum').to(config.device)
    if config.use_noamopt:
        optimizer=get_std_opt(model)
    else:
        optimizer=torch.optim.AdamW(model.parameters(),lr=config.lr)

    train(train_dataloader,dev_dataloader,model,model,criterion,optimizer)
    
    test(test_dataloader,model,criterion)
def one_sentence_translate(sent,beam_search=True):
    model=make_model(config).to(config.device)
    model.eval()
    with torch.no_grad():
        BOS=english_tokenizer_load().bos_id()
        EOS=english_tokenizer_load().eos_id()
        src_tokens=[[BOS]+english_tokenizer_load().EncodeAsIds(sent)+[EOS]]
        batch_input=torch.LongTensor(np.array(src_tokens)).to(config.device)
        print(translate(batch_input,model,use_beam=beam_search))


def translate_example():

     sent = ("The near-term policy remedies are clear: raise the minimum wage to a level that will keep a "
            "fully employed worker and his or her family out of poverty, and extend the earned-income tax credit "
            "to childless workers.")
     one_sentence_translate(sent,beam_search=True)



if __name__ =="__main__":
    #run() #训练
    translate_example()#单句翻译
