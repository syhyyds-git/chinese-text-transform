import torch


d_model=128
n_heads=4
n_layers=6
d_k=32
d_v=32
d_ff=512
dropout=0.1
padding_idx=0
bos_idx=2
eos_idx=3
src_vocab_size=16000
tgt_vocab_size=8000
batch_size=32
epoch_num=30
early_stop=5
lr=1e-4

max_len=60
beam_size=3
use_smoothing=False
use_noamopt=True

data_dir='./data'

train_data_path = './data/json/train.json'
dec_data_path = './data/json/dev.json'
test_data_path = './data/json/test.json'

model_path='./experiment/model.pth'
log_path='./experiment/train.log'
output_path='./experiment/output.txt'

gpu_id='0'
device_id=[0]

import torch

if gpu_id!='' and torch.cuda.is_available():
    device=torch.device(f'cuda:{gpu_id}')
else:
    device=torch.device('cpu')
print("Using device:",device)




