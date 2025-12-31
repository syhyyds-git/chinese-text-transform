import torch
from data_loader import subsequent_mask
class Beam:
    def __init__(self,size,pad,bos,eos,device=False):
        self.size=size
        self._done=False
        self.PAD=pad
        self.BOS=bos
        self.EOS=eos
        
        self.scores=torch.zeros((size,),dtype=torch.float,device=device)
        #每个候选序列到目前为止的总累积分数
        self.all_scores=[]
        #记录每个时间步开始前的分数状态
        self.prev_ks=[]
        #记录每个时间步的前一步来自哪个beam
        self.next_ys=[torch.full((size,),self.PAD,dtype=torch.long,device=device)]
        #记录每个时间步的预测结果(词表中的id)
        self.next_ys[0][0]=self.BOS
    def get_current_state(self):
        return self.get_tentative_hypothesis()
    def get_current_origin(self):
        return self.prev_ks[-1]
    @property
    def done(self):
        return self._done
    #外部代码可以 beam.done 直接访问
    def  advance(self,word_logprob):
        #word_logprob模型对下一个可能词的对数概率预测
        #形状:(beam_size,vocab_size)
        num_words=word_logprob.size(1)
        if len(self.prev_ks)>0:
            beam_lk=word_logprob+self.scores.unsqueeze(1).expand_as(word_logprob)#(beam_size,vocab_size)
        else:
            beam_lk=word_logprob[0]#(1,vocab_size)第一次只从BOS更新

        flat_beam_lk=beam_lk.view(-1)
        best_scores,best_scores_id=flat_beam_lk.topk(self.size,0,True,True)
        """
        0: 表示按第0维(即行)排序
        1.True: 表示按降序排序（即最大的在前面）
        2.True: 表示返回排序后的值(best_scores)和对应的索引(best_scores_id)
        """
        self.all_scores.append(self.scores)
        self.scores=best_scores

        prev_k=best_scores_id//num_words#(beam_size)
        self.prev_ks.append(prev_k)

        self.next_ys.append(best_scores_id-num_words*prev_k)

        if self.next_ys[-1][0].item()==self.EOS:
            self._done=True
            self.all_scores.append(self.scores)

        return self._done

    def sort_scores(self):
        return torch.sort(self.scores,0,True)
        """
        0 - 排序的维度
        True - 降序排列
        """
    def get_the_best_score_and_idx(self):
        sorted_scores, sorted_idx = self.sort_scores()
        return sorted_scores[0], sorted_idx[0]
    
    def get_tentative_hypothesis(self):

        if len(self.next_ys)==1:
            dec_seq=self.next_ys[0].unsqueeze(1)
        else:
            _,keys=self.sort_scores()
            #按照分数，对当前beams的排序
            hyps=[self.get_hypothesis(k) for k in keys]
            hyps=[[self.BOS]+h for h in hyps]
            dec_seq=torch.LongTensor(hyps)

        return dec_seq
    def get_hypothesis(self,k):
        hyp=[]
        for j in range(len(self.prev_ks)-1,-1,-1):
            hyp.append(self.next_ys[j+1][k])#当前步第k个beam的预测值
            k=self.prev_ks[j][k]#回溯上一步是第几个beam

        return list(map(lambda x:x.item(),hyp[::-1]))    










def beam_search(model,src,src_mask,max_len,pad,bos,eos,beam_size,device):

    def get_inst_idx_to_tensor_position_map(inst_idx_list):
        return {inst_idx:tensor_position for tensor_position,inst_idx in enumerate(inst_idx_list) }

    def collect_active_part(beamed_tensor,curr_active_inst_idx,n_prev_active_inst,n_bm):
        """
        beamed_tensor:输入张量，包含所有实例（包括已完成和未完成的）的数据
        curr_active_inst_idx:当前活跃实例的索引列表
        n_prev_active_inst:上一步活跃实例的数量
        n_bm:当前批次的beam大小
        """
        _,*d_hs=beamed_tensor.size()
        #[number_of_batch,beam_size_of_one_batch,seq_len,h_dimension]
        n_curr_active_inst=len(curr_active_inst_idx)

        new_shape=(n_curr_active_inst*n_bm,*d_hs)

        beamed_tensor=beamed_tensor.view(n_prev_active_inst,-1)
        beamed_tensor=beamed_tensor.index_select(0,curr_active_inst_idx)
        """
        沿着指定维度选择索引对应的元素
        dim=0:沿着第0维(行)进行选择
        indices=curr_active_inst_idx:要选择的行索引
        """
        beamed_tensor=beamed_tensor.view(*new_shape)
        return beamed_tensor
    def beam_decode_step(inst_dec_beams,len_dec_seq,enc_output,inst_idx_to_position_map,n_bm):
        def prepare_beam_dec_seq(inst_dec_beams,len_dec_seq):
            dec_partial_seq=[b.get_current_state() for b in inst_dec_beams if not b.done]
            dec_partial_seq=torch.stack(dec_partial_seq).to(device)
            """
            1.未进行stack钱,dec_partial_seq是Python 列表,stack 才是真正构建 batch tensor
            2.torch.stack 是 PyTorch 中的一个函数，
            用于沿着新维度把多个 tensor 堆叠起来，和 torch.cat(按已有维度拼接)不一样，它会产生一个新的维度
            形状:[batch_size,beam_size,len_dec_seq]
            """
            dec_partial_seq=dec_partial_seq.view(-1,len_dec_seq)
            #[batch_size*beam_size,dec_seq_len]
            return dec_partial_seq
        def predict_word(dec_seq,enc_output,n_active_inst,n_bm):
            """
            enc_output: (batch*beam, src_len, hidden)
            src_mask: (batch*beam, 1, src_len)
            dec_seq: (batch*beam, dec_len)
            """
            assert enc_output.shape[0]==dec_seq.shape[0]==src_mask.shape[0]
            out=model.decode(enc_output,src_mask,dec_seq,subsequent_mask(dec_seq.size(1))
                             .type_as(src.data))
            #(batch*beam, dec_len, hidden_dim)
            word_logprob=model.generator(out[:,-1])
            #(batch*beam, vocab_size)
            word_logprob=word_logprob.view(n_active_inst,n_bm,-1)
            #(n_active_inst, beam_size, vocab_size)
            return word_logprob
        def collect_active_inst_idx_list(inst_beams,word_prob,inst_idx_to_position_map):
            active_inst_idx_list=[]
            for inst_idx,inst_position in  inst_idx_to_position_map.items():
                is_inst_complete=inst_beams[inst_idx].advance(word_prob[inst_position])

                if not is_inst_complete:
                    active_inst_idx_list+=[inst_idx]
                
            return active_inst_idx_list


        n_active_inst=len(inst_idx_to_position_map)

        dec_seq=prepare_beam_dec_seq(inst_dec_beams,len_dec_seq)

        word_logprob=predict_word(dec_seq,enc_output,n_active_inst,n_bm)
        
        active_inst_idx_list=collect_active_inst_idx_list(inst_dec_beams,word_logprob,inst_idx_to_position_map)
        return active_inst_idx_list

        
    def collate_active_info(src_enc,src_mask,inst_idx_to_position_map,active_inst_idx_list):
        n_prev_active_inst=len(inst_idx_to_position_map)
        active_inst_idx=[inst_idx_to_position_map[inst_idx] for inst_idx in active_inst_idx_list]
        
        active_inst_idx=torch.LongTensor(active_inst_idx).to(device)

        active_src_enc=collect_active_part(src_enc,active_inst_idx,n_prev_active_inst,beam_size)
        active_src_mask=collect_active_part(src_mask,active_inst_idx,n_prev_active_inst,beam_size)
        active_inst_idx_to_position_map=get_inst_idx_to_tensor_position_map(active_inst_idx_list)

        return active_src_enc,active_src_mask,active_inst_idx_to_position_map

    def collect_hypothesis_and_scores(inst_dec_beams,n_best):
        all_hyp,all_scores=[],[]
        for inst_idx in range(len(inst_dec_beams)):
            scores,tail_idxs=inst_dec_beams[inst_idx].sort_scores()
            all_scores+=[scores[:n_best]]
            hyps=[inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
            all_hyp+=[hyps]

        return all_hyp,all_scores


    with torch.no_grad():
        src_enc=model.encode(src,src_mask)

        NBEST=beam_size
        batch_size,sent_len,h_dim=src_enc.size()
        src_enc=src_enc.repeat(1,beam_size,1).view(batch_size*beam_size,sent_len,h_dim)
        src_mask=src_mask.repeat(1,beam_size,1).view(batch_size*beam_size,1,src_mask.shape[-1])

        inst_dec_beams=[Beam(beam_size,pad,bos,eos,device) for _ in range(batch_size)]

        active_inst_idx_list=list(range(batch_size))

        inst_idx_to_position_map=get_inst_idx_to_tensor_position_map(active_inst_idx_list)

        for len_dec_seq in range(1,max_len+1):

            active_inst_idx_list=beam_decode_step(inst_dec_beams,len_dec_seq,src_enc,inst_idx_to_position_map,beam_size)

            if not active_inst_idx_list:
                break
            src_enc,src_mask,inst_idx_to_position_map=collate_active_info(src_enc,src_mask,inst_idx_to_position_map,active_inst_idx_list)
            
    batch_hyp,batch_scores=collect_hypothesis_and_scores(inst_dec_beams,NBEST)
    return batch_hyp,batch_scores
