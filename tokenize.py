import sentencepiece as spm
def train(input, vocab_size, model_name, model_type, character_coverage):
    spm.SentencePieceTrainer.train(
    f"--input={input} "
    f"--model_prefix={model_name} "
    f"--vocab_size={vocab_size} "
    f"--model_type={model_type} "
    f"--character_coverage={character_coverage} "
    f"--pad_id=0 "
    f"--unk_id=1 "
    f"--bos_id=2 "
    f"--eos_id=3"
    )


def run():
    en_input='./corpus.en'
    en_vocab_size=16000
    en_model_name='eng'
    en_model_type='bpe'
    en_character_coverage=1
    train(en_input,en_vocab_size,en_model_name,en_model_type,en_character_coverage)

    ch_input='./corpus.ch'
    ch_vocab_size=8000
    ch_model_name='ch'
    ch_model_type='bpe'
    ch_character_coverage=1
    train(ch_input,ch_vocab_size,ch_model_name,ch_model_type,ch_character_coverage)
def test():
    sp=spm.SentencePieceProcessor()
    text="美国总统特朗普今日抵达夏威夷。"
    
    sp.Load('./ch.model')
    print(sp.EncodeAsPieces(text))
    print(sp.EncodeAsIds(text))
    a = [12907, 277, 7419, 7318, 18384, 28724]
    print(sp.decode_ids(a))


if __name__ =='__main__':
    run()
    #test()