import os
import logging 
import sentencepiece as spm
def chinese_tokenizer_load():
    sp_chn=spm.SentencePieceProcessor()
    sp_chn.Load('./ch.model')
    return sp_chn
def english_tokenizer_load():
    sp_eng=spm.SentencePieceProcessor()
    sp_eng.Load('./eng.model')
    return sp_eng



def set_logger(log_path):
    

    if os.path.exists(log_path) is True:
        os.remove(log_path)
    logger=logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler=logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

        stream_handler=logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
        




