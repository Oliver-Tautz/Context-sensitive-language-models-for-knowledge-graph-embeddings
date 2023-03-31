import argparse 
import configparser
from pathlib import Path

cfg_parser = configparser.ConfigParser(allow_no_value=True)
cfg_parser.read('example_config.ini')

SETTING_DATASET_PATH = Path(cfg_parser['TRAIN']['DATASET_PATH'])
SETTING_VECTOR_SIZE= cfg_parser.getint('TRAIN','VECTOR_SIZE')
SETTING_BERT_EPOCHS = cfg_parser.getint('TRAIN','BERT_EPOCHS')
SETTING_BERT_NAME = cfg_parser.get('TRAIN','BERT_NAME')
SETTING_BERT_MAXLEN = cfg_parser.getint('TRAIN','BERT_MAXLEN')
SETTING_BERT_BATCHSIZE = cfg_parser.getint('TRAIN', 'BERT_BATCHSIZE')
SETTING_DEBUG = cfg_parser.getboolean('TRAIN', 'DEBUG')


print(SETTING_BERT_BATCHSIZE)
print(SETTING_BERT_EPOCHS)
print(SETTING_DATASET_PATH)