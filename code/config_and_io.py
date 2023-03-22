import configparser
from pathlib import Path

def parse_config(config_path):
    try:

            cfg_parser = configparser.ConfigParser(allow_no_value=True)
            cfg_parser.read(config_path)
            global SETTING_DATASET_PATH
            SETTING_DATASET_PATH = Path(cfg_parser['TRAIN']['DATASET_PATH'])
            SETTING_VECTOR_SIZE= cfg_parser.getint('TRAIN','VECTOR_SIZE')
            SETTING_BERT_EPOCHS = cfg_parser.getint('TRAIN','BERT_EPOCHS')
            SETTING_BERT_NAME = cfg_parser.get('TRAIN','BERT_NAME')
            SETTING_BERT_MAXLEN = cfg_parser.getint('TRAIN','BERT_MAXLEN')
            SETTING_BERT_BATCHSIZE = cfg_parser.getint('TRAIN', 'BERT_BATCHSIZE')
            SETTING_DEBUG = cfg_parser.getboolean('TRAIN', 'DEBUG')
            SETTING_WORK_FOLDER = Path(SETTING_BERT_NAME)



    except Exception as e:
            print(f'Your config is wrong or missing parameters :(\n Exception was {e}')
            exit(-1)

if __name__ == "__main__":
    parse_config("/home/olli/gits/Better_Knowledge_Graph_Embeddings2/code/conf/example_config.ini")
    print(SETTING_DATASET_PATH)