import transformers
from tokenizers.models import WordLevel
from tokenizers import Tokenizer
from transformers import BertTokenizer, EncoderDecoderModel, BertForTokenClassification
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import WordLevelTrainer
from tokenizers.processors import BertProcessing
from transformers import BertConfig, BertModel, AutoModel
from utils_graph import get_entities
import textwrap
import argparse
import configparser
from pathlib import Path
import os
import json
import copy
from collections import defaultdict
import torchmetrics

from tqdm import tqdm, trange
from rdflib import Graph

import torch
from torch.utils.data import DataLoader
from utils_data import DatasetBertTraining
import pandas as pd
import shutil
from utils import verbprint
from utils_train import train_bert_embeddings, score_bert_model
from jrdf2vec_walks_for_bert import generate_walks
from profiler import Profiler

def main(args):
    try:

        cfg_parser = configparser.ConfigParser(allow_no_value=True)
        cfg_parser.read(args.config)

        SETTING_DATASET_PATH = Path(cfg_parser['TRAIN']['DATASET_PATH'])
        SETTING_VECTOR_SIZE = cfg_parser.getint('TRAIN', 'VECTOR_SIZE')
        SETTING_BERT_EPOCHS = cfg_parser.getint('TRAIN', 'BERT_EPOCHS')
        SETTING_BERT_NAME = cfg_parser.get('TRAIN', 'BERT_NAME')
        SETTING_BERT_MAXLEN = cfg_parser.getint('TRAIN', 'BERT_MAXLEN')
        SETTING_BERT_BATCHSIZE = cfg_parser.getint('TRAIN', 'BERT_BATCHSIZE')
        SETTING_BERT_CLASSIFIER_DROPOUT = cfg_parser.getfloat('TRAIN', 'BERT_CLASSIFIER_DROPOUT')
        SETTING_DEBUG = cfg_parser.getboolean('TRAIN', 'DEBUG')
        SETTING_TMP_DIR = args.tmp_dir

        SETTING_BERT_WALK_USE = cfg_parser.getboolean('TRAIN', 'BERT_WALK_USE')
        SETTING_BERT_WALK_DEPTH = cfg_parser.getint('TRAIN', 'BERT_WALK_DEPTH')
        SETTING_BERT_WALK_COUNT = cfg_parser.getint('TRAIN', 'BERT_WALK_COUNT')
        SETTING_BERT_WALK_GENERATION_MODE = cfg_parser.get('TRAIN', 'BERT_WALK_GENERATION_MODE')

        SETTING_BERT_DATASET_TYPE = cfg_parser.get('TRAIN', 'BERT_DATASET_TYPE')

        if SETTING_BERT_WALK_USE:
            SETTING_WORK_FOLDER = Path(
                f"{SETTING_BERT_NAME}_ep{SETTING_BERT_EPOCHS}_vec{SETTING_VECTOR_SIZE}_walks_d={SETTING_BERT_WALK_DEPTH}_w={SETTING_BERT_WALK_COUNT}_g={SETTING_BERT_WALK_GENERATION_MODE}_dataset={SETTING_BERT_DATASET_TYPE}")
        else:
            SETTING_WORK_FOLDER = Path(f"{SETTING_BERT_NAME}_ep{SETTING_BERT_EPOCHS}_vec{SETTING_VECTOR_SIZE}")

        SETTING_PLOT_FOLDER = SETTING_WORK_FOLDER / 'plot'
        SETTING_DATA_FOLDER = SETTING_WORK_FOLDER / 'data'

        # Prompt user for overwrite
        if not args.no_prompt:
            if SETTING_WORK_FOLDER.is_file() or SETTING_WORK_FOLDER.is_dir():
                inp = input('Model already present! Overwrite? [y/N]')
                if inp == "" or not (inp[0] == 'y' or inp[0] == 'Y'):
                    print('User abort.')
                    exit(0)
        # Overwrite or create workfolder
        os.makedirs(SETTING_WORK_FOLDER, exist_ok=True)
        shutil.copyfile(args.config, SETTING_WORK_FOLDER / 'config.ini')


    except Exception as e:
        print(f'Your config is wrong or missing parameters :(\n Exception was {e}')
        exit(-1)

    if args.debug:
        SETTING_DEBUG = True
    verbprint("Loading Dataset")

    if not SETTING_BERT_WALK_USE:
        g_train = Graph()
        g_val = Graph()

        g_train = g_train.parse(SETTING_DATASET_PATH / 'train.nt', format='nt')
        g_val = g_val.parse(SETTING_DATASET_PATH / 'valid.nt', format='nt')

        # Join triples together
        dataset = [' '.join(x) for x in g_train]
        dataset_eval = [' '.join(x) for x in g_val]
    else:

        walks_name = f'./tmp/{f"{SETTING_BERT_NAME}_ep{SETTING_BERT_EPOCHS}_vec{SETTING_VECTOR_SIZE}"}'
        walkspath_eval_name = f'./tmp/{f"{SETTING_BERT_NAME}_ep{SETTING_BERT_EPOCHS}_vec{SETTING_VECTOR_SIZE}"}'

        walks_name = generate_walks(SETTING_TMP_DIR, SETTING_DATASET_PATH / 'train.nt',
                                    walks_name, 4,
                                    SETTING_BERT_WALK_DEPTH, SETTING_BERT_WALK_COUNT,
                                    SETTING_BERT_WALK_GENERATION_MODE)

        walkspath_eval_name = generate_walks('./tmp', SETTING_DATASET_PATH / 'train.nt',
                                             walkspath_eval_name, 4,
                                             SETTING_BERT_WALK_DEPTH, SETTING_BERT_WALK_COUNT,
                                             SETTING_BERT_WALK_GENERATION_MODE)

        # Warning, this reads lines with \n in it. Whitespace splitting takes care of it in the tokenizer.
        walkspath_file = open(walks_name, 'r')
        dataset = walkspath_file.readlines()
        walkspath_file.close()

        walkspath_eval_file = open(walkspath_eval_name, 'r')
        dataset_eval = walkspath_eval_file.readlines()
        walkspath_eval_file.close()

    if SETTING_DEBUG:
        dataset = dataset[0:10000]
        dataset_eval = dataset_eval[0:1000]
        SETTING_BERT_EPOCHS = 10
        SETTING_BERT_BATCHSIZE = 5000


    verbprint(f"example data: {dataset[0:10]}")
    verbprint("Init Model.")
    # Init tokenizer and config from tinybert
    tz = BertTokenizer.from_pretrained("bert-base-cased")
    special_tokens_map = tz.special_tokens_map_extended

    verbprint(f"special tokens: {special_tokens_map}")

    dataset = DatasetBertTraining(dataset, special_tokens_map, dataset_type=SETTING_BERT_DATASET_TYPE)
    tz = dataset.get_tokenizer()
    dataset_simple_eval = DatasetBertTraining(dataset_eval, special_tokens_map, tokenizer=tz,
                                              dataset_type=SETTING_BERT_DATASET_TYPE)

    verbprint(f"example data processed: {dataset[0]}")

    tiny_pretrained = AutoModel.from_pretrained('prajjwal1/bert-tiny')
    tiny_config = tiny_pretrained.config

    # Change parameters
    tiny_config._name_or_path = "otautz/tiny"
    encoder_config = copy.copy(tiny_config)
    encoder_config.is_decoder = False
    encoder_config.add_cross_attention = False
    encoder_config.num_labels = tz.get_vocab_size()
    encoder_config.hidden_size = SETTING_VECTOR_SIZE
    encoder_config.max_position_embeddings = SETTING_BERT_MAXLEN
    encoder_config.classifier_dropout = SETTING_BERT_CLASSIFIER_DROPOUT


    del tiny_pretrained

    #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'cuda available: {torch.cuda.is_available()}')
    tiny_encoder = BertForTokenClassification(encoder_config)
    tiny_encoder = tiny_encoder.to(device)

    lossF = torch.nn.CrossEntropyLoss()

    verbprint("Starting training")


    tiny_encoder, optimizer, history, profile = train_bert_embeddings(tiny_encoder, SETTING_BERT_EPOCHS, dataset,
                                                             dataset_simple_eval, SETTING_BERT_BATCHSIZE,
                                                             torch.optim.Adam, lossF, device)



    # Save data
    os.makedirs(SETTING_WORK_FOLDER, exist_ok=True)
    os.makedirs(SETTING_PLOT_FOLDER, exist_ok=True)
    os.makedirs(SETTING_DATA_FOLDER, exist_ok=True)

    pd.DataFrame(profile).to_csv(SETTING_DATA_FOLDER / 'performance_profile.csv')

    loss_df = pd.DataFrame({'loss': history['loss'], 'loss_eval': history['loss_eval']})

    loss_df.to_csv(SETTING_DATA_FOLDER / 'bert_loss_eval.csv')

    pl = loss_df.plot()
    pl.figure.savefig(SETTING_PLOT_FOLDER / 'loss_eval_loss.pdf')

    pd.DataFrame(history['batchloss_metric'].compute().detach().cpu()).to_csv(
        SETTING_DATA_FOLDER / 'bert_batchloss.csv')
    pl = pd.DataFrame(history['batchloss_metric'].compute().detach().cpu()).plot()
    pl.figure.savefig(SETTING_PLOT_FOLDER / 'batchloss.pdf')

    pd.DataFrame(history['batchloss_metric_eval'].compute().detach().cpu()).to_csv(
        SETTING_DATA_FOLDER / 'bert_batchloss_eval.csv')
    pl = pd.DataFrame(history['batchloss_metric_eval'].compute().detach().cpu()).plot()
    pl.figure.savefig(SETTING_PLOT_FOLDER / 'batchloss_eval.pdf')

    # Save model
    tiny_encoder.save_pretrained(SETTING_WORK_FOLDER / "model")
    tz.save(str(SETTING_WORK_FOLDER / "tokenizer.json"))\

    with open(SETTING_WORK_FOLDER  /'special_tokens_map.json', 'w', encoding='utf-8') as f:
        json.dump(special_tokens_map, f, ensure_ascii=False, indent=4)

    # compute test score
    g_test = Graph()
    g_test = g_test.parse(SETTING_DATASET_PATH / 'test.nt', format='nt')
    dataset_most_simple_test = [' '.join(x) for x in g_test]
    dataset_simple_test = DatasetBertTraining(dataset_most_simple_test, special_tokens_map, tokenizer=tz,
                                              dataset_type=SETTING_BERT_DATASET_TYPE)

    if SETTING_DEBUG:
        dataset_most_simple_test = dataset_most_simple_test[0:1000]

    testscore = score_bert_model(tiny_encoder, device, dataset_simple_test, SETTING_BERT_BATCHSIZE, optimizer, lossF)
    testscorefile = open(SETTING_DATA_FOLDER / 'testscore.txt', mode='w')
    testscorefile.write(str(testscore))
    testscorefile.close()
    print('testloss = ' + str(testscore))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Script to train bert model on a Knowledge graph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""""")
    )

    parser.add_argument("--config", help="Filepath of the config file. See example config.", type=str,
                        default="conf/example_config.ini")

    parser.add_argument("--debug", help="Use small datasets and model for testing. Overwrites config option",
                        action='store_true')
    parser.add_argument("--no-prompt",
                        help="Dont prompt user, ever! Use this for automated scripting etc. Dangerous though! Can overwrite existing files.",
                        action='store_true')

    parser.add_argument("--torch-threads",
                        help="Set torch.set_num_threads for better performance on multicore machines.", type=int,
                        default=None)

    parser.add_argument("--tmp-dir",
                        help="Set temporary directory. Will be cleared!!! Walks and other data will be stored there. The directory should be either empty or non existent. Make sure you have r/w rights!",
                        type=str,
                        default="./tmp")

    args = parser.parse_args()

    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)

    verbprint("passed args: " + str(args))
    main(args)
