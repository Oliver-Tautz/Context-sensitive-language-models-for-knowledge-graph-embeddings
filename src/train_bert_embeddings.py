import transformers
from tokenizers.models import WordLevel
from tokenizers import Tokenizer
from transformers import BertTokenizer, EncoderDecoderModel, BertForTokenClassification, BertForNextSentencePrediction
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
import math
from tqdm import tqdm, trange
from rdflib import Graph
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils_data import DatasetBertTraining, DatasetBertTraining_LP, DatasetBertTraining_LM
import pandas as pd
import shutil
from utils import verbprint
from utils_train import train_bert_embeddings_mlm, score_bert_model_mlm, train_bert_embeddings_lp, score_bert_model_lp, train_bert_embeddings_lm
from jrdf2vec_walks_for_bert import generate_walks
from profiler import Profiler
from sklearn.model_selection import train_test_split
from utils_graph import parse_kg

def main(args):
    try:

        cfg_parser = configparser.ConfigParser(allow_no_value=True)
        cfg_parser.read(args.config)

        SETTING_DATASET_PATH = Path(cfg_parser['TRAIN']['DATASET_PATH'])
        SETTING_VECTOR_SIZE = cfg_parser.getint('TRAIN', 'VECTOR_SIZE')
        SETTING_BERT_EPOCHS = cfg_parser.getint('TRAIN', 'BERT_EPOCHS')
        SETTING_BERT_NAME = cfg_parser.get('TRAIN', 'BERT_NAME')

        SETTING_BERT_BATCHSIZE = cfg_parser.getint('TRAIN', 'BERT_BATCHSIZE')
        SETTING_BERT_CLASSIFIER_DROPOUT = cfg_parser.getfloat('TRAIN', 'BERT_CLASSIFIER_DROPOUT')
        SETTING_DEBUG = cfg_parser.getboolean('TRAIN', 'DEBUG')
        SETTING_TMP_DIR = args.tmp_dir

        SETTING_BERT_WALK_USE = cfg_parser.getboolean('TRAIN', 'BERT_WALK_USE')
        SETTING_BERT_WALK_DEPTH = cfg_parser.getint('TRAIN', 'BERT_WALK_DEPTH')
        SETTING_BERT_WALK_COUNT = cfg_parser.getint('TRAIN', 'BERT_WALK_COUNT')
        SETTING_BERT_WALK_GENERATION_MODE = cfg_parser.get('TRAIN', 'BERT_WALK_GENERATION_MODE')

        SETTING_BERT_DATASET_TYPE = cfg_parser.get('TRAIN', 'BERT_DATASET_TYPE').upper()

        SETTING_STOP_EARLY = cfg_parser.getint('TRAIN', 'BERT_EARLY_STOPPING_PATIENCE')
        SETTINGS_STOP_EARLY_DELTA = cfg_parser.getfloat('TRAIN', 'BERT_EARLY_STOPPING_DELTA')

        SETTING_BERT_MASK_CHANCE = cfg_parser.getfloat('TRAIN', 'BERT_MASK_CHANCE')
        SETTING_BERT_MASK_TOKEN_CHANCE = cfg_parser.getfloat('TRAIN', 'BERT_MASK_TOKEN_CHANCE')
        SETTING_BERT_LM_WRONG_SAMPLE_NUMBER = 10

        if SETTING_BERT_WALK_USE:
            SETTING_WORK_FOLDER = Path(
                f"{SETTING_BERT_NAME}_ep{SETTING_BERT_EPOCHS}_vec{SETTING_VECTOR_SIZE}_walks_d={SETTING_BERT_WALK_DEPTH}_w={SETTING_BERT_WALK_COUNT}_g={SETTING_BERT_WALK_GENERATION_MODE}_dataset={SETTING_BERT_DATASET_TYPE}_mask-chance={SETTING_BERT_MASK_CHANCE}_mask-token-chance={SETTING_BERT_MASK_TOKEN_CHANCE}")

            # depth*2 + path + CLS + SEP
            SETTING_BERT_MAXLEN = SETTING_BERT_WALK_DEPTH * 2 + 3
        else:
            # triple, so we have cls,e,r,e,sep
            SETTING_BERT_MAXLEN = 5
            SETTING_WORK_FOLDER = Path(
                f"{SETTING_BERT_NAME}_ep{SETTING_BERT_EPOCHS}_vec{SETTING_VECTOR_SIZE}_dataset={SETTING_BERT_DATASET_TYPE}_mask-chance={SETTING_BERT_MASK_CHANCE}_mask-token-chance={SETTING_BERT_MASK_TOKEN_CHANCE}")

        if SETTING_BERT_DATASET_TYPE == 'LP':
            SETTING_BERT_MAXLEN = 5
        elif SETTING_BERT_DATASET_TYPE == 'LM':
            SETTING_BERT_MAXLEN = 9

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
    if SETTING_BERT_DATASET_TYPE == 'MLM' or SETTING_BERT_DATASET_TYPE == 'MASS':
        if not SETTING_BERT_WALK_USE:
            print('using input!')
            g_train = Graph()
            g_val = Graph()

            g_train = g_train.parse(SETTING_DATASET_PATH / 'train.nt', format='nt')
            g_val = g_val.parse(SETTING_DATASET_PATH / 'valid.nt', format='nt')

            # Join input together
            dataset = [' '.join(x) for x in g_train]
            dataset_eval = [' '.join(x) for x in g_val]
        else:
            # use walks
            # train/test split from train walks as eval graph is not connected.
            walks_name = f'{f"{SETTING_BERT_NAME}_ep{SETTING_BERT_EPOCHS}_vec{SETTING_VECTOR_SIZE}"}'

            walks_name = generate_walks(SETTING_TMP_DIR, SETTING_DATASET_PATH / 'train.nt',
                                        walks_name, 4,
                                        SETTING_BERT_WALK_DEPTH, SETTING_BERT_WALK_COUNT,
                                        SETTING_BERT_WALK_GENERATION_MODE)

            # Warning, this reads lines with \n in it. Whitespace splitting takes care of it in the tokenizer.
            walkspath_file = open(walks_name, 'r')
            dataset = walkspath_file.readlines()
            walkspath_file.close()

            dataset, dataset_eval = train_test_split(dataset, train_size=0.75, test_size=0.25, random_state=328)

            print(len(dataset), len(dataset_eval))
    elif SETTING_BERT_DATASET_TYPE == "LP":
        g_train = Graph()
        g_val = Graph()

        g_train = g_train.parse(SETTING_DATASET_PATH / 'train.nt', format='nt')
        g_val = g_val.parse(SETTING_DATASET_PATH / 'valid.nt', format='nt')

        # Join input together
        dataset = [' '.join(x) for x in g_train]
        dataset_eval = [' '.join(x) for x in g_val]

    elif SETTING_BERT_DATASET_TYPE == "LM":


        walks_name = f'{f"{SETTING_BERT_NAME}_ep{SETTING_BERT_EPOCHS}_vec{SETTING_VECTOR_SIZE}"}'

        walks_name = generate_walks(SETTING_TMP_DIR, SETTING_DATASET_PATH / 'train.nt',
                                    walks_name, 4,
                                    3, 10,
                                    SETTING_BERT_WALK_GENERATION_MODE)

        # Warning, this reads lines with \n in it. Whitespace splitting takes care of it in the tokenizer.
        walkspath_file = open(walks_name, 'r')
        real_walks = [ s for s in [l.split() for l in walkspath_file.readlines()] if len(s) == 7]

        real_walks , real_walks_eval  = train_test_split(real_walks, train_size=0.75, test_size=0.25, random_state=328)
        walkspath_file.close()
        entities, predicates, edges, predicate_ix = parse_kg(SETTING_DATASET_PATH / 'train.nt')

        def get_wrong_walks(walks, randomize = True):
            """
            Randomize just picks random input. If your dataset has lots of input for a single entity and not many for others
            you might want to implement a check, wether wrong walks are really wrong ...
            """

            if randomize:
                wrong_walks = []
                for walk in walks:
                    original_triple = walk[0:4]
                    random_edge_ix = np.random.randint(0,len(edges),SETTING_BERT_LM_WRONG_SAMPLE_NUMBER)

                    for random_ix in random_edge_ix:
                        edge = edges[random_ix]
                        predicate = predicates[predicate_ix[random_ix]]
                        entity0 = entities[edge[0]]
                        entity1 = entities[edge[1]]
                        random_triple = [entity0,predicate,entity1]
                        wrong_walks.append(original_triple + random_triple)


                return wrong_walks
            else:
                # Not implemented
                pass

        def get_split_walks(walks):
            return [[f"{w[0]} {w[1]} {w[2]}",f"{w[4]} {w[5]} {w[6]}" ] for w in walks]

        wrong_walks = get_wrong_walks(real_walks)
        wrong_walks_eval = get_wrong_walks(real_walks_eval)


        real_walks =  get_split_walks(real_walks)
        wrong_walks = get_split_walks(wrong_walks)

        real_walks_eval = get_split_walks(real_walks_eval)
        wrong_walks_eval = get_split_walks(wrong_walks_eval)

        dataset = real_walks + wrong_walks
        labels = torch.cat((torch.zeros(len(real_walks),dtype=torch.long),torch.ones(len(wrong_walks),dtype=torch.long)))
        dataset_eval = real_walks_eval + wrong_walks_eval
        labels_eval = torch.cat((torch.zeros(len(real_walks_eval),dtype=torch.long), torch.ones(len(wrong_walks_eval),dtype=torch.long)))

        print(dataset[0:2])





    if SETTING_DEBUG:
        if SETTING_BERT_DATASET_TYPE == "LM":
            selection = np.random.randint(0,len(labels),1000)
            dataset = np.array(dataset)[selection]
            labels = np.array(labels)[selection]

            selection_eval = np.random.randint(0,len(labels_eval),100)
            dataset = np.array(dataset_eval)[selection_eval]
            labels = np.array(labels_eval)[selection_eval]

        else:
            dataset = dataset[0:1000]
            dataset_eval = dataset_eval[0:100]

        # SETTING_BERT_EPOCHS = 10
        SETTING_BERT_BATCHSIZE = 5000
    verbprint(f"example data: {dataset[0:10]}")
    verbprint("Init Model.")
    # Init tokenizer and config from tinybert
    tz = BertTokenizer.from_pretrained("bert-base-cased")
    special_tokens_map = tz.special_tokens_map_extended

    verbprint(f"special tokens: {special_tokens_map}")
    if SETTING_BERT_DATASET_TYPE == 'MLM':
        dataset = DatasetBertTraining(dataset, special_tokens_map, dataset_type=SETTING_BERT_DATASET_TYPE,
                                      mask_chance=SETTING_BERT_MASK_CHANCE,
                                      mask_token_chance=SETTING_BERT_MASK_TOKEN_CHANCE)
        tz = dataset.get_tokenizer()
        dataset_eval = DatasetBertTraining(dataset_eval, special_tokens_map, tokenizer=tz,
                                           dataset_type=SETTING_BERT_DATASET_TYPE, mask_chance=SETTING_BERT_MASK_CHANCE,
                                           mask_token_chance=SETTING_BERT_MASK_TOKEN_CHANCE)
    elif SETTING_BERT_DATASET_TYPE == 'LP':
        dataset = DatasetBertTraining_LP(dataset, special_tokens_map, tokenizer=None, max_length=128)
        tz = dataset.get_tokenizer()
        dataset_eval = DatasetBertTraining_LP(dataset_eval, special_tokens_map, tokenizer=tz, max_length=128)
    elif SETTING_BERT_DATASET_TYPE == 'LM':
        dataset = DatasetBertTraining_LM(dataset,labels,special_tokens_map,tokenizer = None, max_length = 128)
        tz = dataset.get_tokenizer()
        dataset_eval = DatasetBertTraining_LM(dataset_eval,labels_eval,special_tokens_map,tokenizer = None, max_length = 128)


    else:
        print('Wrong Dataset Option!')
        exit(-1)

    verbprint(f"example data processed: {dataset[0:2]}")

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
    print(SETTING_BERT_DATASET_TYPE)
    print(f'cuda available: {torch.cuda.is_available()}')
    if SETTING_BERT_DATASET_TYPE == 'MLM':
        tiny_encoder = BertForTokenClassification(encoder_config)
    elif SETTING_BERT_DATASET_TYPE == 'LP':
        tiny_encoder = BertModel(encoder_config)
    elif SETTING_BERT_DATASET_TYPE == 'LM':
        tiny_encoder = BertForNextSentencePrediction(encoder_config)
        print('heree!',tiny_encoder)

    tiny_encoder = tiny_encoder.to(device)



    verbprint("Starting training")

    os.makedirs(SETTING_WORK_FOLDER, exist_ok=True)
    os.makedirs(SETTING_PLOT_FOLDER, exist_ok=True)
    os.makedirs(SETTING_DATA_FOLDER, exist_ok=True)

    if SETTING_BERT_DATASET_TYPE == 'MLM':
        lossF = torch.nn.CrossEntropyLoss()
        tiny_encoder, optimizer, history, profile = train_bert_embeddings_mlm(tiny_encoder, SETTING_BERT_EPOCHS, dataset,
                                                                              dataset_eval, SETTING_BERT_BATCHSIZE,
                                                                              torch.optim.Adam, lossF, device,
                                                                              SETTING_WORK_FOLDER,
                                                                              stop_early_patience=SETTING_STOP_EARLY,
                                                                              stop_early_delta=SETTINGS_STOP_EARLY_DELTA)
    elif SETTING_BERT_DATASET_TYPE == 'LP':
        lossF = torch.nn.BCEWithLogitsLoss()
        tiny_encoder, optimizer, history, profile, classifier = train_bert_embeddings_lp(tiny_encoder, SETTING_BERT_EPOCHS, dataset,
                                                                              dataset_eval, SETTING_BERT_BATCHSIZE,
                                                                              torch.optim.Adam, lossF, device,
                                                                              SETTING_WORK_FOLDER,
                                                                              stop_early_patience=SETTING_STOP_EARLY,
                                                                              stop_early_delta=SETTINGS_STOP_EARLY_DELTA)

    elif SETTING_BERT_DATASET_TYPE == 'LM':
        tiny_encoder, optimizer, history, profile = train_bert_embeddings_lm(tiny_encoder,
                                                                                         SETTING_BERT_EPOCHS, dataset,
                                                                                         dataset_eval,
                                                                                         SETTING_BERT_BATCHSIZE,
                                                                                         torch.optim.Adam,
                                                                                         device,
                                                                                         SETTING_WORK_FOLDER,
                                                                                         stop_early_patience=SETTING_STOP_EARLY,
                                                                                         stop_early_delta=SETTINGS_STOP_EARLY_DELTA)
        pass


    # Save data

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
    tz.save(str(SETTING_WORK_FOLDER / "tokenizer.json"))

    if SETTING_BERT_DATASET_TYPE == 'LP':
        torch.save(classifier.state_dict(),str(SETTING_WORK_FOLDER / "classifier_model"))

    with open(SETTING_WORK_FOLDER / 'special_tokens_map.json', 'w', encoding='utf-8') as f:
        json.dump(special_tokens_map, f, ensure_ascii=False, indent=4)

    # compute test score
    g_test = Graph()
    g_test = g_test.parse(SETTING_DATASET_PATH / 'test.nt', format='nt')
    dataset_test = [' '.join(x) for x in g_test]

    if SETTING_DEBUG:
        dataset_test = dataset_test[0:1000]

    if SETTING_BERT_DATASET_TYPE == 'MLM':
        dataset_test = DatasetBertTraining(dataset_test, special_tokens_map, tokenizer=tz,
                                           dataset_type=SETTING_BERT_DATASET_TYPE, mask_chance=SETTING_BERT_MASK_CHANCE,
                                           mask_token_chance=SETTING_BERT_MASK_TOKEN_CHANCE)
        testscore = score_bert_model_mlm(tiny_encoder, device, dataset_test, SETTING_BERT_BATCHSIZE, optimizer, lossF)
    elif SETTING_BERT_DATASET_TYPE == 'LP':
        dataset_test = DatasetBertTraining_LP(dataset_test, special_tokens_map, tokenizer=tz, max_length=128)
        testscore = score_bert_model_lp(tiny_encoder, device, dataset_test, SETTING_BERT_BATCHSIZE, optimizer,
                                                 lossF,classifier)
    elif SETTING_BERT_DATASET_TYPE == 'LM':
        pass
        exit(0)




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
