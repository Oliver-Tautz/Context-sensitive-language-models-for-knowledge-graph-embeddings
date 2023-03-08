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

import copy
from collections import defaultdict
import torchmetrics

from tqdm import tqdm, trange
from rdflib import Graph
from settings import VECTOR_SIZE,BERT_MAXLEN,BERT_EPOCHS,BERT_BATCHSIZE,DEBUG
import sys
import torch
from torch.utils.data import DataLoader
from utils_data import DataseSimpleTriple
import pandas as pd
import shutil
VERBOSE=1

def verbprint(str):
    if VERBOSE:
        print(str)

def main(args):

    try:

        cfg_parser = configparser.ConfigParser(allow_no_value=True)
        cfg_parser.read(args.config)

        SETTING_DATASET_PATH = Path(cfg_parser['TRAIN']['DATASET_PATH'])
        SETTING_VECTOR_SIZE= cfg_parser.getint('TRAIN','VECTOR_SIZE')
        SETTING_BERT_EPOCHS = cfg_parser.getint('TRAIN','BERT_EPOCHS')
        SETTING_BERT_NAME = cfg_parser.get('TRAIN','BERT_NAME')
        SETTING_BERT_MAXLEN = cfg_parser.getint('TRAIN','BERT_MAXLEN')
        SETTING_BERT_BATCHSIZE = cfg_parser.getint('TRAIN', 'BERT_BATCHSIZE')
        SETTING_DEBUG = cfg_parser.getboolean('TRAIN', 'DEBUG')
        SETTING_WORK_FOLDER = Path(SETTING_BERT_NAME)

        shutil.copyfile(args.config, SETTING_WORK_FOLDER / Path(args.config).name)
        
        if SETTING_WORK_FOLDER.is_file():
            inp = input('Model already present! Overwrite? [y/N]')
            if not(inp[0] == 'y' or inp[0] == 'Y'):
                print('User abort.')
                exit(0)

    except Exception as e:
        print(f'Your config is wrong or missing parameters :(\n Exception was {e}')
        exit(-1)


    verbprint("Loading Dataset")

    g_train = Graph()
    g_val = Graph()

    g_train = g_train.parse('FB15k-237/train.nt', format='nt')
    g_val   = g_val.parse('FB15k-237/valid.nt', format='nt')

    # Join triples together
    dataset_most_simple = [' '.join(x) for x in g_train]
    dataset_most_simple_eval = [' '.join(x) for x in g_val]


    if SETTING_DEBUG:
        dataset_most_simple = dataset_most_simple[0:10000]
        dataset_most_simple_eval = dataset_most_simple_eval[0:1000]
        BERT_EPOCHS = 3
        BERT_BATCHSIZE=5000


    verbprint(f"example data: {dataset_most_simple[0:10]}")
    verbprint("Init Model.")
    # Init tokenizer and config from tinybert
    tz = BertTokenizer.from_pretrained("bert-base-cased")
    special_tokens_map = tz.special_tokens_map_extended

    verbprint(f"special tokens: {special_tokens_map}")

    dataset_simple = DataseSimpleTriple(dataset_most_simple,special_tokens_map)
    tz = dataset_simple.get_tokenizer()
    dataset_simple_eval = DataseSimpleTriple(dataset_most_simple_eval,special_tokens_map,tokenizer=tz)


    verbprint(f"example data processed: {dataset_simple[0]}")


    tiny_pretrained = AutoModel.from_pretrained('prajjwal1/bert-tiny')
    tiny_config = tiny_pretrained.config



    # Change parameters
    tiny_config._name_or_path="otautz/tiny"
    encoder_config = copy.copy(tiny_config)
    encoder_config.is_decoder = False
    encoder_config.add_cross_attention = False
    encoder_config.num_labels=tz.get_vocab_size()
    encoder_config.hidden_size = VECTOR_SIZE
    encoder_config.max_position_embeddings = BERT_MAXLEN

    del tiny_pretrained

    #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    print(f'cuda available: {torch.cuda.is_available()}')
    tiny_encoder  = BertForTokenClassification(encoder_config)
    tiny_encoder  = tiny_encoder.to(device)

    lossF = torch.nn.CrossEntropyLoss()

    dl = DataLoader(dataset_simple,batch_size=BERT_BATCHSIZE,shuffle=True,pin_memory=True)
    dl_eval =  DataLoader(dataset_simple_eval, batch_size=BERT_BATCHSIZE, shuffle=False, pin_memory=True)

    optimizer = torch.optim.Adam(tiny_encoder.parameters())

    loss_metric = torchmetrics.aggregation.MeanMetric().to(device)
    batchloss_metric = torchmetrics.aggregation.CatMetric().to(device)
    batchloss_metric_eval = torchmetrics.aggregation.CatMetric().to(device)
    history = defaultdict(list)

    verbprint(f"cuda_model: {next(tiny_encoder.parameters()).is_cuda}")


    verbprint("Starting training")

    for epochs in trange(BERT_EPOCHS):
        for inputs, batch_mask, batch_labels in dl:
            tiny_encoder.train()
            optimizer.zero_grad()
            batch_id = inputs[:, :, 0]

            out = tiny_encoder.forward(batch_id.to(device), batch_mask.to(device))
            logits = out.logits

            # (batchsize, sequence_len, no_labels)
            logits_shape = logits.shape

            # (batchsize * sequence_len, no_labels)
            logits_no_sequence = logits.reshape(logits_shape[0] * logits_shape[1], logits_shape[2])

            # (batchsize)
            batch_labels_no_sequence = batch_labels.flatten().to(device)

            batch_mask = (inputs[:, :, 1] > 0).flatten().to(device)

            loss = lossF(logits_no_sequence[batch_mask], batch_labels_no_sequence[batch_mask])

            loss.backward()
            optimizer.step()

            # detach loss! Otherwise causes memory leakage
            loss_metric(loss.detach())
            batchloss_metric(loss.detach())

        history['loss'].append(loss_metric.compute().detach().item())
        loss_metric.reset()
        with torch.no_grad():
            tiny_encoder.eval()
            for inputs, batch_mask, batch_labels in dl_eval:
                optimizer.zero_grad()
                batch_id = inputs[:, :, 0]

                out = tiny_encoder.forward(batch_id.to(device), batch_mask.to(device))
                logits = out.logits

                # (batchsize, sequence_len, no_labels)
                logits_shape = logits.shape

                # (batchsize * sequence_len, no_labels)
                logits_no_sequence = logits.reshape(logits_shape[0] * logits_shape[1], logits_shape[2])

                # (batchsize)
                batch_labels_no_sequence = batch_labels.flatten().to(device)

                batch_mask = (inputs[:, :, 1] > 0).flatten().to(device)

                loss = lossF(logits_no_sequence[batch_mask], batch_labels_no_sequence[batch_mask])

                loss_metric(loss.detach())
                batchloss_metric_eval(loss.detach())

            history['loss_eval'].append(loss_metric.compute().detach().item())
            loss_metric.reset()


    # Save data
    os.makedirs(SETTING_WORK_FOLDER,exist_ok=True)

    pd.DataFrame(history).to_csv(SETTING_WORK_FOLDER / 'bert_loss_eval.csv')
    pl = pd.DataFrame(history).plot()
    pl.figure.savefig(SETTING_WORK_FOLDER / 'loss_eval_loss.pdf')

    pd.DataFrame(batchloss_metric.compute().detach().cpu()).to_csv(SETTING_WORK_FOLDER / 'bert_batchloss.csv')
    pl = pd.DataFrame(batchloss_metric.compute().detach().cpu()).plot()
    pl.figure.savefig(SETTING_WORK_FOLDER / 'batchloss.pdf')


    pd.DataFrame(batchloss_metric_eval.compute().detach().cpu()).to_csv(SETTING_WORK_FOLDER / 'bert_batchloss_eval.csv')
    pl = pd.DataFrame(batchloss_metric_eval.compute().detach().cpu()).plot()
    pl.figure.savefig(SETTING_WORK_FOLDER / 'batchloss_eval.pdf')

    # Save model
    tiny_encoder.save_pretrained(SETTING_WORK_FOLDER / "tiny_bert_from_scratch_simple_eval")
    tz.save(str(SETTING_WORK_FOLDER / 'tiny_bert_from_scratch_simple_tokenizer_eval.json'))

    # compute test score
    g_test = Graph()
    g_test = g_test.parse('FB15k-237/test.nt', format='nt')
    dataset_most_simple_test = [' '.join(x) for x in g_test]
    dataset_simple_test = DataseSimpleTriple(dataset_most_simple_test,special_tokens_map,tokenizer=tz)
    dl_test =  DataLoader(dataset_simple_test, batch_size=BERT_BATCHSIZE, shuffle=False, pin_memory=True)

    if DEBUG:
        dataset_most_simple_test = dataset_most_simple_test[0:1000]

    with torch.no_grad():
        tiny_encoder.eval()
        loss_metric.reset()
        for inputs, batch_mask, batch_labels in dl_test:
            optimizer.zero_grad()
            batch_id = inputs[:, :, 0]

            out = tiny_encoder.forward(batch_id.to(device), batch_mask.to(device))
            logits = out.logits

            # (batchsize, sequence_len, no_labels)
            logits_shape = logits.shape

            # (batchsize * sequence_len, no_labels)
            logits_no_sequence = logits.reshape(logits_shape[0] * logits_shape[1], logits_shape[2])

            # (batchsize)
            batch_labels_no_sequence = batch_labels.flatten().to(device)

            batch_mask = (inputs[:, :, 1] > 0).flatten().to(device)

            loss = lossF(logits_no_sequence[batch_mask], batch_labels_no_sequence[batch_mask])

            loss_metric(loss.detach())
            batchloss_metric_eval(loss.detach())

        testscorefile = open(SETTING_WORK_FOLDER / 'testscore.txt',mode='w')
        testscorefile.write(str(loss_metric.compute().item()))
        testscorefile.close()
        print('testloss = ' + str(loss_metric.compute().item()))

if __name__ == '__main__':
    print('HERE!')
    parser = argparse.ArgumentParser(
            description="Script to train bert model on a Knowledge graph.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=textwrap.dedent("""""")
        )
    
    parser.add_argument("--config", help="Filepath of the config file. See example config.", type=str,
                        default="bert_train_conf/example_config.ini")

    parser.add_argument("--debug", help="Use small datasets and model for testing. Overwrites config option",
                        action='store_true')
    
    args=parser.parse_args()

    main(args)