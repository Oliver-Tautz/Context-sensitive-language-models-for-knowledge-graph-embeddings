import transformers
from tokenizers.models import WordLevel
from tokenizers import Tokenizer
from transformers import BertTokenizer, EncoderDecoderModel, BertForTokenClassification
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import WordLevelTrainer
from tokenizers.processors import BertProcessing
from transformers import BertConfig, BertModel, AutoModel
from utils_graph import get_entities

import copy
from collections import defaultdict
import torchmetrics

from tqdm import tqdm, trange
from rdflib import Graph
from settings import VECTOR_SIZE,BERT_SIMPLE_MAXLEN,BERT_EPOCHS,BERT_BATCHSIZE,DEBUG
import sys
import torch
from torch.utils.data import DataLoader
from utils_data import DataseSimpleTriple
import pandas as pd

VERBOSE=1

def verbprint(str):
    if VERBOSE:
        print(str)

# Parse rdf

verbprint("Loading Dataset")

g_train = Graph()
g_val = Graph()

g_train = g_train.parse('FB15k-237/train.nt', format='nt')
g_val   = g_val.parse('FB15k-237/valid.nt', format='nt')

# Join triples together
dataset_most_simple = [' '.join(x) for x in g_train]
dataset_most_simple_eval = [' '.join(x) for x in g_val]


if DEBUG:
    dataset_most_simple = dataset_most_simple[0:10000]
    dataset_most_simple_eval = dataset_most_simple_eval[0:1000]
    BERT_EPOCHS = 10
    BERT_BATCHSIZE=5000

verbprint("Init Model.")
# Init tokenizer and config from tinybert
tz = BertTokenizer.from_pretrained("bert-base-cased")
special_tokens_map = tz.special_tokens_map_extended


dataset_simple = DataseSimpleTriple(dataset_most_simple,special_tokens_map)
dataset_simple_eval = DataseSimpleTriple(dataset_most_simple_eval,special_tokens_map)

tz = dataset_simple.get_tokenizer()



tiny_pretrained = AutoModel.from_pretrained('prajjwal1/bert-tiny')
tiny_config = tiny_pretrained.config



# Change parameters
tiny_config._name_or_path="otautz/tiny"
encoder_config = copy.copy(tiny_config)
encoder_config.is_decoder = False
encoder_config.add_cross_attention = False
encoder_config.num_labels=tz.get_vocab_size()
encoder_config.hidden_size = VECTOR_SIZE
encoder_config.max_position_embeddings = BERT_SIMPLE_MAXLEN

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

    history['loss'].append(loss_metric.compute().item())
    loss_metric.reset()
    with torch.no_grad():
        tiny_encoder.eval()
        for inputs, batch_mask, batch_labels in dl_eval:
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

        history['loss_eval'].append(loss_metric.compute().item())
        loss_metric.reset()


# Save data
pd.DataFrame(history).to_csv('bert_loss_eval.csv')
pl = pd.DataFrame(history).plot()
pl.figure.savefig('loss_eval_loss.pdf')

pd.DataFrame(batchloss_metric.compute().detach().cpu()).to_csv('bert_batchloss.csv')
pl = pd.DataFrame(batchloss_metric.compute().detach().cpu()).plot()
pl.figure.savefig('batchloss.pdf')


pd.DataFrame(batchloss_metric_eval.compute().detach().cpu()).to_csv('bert_batchloss_eval.csv')
pl = pd.DataFrame(batchloss_metric_eval.compute().detach().cpu()).plot()
pl.figure.savefig('batchloss_eval.pdf')

# Save model
tiny_encoder.save_pretrained("tiny_bert_from_scratch_simple_eval")
tz.save('tiny_bert_from_scratch_simple_tokenizer_eval.json')

# compute test score
g_test = Graph()
g_test = g_test.parse('FB15k-237/test.nt', format='nt')
dataset_most_simple_test = [' '.join(x) for x in g_test]
dataset_simple_test = DataseSimpleTriple(dataset_most_simple_test,special_tokens_map)
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

    print('testloss = ' + loss_metric.compute().item())

