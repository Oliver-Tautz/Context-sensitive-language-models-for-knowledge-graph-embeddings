from settings import VECTOR_SIZE
from settings import CLASSIFIER_EPOCHS
from rdflib import Graph, URIRef
from gensim.models import Word2Vec
from pathlib import Path
import pandas as pd
import torch
import torchmetrics
from models import ClassifierSimple
from utils_train import fit_1_1

import matplotlib.pyplot as plt
from utils_graph import clean_graph,get_entities

# load dataset
g_train = Graph()
g_val = Graph()
g_test = Graph()

g_train = g_train.parse('FB15k-237/train.nt', format='nt')
g_val   = g_val.parse('FB15k-237/valid.nt', format='nt')
g_test  = g_test.parse('FB15k-237/test.nt', format='nt')


# load word vectors model
word_vectors = Word2Vec.load('walks/model').wv


# remove unmatched triples
print(f"removed {clean_graph(g_train,word_vectors)} triples from training set")
print(f"removed {clean_graph(g_val,word_vectors)} triples from validation set")
print(f"removed {clean_graph(g_test,word_vectors)} triples from test set")

entities = get_entities((g_train,g_val,g_test))


model = ClassifierSimple()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if Path('rdf2vecClassfier.pth').is_file():
    print('found trained model! Loading :)')
    model.load_state_dict(torch.load('rdf2vecClassfier.pth'))
    history = pd.read_csv('log.csv')
    model = model.to(device)
else:
    model = model.to(device)
    model,history = fit_1_1(model,g_train,lambda x: word_vectors[x],3000,entities,metrics = {'acc' :torchmetrics.classification.Accuracy()},graph_eval=g_val,epochs=5)
    model.eval()
    torch.save(model.state_dict(),'rdf2vecClassfier.pth')

pd.DataFrame(history)[['acc', 'acc_val']].plot()
plt.show()
pd.DataFrame(history)[['loss','loss_val']].plot()
plt.show()