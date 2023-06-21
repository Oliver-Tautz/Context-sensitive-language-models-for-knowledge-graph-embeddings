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


def train_rdf2vec_classifier(trainpath='FB15K-237/train.nt',valpath='FB15K-237/valid.nt',rdf2vecmodel='walks/model'):
    # load dataset_file

    g_train = Graph()
    g_val = Graph()


    g_train = g_train.parse(trainpath, format='nt')
    g_val   = g_val.parse(valpath, format='nt')


    # load word vectors model
    word_vectors = Word2Vec.load(rdf2vecmodel).wv


    # remove unmatched input
    print(f"removed {clean_graph(g_train,word_vectors)} triples from training set")
    print(f"removed {clean_graph(g_val,word_vectors)} triples from validation set")


    entities = get_entities((g_train,g_val))


    model = ClassifierSimple()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if Path('rdf2vecClassfier.pth').is_file():
        print('found trained model! Loading :)')
        model.load_state_dict(torch.load('rdf2vecClassfier.pth'))
        history = pd.read_csv('log.csv')
        model = model.to(device)
    else:
        model = model.to(device)
        model,history = fit_1_1(model,g_train,lambda x: word_vectors[x],3000,entities,metrics = {'acc' :torchmetrics.classification.Accuracy()},graph_eval=g_val,epochs=CLASSIFIER_EPOCHS)
        model.eval()
        torch.save(model.state_dict(),'rdf2vecClassfier.pth')

    return model, history
