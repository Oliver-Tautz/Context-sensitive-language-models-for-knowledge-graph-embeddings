# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: std-env
#     language: python
#     name: std-env
# ---

# %% [markdown]
# ## Download and check rdf2vec

# %%
training = 0
VECTOR_SIZE=100
test_classic_ml = False
CLASSIFIER_EPOCHS = 50

# %%
# download jrdf2vec
# ! wget -nc https://raw.githubusercontent.com/dwslab/jRDF2Vec/jars/jars/jrdf2vec-1.3-SNAPSHOT.jar

# %%
# Check jrdf2vec installation. Should print "=> Everything is installed. You are good to go!" somewhere
if training:
    pass
    # ! java -jar jrdf2vec-1.3-SNAPSHOT.jar -checkInstallation

# %% [markdown]
# ## Download Dataset
#

# %%
# download .nt dataset from my drive
# ! wget -q --no-check-certificate 'https://docs.google.com/uc?export=download&id=1pBnn8bjI2VkVvBR33DnvpeyocfDhMCFA' -O fb15k-237_nt.zip

# ! unzip -o fb15k-237_nt.zip

# %% [markdown]
# ## Generate walks and train rdf2vec

# %% tags=[]
# generate walks and train rdf2vec on train set with default parameters
if training:
    # ! rm -rf walks
    # ! java -jar jrdf2vec-1.3-SNAPSHOT.jar -walkDirectory walks -graph FB15k-237/train.nt -threads 10 -dimension 100

# %%
import numpy as np


def get_entities(graphs):
    # get subjects and objects
    entities = []
    
    for g in graphs:
        entities = entities + list(g.subjects(unique=True)) + list(g.objects(unique=True))

    # pythons stupid version of nub
    entities = list(dict.fromkeys(entities))
    return entities

def get_all_corrupted_triples_fast(triple,entities,position = 'object'):
    # not faster ...

    s,p,o = triple

    object_augmented = [(x,y,z) for  (x,y), z in itertools.product([triple[0:2]],entities)]
    subject_augmented =[(x,y,z) for  x, (y,z) in itertools.product(entities,[triple[1:3]])]
    
    
    return itertools.chain(object_augmented , subject_augmented)

def get_all_corrupted_triples(triple,entities):
    #too slow ....
    
    s,p,o = triple
    subject_corrupted = [(s_corr,p,o) for s_corr in entities if s_corr != s]
    object_corrupted = [(s,p,o_corr)   for o_corr in entities if o_corr != o]

    return subject_corrupted + object_corrupted


    

def choose_many_multiple(arrs,n):
    l = len(arrs[0])
    for a in arrs:
        assert len(a) == l, 'Arres not of same length ! :('
        
    
    ix = np.random.choice(range(len(a)),n)
    
    return [np.array(a)[ix] for a in arrs]
    
def choose_many(a,n):
    ix = np.random.choice(range(len(a)),n)
    return np.array(a)[ix]
    
def choose(a):

    L = len(a)

    i = np.random.randint(0,L)

    return a[i]

def get_random_corrupted_triple(triple,entities, corrupt='object'):
    """
    corrupt = one of 'subject', 'object', 'both'
    
    return corrupted triple with random entity
    """

    s,p,o = triple
    
    # set up as the same
    s_corr = s
    o_corr = o
    
    if corrupt == 'subject':  
        # corrupt only the subject
        while s_corr == s:
            s_corr = choose(entities)  
    elif corrupt == 'object':
        # corrupt only the object
        while o_corr == o:
            o_corr = choose(entities)  
    elif corrupt == 'random':
        # corrupt one or both randomly
        ch = np.random.randint(3)
        
        if ch == 0:
            while s_corr == s:
                s_corr = choose(entities)  
        if ch == 1 :
            while o_corr == o:
                o_corr = choose(entities)  
        if ch == 2:
            while s_corr == s or o_corr == o:
                s_corr = choose(entities)  
                o_corr = choose(entities) 
    else:
        while s_corr == s or o_corr == o:
            s_corr = choose(entities)  
            o_corr = choose(entities) 
            
    
    return (s_corr,p,o_corr)
    
def merge_historires(history_list):
    h = {}
    for key in history_list[0].history.keys():
        h[key] = [h.history[key][0] for h in histories]
    return h    


def clean_graph(graph,wv):
    """
    clean graph such that all triples have word vectors present in wv
    
    """
    no_removed = 0 
    for t in graph:
        s,p,o = t
        if not str(s) in wv.key_to_index.keys() or not str(p) in wv.key_to_index.keys() or not str(o) in wv.key_to_index.keys():
            graph.remove(t)
            no_removed+=1
    return no_removed
    
    
def get_vectors_fast(triples,entity_vec_mapping,vector_size=VECTOR_SIZE):
    # ~20-30% faster
    X = np.array(triples)
    X = word_vectors[X.flatten()].reshape(len(triples),vector_size*3)
    
    return X    

def get_vectors(triples,entity_vec_mapping,vector_size=200):
    X = np.array(triples)
    X = [(entity_vec_mapping(x[0]), entity_vec_mapping(x[1]),entity_vec_mapping(x[2])) for x in X]
    X = [np.concatenate(x) for x in X]
    X = np.vstack(X).astype(np.float64)
    
    return X

def get_1_1_dataset(graph, entities,entity_vec_mapping,corrupt='random'):
    
    original_triple_len = len(graph)
    # get triples
    X = list(graph)
    no_t = len(X)
    

    
    corrupted_triples = [get_random_corrupted_triple(x,entities,corrupt=corrupt) for x in X]
    X = X + corrupted_triples
    
    

    # convert uris to strings
    
    X = get_vectors_fast(X,entity_vec_mapping)
    
    # stack them

    Y = np.concatenate((np.ones(no_t),np.zeros(no_t))).astype(np.uint8)
    
    return X, Y

def test_sklearn_model(model,X,Y,x_test,y_test,subset=10000):
    

  
    
    ix = np.random.choice(range(len(X)),size=subset)
    
    scaler = preprocessing.StandardScaler().fit(X)
    
    X_scaled = scaler.transform(X[ix])
    model.fit(X_scaled,Y[ix])

    print(f'train_score ={model.score(scaler.transform(X),Y)}')    
    print(f'test_score ={model.score(scaler.transform(x_test),y_test)}')

def scale_and_predict(model,x):
    x = preprocessing.StandardScaler().fit_transform(x)
    return model.predict(x)


# %% [markdown]
# ## Parse Graph

# %%
from rdflib import Graph, URIRef
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph
import networkx as nx



g_train = Graph()
g_val = Graph()
g_test = Graph()

g_train = g_train.parse('FB15k-237/train.nt', format='nt')
g_val   = g_val.parse('FB15k-237/valid.nt', format='nt')
g_test  = g_test.parse('FB15k-237/test.nt', format='nt')


# %% [markdown]
# ## Plot Graph (not working)

# %%
# taken from https://stackoverflow.com/questions/39274216/visualize-an-rdflib-graph-in-python
# takes way too long ... use subgraph?!
import matplotlib.pyplot as plt
plot = False
if plot:
    G = rdflib_to_networkx_multidigraph(g_train[0:100])


    pos = nx.spring_layout(G, scale=2)
    edge_labels = nx.get_edge_attributes(G, 'r')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    nx.draw(G, with_labels=True)

    #if not in interactive mode for 
    plt.show()
    

# %%
from gensim.models import Word2Vec


word_vectors = Word2Vec.load('walks/model').wv

# %%
import timeit

# %%
word_vectors['http://example.org/award/award_nominee/award_nominations./award/award_nomination/award_nominee']

# %%
entities = np.array(entities)

# %%
timeit.timeit('word_vectors[entities]', number=1000,globals=globals())

# %%
timeit.timeit('[word_vectors.get_vector(x) for x in entities]', number=1000,globals=globals())


# %%
def map_keyed_vectors(word_vectors, iterable):
    
    return (word_vectors.get_vector(x) for x in iterable)


# %%
# clean graphs 
# number of triples removed should be low, a few hundred
print(f"removed {clean_graph(g_train,word_vectors)} triples from training set")
print(f"removed {clean_graph(g_val,word_vectors)} triples from validation set")
print(f"removed {clean_graph(g_test,word_vectors)} triples from test set")

entities = get_entities((g_train,g_val,g_test))

# %%
X, Y = get_1_1_dataset(g_train,entities,lambda x : word_vectors[x])
x_val , y_val= get_1_1_dataset(g_val,entities,lambda x : word_vectors[x])
x_test , y_test= get_1_1_dataset(g_test,entities,lambda x : word_vectors[x])

print(f"training datapoints = {len(X)}")
print(f"validation datapoints = {len(x_val)}")
print(f"test datapoints = {len(x_test)}")

# %%
# test some simple baselines\
import sklearn.linear_model
import sklearn.ensemble
from sklearn import preprocessing



# %%
if test_classic_ml:
    # 0.6 is pretty bad
    LR = sklearn.linear_model.LogisticRegression(max_iter=1000)
    test_sklearn_model(LR,X,Y,x_test,y_test,10000)


# %%
if test_classic_ml:
    # this works pretty well out of the box
    randomforest = sklearn.ensemble.RandomForestClassifier()
    test_sklearn_model(randomforest,X,Y,x_test,y_test,10000)

# %%
import tensorflow as tf
#### Neural Network with a single hidden layer
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tqdm import tqdm, trange
from os.path import exists
from pathlib import Path
import pandas as pd
model = models.Sequential()
model.add(layers.Flatten())
model.add(layers.Dropout(.3))
model.add(layers.Dense(64, activation='relu'))
#model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1))




model.compile(optimizer=keras.optimizers.Adam(),  # Optimizer
            # Loss function to minimize
            loss=tf.keras.losses.BinaryCrossentropy(
    from_logits=True,
    label_smoothing=0.0,
    axis=-1,
    reduction="auto",
    name="binary_crossentropy",
)
,
            # List of metrics to monitor
            metrics=['accuracy'])

modelname = Path(f'./rdf2vec_classifier/trained_for={CLASSIFIER_EPOCHS}/rdf2vec_classifier')

try:
    model.load_weights(modelname) 
    h = pd.read_csv(modelname.parent / 'training_log.csv')
    print("trained weights found! not training again :)")
except:


    histories = []

    for epochs in trange(CLASSIFIER_EPOCHS):
        X, Y = get_1_1_dataset(g_train,entities,lambda x : word_vectors[x])
        histories.append(model.fit(X, Y, batch_size=3000, epochs=1, validation_data=(x_val,y_val),))


    model.save_weights(modelname) 
    h = merge_historires(histories)
    pd.DataFrame(h).to_csv(modelname.parent / 'training_log.csv')


# %%
import pandas as pd
pd.DataFrame(h)[['loss','val_loss']].plot()
pd.DataFrame(h)[['accuracy','val_accuracy']].plot();

# %%
evalu = model.evaluate(x_test, y_test, batch_size=256)
print(evalu)

# %%
print(len(entities))

# %%
word_vectors['']

# %%
import time
from tqdm import tqdm
from collections import defaultdict
from itertools import chain
import pickle

def evaluate_link_pred(score_f,graph,entity_vec_mapping,entities,vector_size = 100, max_triples=100, plot = False):
    print('start_evaluating ...')
    
    entities = np.array(entities)
    
    stats = {'generate_time' : [],
        'convert_time:': [],
        'score_time': [],
        'sort_time': [],
        'total_time': []}

    ranks = []
    
    
    # slightly complicated defaultdict trick. Uninitialised 2dim matrix. 
    # Only store what was seen. Return None for other values

    embeddings_scores = defaultdict(lambda: defaultdict(lambda :None))

    
    graph = np.array(graph)
    
    if max_triples:
        graph = choose_many(graph,max_triples)
    
    
    for i,tp in enumerate(tqdm(np.array(graph))):
                                   
        s,p,o = tp
        start_timer = time.perf_counter()
        # construct corrupted triples

        triples_to_test = np.array([tp]+get_all_corrupted_triples(tp,entities))
        
        
        

        generate_timer = time.perf_counter()
        
        if p in embeddings_scores.keys():
            
            # init
            lookup = embeddings_scores[p]
            lookup_scores = []
            ix_to_score = []
            
            # find scores and triple not scored
            for j,ttt in enumerate(triples_to_test):
                s,p,o = ttt
                
                if (s,o) in lookup.keys():
                    l = lookup[(s,o)]
                    lookup_scores.append((l,j))
                else:

                    ix_to_score.append(j)
            #print(pd.DataFrame(triples_to_test[ix_to_score]))
            pd.DataFrame(triples_to_test[ix_to_score], columns = ['s','p','o']).to_feather(f'debug/not_found_{i}.feather')
            pd.DataFrame(lookup.keys(), columns = ['s','o']).to_feather(f'debug/keys_{i}.feather')
            pd.DataFrame([[len(lookup_scores),len(triples_to_test),len(lookup_scores)/len(triples_to_test)]], columns = ['n_looked_up','n','%']).to_feather(f'debug/stats_{i}.feather')
            
            print(f"looked up {len(lookup_scores)} of {len(triples_to_test)} triples")
            # score unscored triples
            triples_to_score = triples_to_test[ix_to_score]
            if len(triples_to_score) > 0:
                vectors_to_score = get_vectors_fast(triples_to_score,entity_vec_mapping,vector_size=vector_size)
                convert_timer = time.perf_counter()
                scores = score_f(vectors_to_score).numpy().squeeze()

                # add scores to lookup
                for ttt, score in zip(triples_to_score,scores):
                    s, _ ,o = tp
                    embeddings_scores[p][(s,o)] = score
            else:
                scores =[]
                
            score_timer = time.perf_counter()
            
            # combine scores

            
            scores =  lookup_scores + list(zip(scores,ix_to_score))
 
            scores = np.array(scores)

            # sort by score
            scores = scores[np.argsort(scores[:,0])]
            
            # get rank
            # look at col 1 (ix) and find original triple ix.
            
            rank = len(scores) - np.where(scores[:,1] == 0)[0][0]
            sort_timer = time.perf_counter()
            
           
            
            
            
                
        else:
            vectors_to_test = get_vectors_fast(triples_to_test,entity_vec_mapping,vector_size=vector_size)
            convert_timer = time.perf_counter()

                # score them
            scores = (score_f(vectors_to_test)).numpy().squeeze()
            
            for tp, score in zip(triples_to_test,scores):
                    s,p,o = tp
                    embeddings_scores[p][(s,o)] = score                        

            if plot:
                fig = plot_embeddings(vectors_to_test,scores)
                fig.show()

            score_timer = time.perf_counter()
            # sort them 
            sort_ix = np.argsort(scores ,axis=0)
            rank =  len(sort_ix) - np.where(sort_ix == 0)[0][0]
         
        
            sort_timer = time.perf_counter()

        

        stats['generate_time'].append(generate_timer - start_timer)
        stats['convert_time:'].append(convert_timer -  generate_timer)
        stats['score_time'].append(score_timer-convert_timer)
        stats['sort_time'].append(sort_timer-score_timer)
        stats['total_time'].append(sort_timer-start_timer)
        ranks.append(rank)
        
        if i % 100 == 0:
            print(np.mean(rank),'\n',{key: np.mean(value) for key,value in list(stats.items())[-100:]})
            pass
            
        
    stats =  {key: np.mean(value) for key,value in stats.items()}
    
    ranks = np.array(ranks)
    
    stats['MR'] = np.mean(ranks)
    stats['MRR'] = np.mean(1/(ranks))
    stats['HITS1'] = np.count_nonzero(ranks <= 1)
    stats['HITS3'] = np.count_nonzero(ranks <= 3)
    stats['HITS10'] = np.count_nonzero(ranks <= 10)
        
    return  ranks, stats, embeddings_scores
    
        
        
        


# %% jupyter={"outputs_hidden": true} tags=[]
#ranks, stats, table = evaluate_link_pred(lambda x : tf.keras.activations.sigmoid(model.predict(x)), testgraph, lambda x : word_vectors[x], testentities, max_triples=None)    
#z, stats

# %%
#scores = evaluate_link_pred(lambda x : tf.keras.activations.sigmoid(model.predict(x)), testgraph, lambda x : map_keyed_vectors(word_vectors,x), entities, max_triples=None)    
#z, stats

# %%
scores = evaluate_link_pred(lambda x : tf.keras.activations.sigmoid(model.predict(x)), g_val, lambda x : map_keyed_vectors(word_vectors,x), entities, max_triples=None)    
#z, stats

# %%
def square(x):
    x**2
    
    

# %%
timeit.timeit('square(12345)', number=10000000,globals=globals())

# %%
timeit.timeit('(lambda x: x**2)(12345)', number=10000000,globals=globals())

# %%

# %%
scores = evaluate_link_pred(lambda x : tf.keras.activations.sigmoid(model.predict(x)), g_test, lambda x : word_vectors[x], entities, max_triples=None)    
#z, stats

# %%
scores = evaluate_link_pred(lambda x : tf.keras.activations.sigmoid(model.predict(x)), g_train, lambda x : word_vectors[x], entities, max_triples=None)    
#z, stats

# %% [markdown]
# # Plot the embeddings 

# %%
X, Y = get_1_1_dataset(g_train,entities,lambda x : word_vectors[x],corrupt='both')
#x_val , y_val= get_1_1_dataset(g_val,entities,lambda x : word_vectors[x])
#x_test , y_test= get_1_1_dataset(g_test,entities,lambda x : word_vectors[x])

x,y = X, Y
#for x,y in [(X,Y),(x_val,y_val),(x_test,y_test)]:
    

# %%
import sklearn.manifold
import pandas as pd
import plotly.express as px

reduction_model = sklearn.manifold.TSNE(learning_rate='auto',init='pca').fit_transform


# %%
def plot_embeddings(x,y,reduction_model=sklearn.manifold.TSNE(learning_rate='auto',init='pca').fit_transform,number_of_examples=10000):
"""
x = multi dim array (SAMPLES,EMBEDDING_DIM)
y = one-dim array (SAMPLES,)
"""
    x,y = choose_many_multiple([x,y],number_of_examples)
    x_reduced=reduction_model(x)

    plot_data = np.concatenate((x_reduced,np.expand_dims(y,1)),1)

    df = pd.DataFrame(plot_data,columns=list(range(x_tsne.shape[1]))+['label'])


    fig = px.scatter(df, x=0, y=1, color="label")
    return fig

# %%
Y.shape

# %%
X, Y = get_1_1_dataset(g_train,entities,lambda x : word_vectors[x],corrupt='both')

plot_embeddings(X,Y)

# %%
X, Y = get_1_1_dataset(g_train,entities,lambda x : word_vectors[x],corrupt='subject')
plot_embeddings(X,Y)

# %%
X, Y = get_1_1_dataset(g_train,entities,lambda x : word_vectors[x],corrupt='object')
plot_embeddings(X,Y)

# %% [markdown]
# ## Get walks into ram :)

# %%
import gzip


f=gzip.open('walks/walk_file_0.txt.gz','rb')
lines = [x.decode('utf-8') for x in f.readlines()]

# %%
lines[0]
