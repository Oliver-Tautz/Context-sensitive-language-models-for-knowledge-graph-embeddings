from rdflib import Graph, URIRef
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph
from utils import *
from rdflib import Graph
from gensim.models import Word2Vec
import pandas as pd
import zipfile
from settings import *
import subprocess
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tqdm import tqdm, trange
from os.path import exists
from pathlib import Path
import pandas as pd

subprocess.run(['wget', '-qnc', '--no-check-certificate',
                "https://docs.google.com/uc?export=download&id=1pBnn8bjI2VkVvBR33DnvpeyocfDhMCFA", '-O',
                'fb15k-237_nt.zip'])
subprocess.run(['unzip' ,'-n', 'fb15k-237_nt.zip'])

if JRDF2VEC_TRAINING:
    subprocess.run(
        ['wget', '-nc', 'https://raw.githubusercontent.com/dwslab/jRDF2Vec/jars/jars/jrdf2vec-1.3-SNAPSHOT.jar'])
    subprocess.run(['java', '-jar', 'jrdf2vec-1.3-SNAPSHOT.jar', '-checkInstallation'])
    subprocess.run(['rm', '-rf', 'walks'])
    subprocess.run(['java', '-jar', 'jrdf2vec-1.3-SNAPSHOT.jar', '-walkDirectory walks', '-graph', 'FB15k-237/train.nt', '-threads', f'{JRDF2VEC_THREADS}' , '-dimension' , f'{VECTOR_SIZE}'])
# Load graphs
g_train = Graph()
g_val = Graph()
g_test = Graph()

g_train = g_train.parse('FB15k-237/train.nt', format='nt')
g_val = g_val.parse('FB15k-237/valid.nt', format='nt')
g_test = g_test.parse('FB15k-237/test.nt', format='nt')

# load rdf2vec model
word_vectors = Word2Vec.load('walks/model').wv

# clean graph
print(f"removed {clean_graph(g_train, word_vectors)} triples from training set")
print(f"removed {clean_graph(g_val, word_vectors)} triples from validation set")
print(f"removed {clean_graph(g_test, word_vectors)} triples from test set")

# collect entities
entities = get_entities((g_train, g_val, g_test))

# construct training dataset
X, Y = get_1_1_dataset(g_train, entities, lambda x: word_vectors[x])
x_val, y_val = get_1_1_dataset(g_val, entities, lambda x: word_vectors[x])
x_test, y_test = get_1_1_dataset(g_test, entities, lambda x: word_vectors[x])

print(f"training datapoints = {len(X)}")
print(f"validation datapoints = {len(x_val)}")
print(f"test datapoints = {len(x_test)}")

#### Test basi ml models

# %%
if TEST_CLASSIC_ML:
    # 0.6 is pretty bad
    LR = sklearn.linear_model.LogisticRegression(max_iter=1000)
    test_sklearn_model(LR, X, Y, x_test, y_test, 10000)

    # this works pretty well out of the box
    randomforest = sklearn.ensemble.RandomForestClassifier()
    test_sklearn_model(randomforest, X, Y, x_test, y_test, 10000)

# Train and eval Tensorflow model


#### Neural Network with a single hidden layer
model = models.Sequential()
model.add(layers.Flatten())
model.add(layers.Dropout(.3))
model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(128, activation='relu'))
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
        X, Y = get_1_1_dataset(g_train, entities, lambda x: word_vectors[x])
        histories.append(model.fit(X, Y, batch_size=3000, epochs=1, validation_data=(x_val, y_val), ))

    model.save_weights(modelname)
    h = merge_historires(histories)
    pd.DataFrame(h).to_csv(modelname.parent / 'training_log.csv')

# %%


pd.DataFrame(h)[['loss', 'val_loss']].plot()
pd.DataFrame(h)[['accuracy', 'val_accuracy']].plot();

# %%
evalu = model.evaluate(x_test, y_test, batch_size=256)
print(evalu)

# %%


scores = evaluate_link_pred(lambda x: tf.keras.activations.sigmoid(model.predict(x)), g_val,
                            lambda x: map_keyed_vectors(word_vectors, x), entities, max_triples=None)

scores = evaluate_link_pred(lambda x: tf.keras.activations.sigmoid(model.predict(x)), g_test,
                            lambda x: map_keyed_vectors(word_vectors, x), entities, max_triples=None)

reduction_model = sklearn.manifold.TSNE(learning_rate='auto', init='pca').fit_transform

# %%
X, Y = get_1_1_dataset(g_train, entities, lambda x: word_vectors[x], corrupt='both')

plot_embeddings(X, Y)

# %%
X, Y = get_1_1_dataset(g_train, entities, lambda x: word_vectors[x], corrupt='subject')
plot_embeddings(X, Y)

# %%
X, Y = get_1_1_dataset(g_train, entities, lambda x: word_vectors[x], corrupt='object')
plot_embeddings(X, Y)
