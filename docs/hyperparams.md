# Dataset Hyperparams

options:
* triples only
* walks of differing length
  
for walks:
* depth = 2..6
* number of walks per entity = 100
* sample algorithm =  RANDOM_WALKS_NO_DUPLICATE

# Bert Hyperparams
## Model
* vector_size = 200
* dropout classifier = 0.25
* dropout hidden layers = 0.1
* hidden_layers=2

## Training
* epochs = <1000
* batchsize= 1000
* learning_rate = ?
* options: MLM, MASS, LM, LP

MLM:
* mask_chance = 0.33
* mask_token_chance = 1 (==MASS with mask_token_chance = 1)

Early Stopping:
* patience = 8
* delta = 0.005

