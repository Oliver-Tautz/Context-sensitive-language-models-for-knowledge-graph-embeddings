# Context-Sensitive Language Models For Knowledge Graph Embeddings

##### Train a BERT model to generate Knowledge Graph embeddings. 



## Install

## Train BERT

To train BERT a config file needs to be created. An example config can be found in [example_config.ini](src/conf/example_config.ini) . Adjust the parameters and run 

```
python src/train_bert_embeddings.py --config $CONFIG_PATH
```

After training the trained model and its training data are contained in a new subfolder with the name given in the config file.

## Generate Embeddings

To generate embeddings with a trained model, run 

```
python generate_bert_vectors.py --bert-path tiny_bert_from_scratch_simple_ep100_vec200_dataset\=MLM_mask-chance\=0.33_mask-token-chance\=0.33/ --kg-path FB15k-237/train.nt --out-path 'fb15k-237_vectors.txt'

```


## Evaluate with [pykeen](https://github.com/pykeen/pykeen)

Evaluate the embeddings in pykeen with

```
python get_vecs_and_pykeen_eval.py --bert-path tiny_bert_from_scratch_simple_ep2_vec200_dataset\=MLM_mask-chance\=0.33_mask-token-chance\=0.33/ --pykeen-train-triples FB15k-237/train.nt --pykeen-test-triples FB15k-237/test.nt
```
