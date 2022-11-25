# List of possible research directions

## Model choice

Just use Bert or test other models? I would not like to focus on this. 

## Tokenization

Simple WordLevel tokenization should be the best for the KG usecase ... **Still test others just to be sure?**

## Model Size

Small models are faster to train, but relatvion of hidden_size and number of layers to performance and training convergence could be interesting.

**Test tiny, small, medium, large(normal ...) when tiny works?**

## Embedding Training Objective

Lots of objectives are available:

### Denoising Objectives

Corruption rate can be varied for these.

* Bert-style MLM with mask token and random token
* MASS-style MLM with only mask tokens

This can be done encoder-only with token classifier head.

### Language Modeling

* Next sentence/word prediciton

Differing length of input, output possible. Direction switch or predicting middle of sentence possible.

This needs to be sequence to sequence?!

### Deshuffling

Predict original token ordering from random combination of input tokens. This needs to be sequence to sequence?!

### Link prediction

Train on link prediction with 11 or 1N sentence classifier head.

### Class prediction

supervised task with labeled dataset possible, but less interesting.

## Pretrained weights

Use pretrained weights of NLP transformer and see if it improves results? Could be bad because of tokenizer not being word level ...
