# Some Ideas how to train and evaluate Bert

## Tokenization

Which tokenizer makes sense? See https://huggingface.co/docs/transformers/tokenizer_summary

Many choices, BPE, WordPiece, Unigram ...

Currently just use WordLevel for the time.

**If tokenizer splits entities into subwords, bert embeddings will be either 'sentence embeddings' of one-word sentence or some_mean(token_embeddings). Makes postprocessing more difficult.**

## Training Task

Mostly from T5 https://arxiv.org/abs/1910.10683

## Unsupervised denoising

![]('T5_unsupervised_objectives.png')



#### Bert-Style MlM :

* Corrupt 15% of input tokens. 
* 90% of the corrupted tokens are replaced with out-of-alphabet masking token
* 10% of corrupted tokens are replaced with random tokens

predict original text.


#### Denoising discussion:

* Overall, the most significant difference in performance we observed was that denoising objectives outperformed language modeling and deshuffling for pre-training.
* We did not observe a remarkable difference across the many variants of the denoising objectives we explored.
* This implies that choosing among the denoising objectives we considered here should mainly be done according to their computational cost.

## Multi-Task Learning

* ... Use multiple training objectives or datasets at once, i.e. train for a few epochs on one, then the other, repeat.
* Look into this maybe?

## Other Unsupervised Tasks

### Next Sentence prediction

From Bert https://arxiv.org/abs/1810.04805

Use two sentences as [CLS],t1,t2,...,tn,[SEP],t21,t22,...,t2k,[SEP] and mask the second one. Then predict the original

* This seems promising for KG sentences as it captures bigger relations then one triple.
* Could even be done with longer sequences, or randomly long sequences of triples.


### Sentence boundary disambiguation

https://en.wikipedia.org/wiki/Sentence_boundary_disambiguation

Own Idea. Give in sequence without [SEP] tokens and predict their location.

### Link prediction

Use Link prediction as 'unsupervised' objective directly.

* Could be done with 1-1 or 1-N dataset.


### 
