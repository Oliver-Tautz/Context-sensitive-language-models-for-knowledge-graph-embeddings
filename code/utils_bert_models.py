import numpy as np
import torch

def get_embeddings(entities,bert_model,tokenizer):
    entities = [tokenizer.encode(x) for x in np.array(entities)]
    embeddings = bert_model(torch.tensor(entities))
    embeddings = embeddings['last_hidden_state'][:,1]
    return embeddings