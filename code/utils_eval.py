from utils_graph import parse_rdflib_to_torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import  PerfTimer
from sklearn.utils.extmath import cartesian
import torch

from settings import VECTOR_SIZE
import gc

def compute_rank(scores, ix, mask=None):
    if mask == None:
        optimistic_rank = (scores > scores[ix]).sum() + 1
        pessimistic_rank = (scores >= scores[ix]).sum()


    else:

        optimistic_rank = ((scores > scores[ix]).index_fill(0, mask, False)).sum() + 1
        pessimistic_rank = ((scores >= scores[ix]).index_fill(0, mask, False)).sum()

    rank = (optimistic_rank + pessimistic_rank) * 0.5

    return rank


def compute_rank(scores, ix, mask=None):
    if mask == None:
        optimistic_rank = (scores > scores[ix]).sum() + 1
        pessimistic_rank = (scores >= scores[ix]).sum()


    else:

        optimistic_rank = ((scores > scores[ix]).index_fill(0, mask, False)).sum() + 1
        pessimistic_rank = ((scores >= scores[ix]).index_fill(0, mask, False)).sum()

    rank = (optimistic_rank + pessimistic_rank) * 0.5

    return rank


def get_comb_ix(edge, no_entities):
    return edge[0] * no_entities + edge[1]


def create_mask(length, ix, reverse=False):
    if not reverse:
        mask = torch.ones(length)
        mask = mask.index_fill(0, ix, 0)
    else:
        mask = torch.zeros(length)
        mask = mask.index_fill(0, ix, 1)

    return mask


def get_headmask(x, no_entities):
    # ix from x to x+1
    return torch.arange(x * no_entities, x * no_entities + no_entities)


def get_tailmask(x, no_entities):
    # ix x, x+1, x+2 ...
    return torch.arange(x, no_entities ** 2, no_entities)


def eval_ranks(model, graph,word_vec_mapping, filtered=True, batchsize=None, vecsize=VECTOR_SIZE, force_cpu=True, bit16tensors=False,
               bench=False):
    # get data from graph

    ### Setup ###
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if force_cpu:
        device = torch.device('cpu')

    if bit16tensors:
        model = model.half().to(device)
    else:
        model = model.to(device)

    perfTimer = PerfTimer()

    perfTimer.start()

    ### Preprocessing graph into torch arrays and ix
    edges, predicate_ix, entity_ix_mapping, predicate_ix_mapping, entity_vecs, predicate_vecs = parse_rdflib_to_torch(
        graph,word_vec_mapping=word_vec_mapping)

    entity_vecs = entity_vecs.to(device)
    ix_predicate_mapping = {k: v for (v, k) in predicate_ix_mapping.items()}
    ix_entity_mapping = {k: v for (v, k) in entity_ix_mapping.items()}

    no_entities = len(entity_ix_mapping.keys())

    # could also try torch.cross and torch.combinations if needed
    # all possible s_o_combinations
    s_o_combs = torch.tensor(cartesian((range(no_entities), range(no_entities))))
    # Allocate array only once

    if not batchsize:
        # Full array in RAM
        if bit16tensors:
            to_score_embeddings = torch.empty((len(s_o_combs), vecsize * 3)).half().to(device)
        else:
            to_score_embeddings = torch.empty((len(s_o_combs), vecsize * 3)).to(device)

        to_score_embeddings[:, 0:vecsize] = entity_vecs[s_o_combs[:, 0]]
        to_score_embeddings[:, 2 * vecsize:] = entity_vecs[s_o_combs[:, 1]]
    else:
        # Allocate Array of (Batchsize,dim)
        if bit16tensors:
            to_score_embeddings = torch.empty((batchsize, vecsize * 3)).half().to(device)
        else:
            to_score_embeddings = torch.empty((batchsize, vecsize * 3)).to(device)

    head_ranks = []
    tail_ranks = []

    perfTimer.track('preprocessing')

    for pred_ix in tqdm(ix_predicate_mapping.keys()):
        # loop over all predicates

        current_edges = edges[predicate_ix == pred_ix]
        number_of_real_edges = len(current_edges)

        perfTimer.track('subgraph')

        if batchsize:
            dl = DataLoader(s_o_combs, batch_size=batchsize, shuffle=False)
            perfTimer.track('dl')
            predicate_embedding = predicate_vecs[pred_ix]

            # use expand as memory of rows is shared. This may cause bugs ... investigate!
            predicate_embedding = predicate_embedding.reshape(1, vecsize).expand(batchsize, vecsize)
            perfTimer.track('expand_pediacte')

            scores = []

            for batch in tqdm(dl):
                # print(s_o_combs[:,0].type())
                # print(batch[:,0].type())
                if len(batch) != batchsize:

                    to_score_embeddings[:, 0:vecsize][0:len(batch)] = entity_vecs[batch[:, 0]]
                    to_score_embeddings[:, 2 * vecsize:][0:len(batch)] = entity_vecs[batch[:, 1]]

                    to_score_embeddings[:, vecsize:2 * vecsize] = predicate_embedding

                    perfTimer.track('copy embeddings into array')

                    batch_scores = model(to_score_embeddings)
                    perfTimer.track('predict_batch')
                    batch_scores = batch_scores[0:len(batch)]
                    scores.append(batch_scores)


                else:
                    to_score_embeddings[:, 0:vecsize] = entity_vecs[batch[:, 0]]
                    to_score_embeddings[:, 2 * vecsize:] = entity_vecs[batch[:, 1]]
                    to_score_embeddings[:, vecsize:2 * vecsize] = predicate_embedding

                    perfTimer.track('copy embeddings into array')

                    batch_scores = model(to_score_embeddings)
                    perfTimer.track('predict_batch')
                    scores.append(batch_scores)

            scores = torch.vstack(scores).squeeze()
            perfTimer.track('stack_all')

        else:

            predicate_embedding = predicate_vecs[pred_ix]
            # use expand as memory of rows is shared. This may cause bugs ... investigate!
            predicate_embedding = predicate_embedding.reshape(1, vecsize).expand(len(to_score_embeddings), vecsize)

            perfTimer.track('collect_embeddings')
            to_score_embeddings[:, vecsize:2 * vecsize] = predicate_embedding
            perfTimer.track('copy embeddings into array')

            scores = model(to_score_embeddings).squeeze()
            perfTimer.track('score_embeddings')

            #return scores, edges, predicate_embedding, pred_ix, entity_vecs, predicate_ix

        # sorted_ix = scor

        head_edge_ix = None
        tail_edge_ix = None

        for head in torch.unique(current_edges[:, 0]):

            head_edges = current_edges[current_edges[:, 0] == head]
            headmask = get_headmask(head, len(ix_entity_mapping.keys()))

            headscores = scores.index_select(0, headmask)

            if filtered:
                head_edge_ix = current_edges[:, 1]

            for edge in head_edges:
                head_rank = compute_rank(headscores, edge[1], head_edge_ix)
                head_ranks.append(head_rank)

        headscores = None
        headmask = None

        for tail in torch.unique(current_edges[:, 1]):

            tail_edges = current_edges[current_edges[:, 1] == tail]

            tailmask = get_tailmask(tail, len(ix_entity_mapping.keys()))
            tailscores = scores.index_select(0, tailmask)

            if filtered:
                tail_edge_ix = current_edges[:, 0]

            for edge in tail_edges:
                tail_rank = compute_rank(tailscores, edge[0], tail_edge_ix)
                tail_ranks.append(tail_rank)

        perfTimer.track('rank_embeddings')
        if bench:
            break
    return torch.tensor(head_ranks), torch.tensor(tail_ranks), perfTimer

def mean_rank(head_ranks,tail_ranks):
    return (torch.mean(head_ranks)+torch.mean(tail_ranks))/2

def mean_reciprocal_rank(head_ranks,tail_ranks):
    return (torch.mean(1/head_ranks)+torch.mean(1/tail_ranks))/2

def hitsAt(head_ranks,tail_ranks,at):
    return ((head_ranks <=at).sum() + (tail_ranks<=at).sum())/(len(head_ranks)+len(tail_ranks))

def eval_model(model,graph,word_vec_mapping,return_tensors=False):
    
    with torch.no_grad():
        model.eval()
        head_ranks, tail_ranks, pt = eval_ranks(model, graph, word_vec_mapping, force_cpu=True,
                                                filtered=True, bit16tensors=False)
        # collect garbage because of large arrays
        gc.collect()
        stats = dict()
        
        stats['MR'] = mean_rank(head_ranks, tail_ranks)
        stats['MRR'] = mean_reciprocal_rank(head_ranks, tail_ranks)
        stats['Hits@1'] = hitsAt(head_ranks, tail_ranks, 1)
        stats['Hits@3'] = hitsAt(head_ranks, tail_ranks, 3)
        stats['Hits@10'] = hitsAt(head_ranks, tail_ranks, 10)
        
        
        stats['MR_head'] = torch.mean(head_ranks)
        stats['MRR_head'] = torch.mean(1 / head_ranks)
        stats['Hits@1_head']  = (head_ranks <= 1).sum() / len(tail_ranks)
        stats['Hits@3_head']  = (head_ranks <= 3).sum() / len(tail_ranks)
        stats['Hits@10_head'] =( head_ranks <= 10).sum() / len(tail_ranks)
        
        stats['MR_tail'] = torch.mean(tail_ranks)
        stats['MRR_tail'] = torch.mean(1 / tail_ranks)
        stats['Hits@1_tail']  = (tail_ranks <= 1).sum() / len(tail_ranks)
        stats['Hits@3_tail']  = (tail_ranks <= 3).sum() / len(tail_ranks)
        stats['Hits@10_tail'] =( tail_ranks <= 10).sum() / len(tail_ranks)

        if not return_tensors:
            stats = {k:[v.item()] for k,v in stats.items()}

        


    return stats ,pt