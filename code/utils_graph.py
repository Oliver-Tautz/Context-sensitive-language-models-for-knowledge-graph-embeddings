import numpy as np
import gc
import itertools

from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import choose, PerfTimer
from sklearn.utils.extmath import cartesian
import torch




def get_entities(graphs):
    # get subjects and objects
    entities = []

    for g in graphs:
        entities = entities + list(g.subjects(unique=True)) + list(g.objects(unique=True))

    # pythons stupid version of nub
    entities = list(dict.fromkeys(entities))
    return entities


def get_all_corrupted_triples_fast(triple, entities, position='object'):
    # not faster ...

    s, p, o = triple

    object_augmented = [(x, y, z) for (x, y), z in itertools.product([triple[0:2]], entities)]
    subject_augmented = [(x, y, z) for x, (y, z) in itertools.product(entities, [triple[1:3]])]

    return itertools.chain(object_augmented, subject_augmented)


def get_all_corrupted_triples(triple, entities):
    # too slow ....

    s, p, o = triple
    subject_corrupted = [(s_corr, p, o) for s_corr in entities if s_corr != s]
    object_corrupted = [(s, p, o_corr) for o_corr in entities if o_corr != o]

    return subject_corrupted + object_corrupted


def get_random_corrupted_triple(triple, entities, corrupt='object'):
    """
    corrupt = one of 'subject', 'object', 'both'

    return corrupted triple with random entity
    """

    s, p, o = triple

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
        if ch == 1:
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

    return (s_corr, p, o_corr)


def clean_graph(graph, wv):
    """
    clean graph such that all triples have word vectors present in wv

    """
    no_removed = 0
    for t in graph:
        s, p, o = t
        if not str(s) in wv.key_to_index.keys() or not str(p) in wv.key_to_index.keys() or not str(
                o) in wv.key_to_index.keys():
            graph.remove(t)
            no_removed += 1
    return no_removed


def parse_rdflib_to_torch(graph,word_vec_mapping):
    entities = get_entities([graph])

    entities = np.array(entities)
    entity_vecs = torch.tensor(word_vec_mapping(np.array(entities)))
    entities = dict(zip(entities, range(len(entities))))

    predicates = np.array(list(set(graph.predicates())))
    predicate_vecs = torch.tensor(word_vec_mapping(predicates))
    predicates = dict(zip(predicates, range(len(predicates))))

    edges = []
    predicate_ix = []
    for s, p, o in np.array(graph):
        try:
            edge = (entities[s], entities[o])
            edges.append(edge)
            predicate_ix.append(predicates[p])
        except:
            print(f"Unknown entities encountered! ({s},{o})")

    edges = torch.tensor(edges)
    predicate_ix = torch.tensor(predicate_ix)

    return edges, predicate_ix, entities, predicates, entity_vecs, predicate_vecs


