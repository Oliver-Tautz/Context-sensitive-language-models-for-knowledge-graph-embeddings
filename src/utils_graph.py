import numpy as np
import gc
import itertools

from torch.utils.data import DataLoader
from tqdm import tqdm
import urllib
from utils import choose, PerfTimer
from sklearn.utils.extmath import cartesian
import torch
from rdflib import Graph
import lightrdf
from utils import iter_exception_wrapper

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


def get_random_corrupted_triple(triple, entities, corrupt='random'):
    """
    corrupt = one of 'subject', 'object', 'both'

    return corrupted triple with random path
    """

    cls, s, p, o, sep = triple

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
        ch = np.random.randint(2)

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

    return torch.tensor((cls,s_corr, p, o_corr,sep))


def clean_graph(graph, wv):
    """
    clean graph such that all input have word vectors present in wv

    """
    no_removed = 0
    for t in graph:
        s, p, o = t
        if not str(s) in wv.key_to_index.keys() or not str(p) in wv.key_to_index.keys() or not str(
                o) in wv.key_to_index.keys():
            graph.remove(t)
            no_removed += 1
    return no_removed



def parse_kg(path):

    graph = Graph()
    graph.parse(path)
    entities = get_entities([graph])

    entities = np.array(entities)
    entities = dict(zip(entities, range(len(entities))))

    predicates = np.array(list(set(graph.predicates())))
    predicates = dict(zip(predicates, range(len(predicates))))

    edges = []
    predicate_ix = []

    for s, p, o in np.array(graph):
        try:
            edge = (entities[s], entities[o])
            edges.append(edge)
            predicate_ix.append(predicates[p])
        except:
            print(f"Unknown entities or predicates encountered! ({s},{p},{o})")
    # return numpy arrays. Maybe change to tensors?
    return np.array(list(entities.keys())), np.array(list(predicates.keys())), np.array(edges), np.array(predicate_ix)


def parse_kg_to_torch(graph,word_vec_mapping):
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


def parse_kg_fast(path, filterpath=None):
    entities = dict()
    predicates = dict()
    edges = []
    predicate_ix = []

    parser = lightrdf.Parser()

    if filterpath:
        filterset = get_filterset(filterpath)

    f = open(path, 'rb')

    for e1, r, e2 in tqdm(iter_exception_wrapper(parser.parse(f, format='nt'))):

        e1 = get_entity_example(e1)
        e2 = get_entity_example(e2)
        r = get_entity_example(r)

        if filterpath:
            if not (e1 in filterset or e2 in  filterset):
                continue

        if e1 in entities:
            e1_ix = entities[e1]
        else:
            e1_ix = len(entities)
            entities[e1] = e1_ix

        if e2 in entities:
            e2_ix = entities[e2]
        else:
            e2_ix = len(entities)
            entities[e2] = e2_ix

        if r in predicates:
            r_ix = predicates[r]
        else:
            r_ix = len(predicates)
            predicates[r] = r_ix

        edges.append((e1_ix, e2_ix))
        predicate_ix.append(r_ix)
    f.close()
    entities = torch.tensor(entities.keys())
    predicates = torch.tensor(predicates.keys())
    edges = torch.tensor(edges)
    predicate_ix = torch.tensor(predicate_ix)

    return entities, predicates, edges, predicate_ix

def get_entity(url):
    return urllib.parse.urlparse(url.strip()).path
def get_entity_example(url):
    return f"http://example.org{get_entity(url)}"


def get_filterset(path):
    entity_file = open(path, 'r')
    entities = entity_file.readlines()
    for i, e in enumerate(entities):
        parse_result = urllib.parse.urlparse(e.strip())
        if not parse_result.scheme:
            parse_result = urllib.parse.urlparse(e.strip()[1:-1])

        entities[i] = f"http://example.org{parse_result.path}"
    return set(entities)

def graph_to_string(path,filterfilepath = None):
    parser = lightrdf.Parser()
    f = open(path, 'rb')
    tp = []
    skipped = 0
    taken = 0

    if filterfilepath:
        filterset = get_filterset(filterfilepath)

    for e1, r, e2 in tqdm(iter_exception_wrapper(parser.parse(f, format='nt', base_iri=None))):


        e1 = get_entity_example(e1)
        e2 = get_entity_example(e2)
        r = get_entity_example(r)


        if not (e1 in filterset or e2 in filterset):
            skipped+=1
            continue
        t = [get_entity_example(x[1:-1]) for x in [e1,r,e2]]
        tp.append(' '.join(t))
        taken +=1

    print(f"filtered {skipped / (skipped+taken)} of the dataset.")
    return tp
