from tqdm import trange, tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
import torch
import torchmetrics
import numpy as np

from settings import VECTOR_SIZE
from utils import choose
from utils_graph import get_random_corrupted_triple


def get_vectors_fast(triples, entity_vec_mapping, vector_size=VECTOR_SIZE):
    # ~20-30% faster
    X = np.array(triples)
    X = entity_vec_mapping(X.flatten()).reshape(len(triples), vector_size * 3)

    return X


def get_1_1_dataset(graph, entities, entity_vec_mapping, corrupt='random'):
    original_triple_len = len(graph)
    # get triples
    X = list(graph)
    no_t = len(X)

    corrupted_triples = [get_random_corrupted_triple(x, entities, corrupt=corrupt) for x in X]
    X = X + corrupted_triples

    # convert uris to strings

    X = get_vectors_fast(X, entity_vec_mapping)

    # stack them

    Y = np.concatenate((np.ones(no_t), np.zeros(no_t))).astype(np.uint8)

    return X, Y


def get_random_corrupted_triple_embedded(triple, entities, corrupt='object', vector_size=VECTOR_SIZE):
    """
    corrupt = one of 'subject', 'object', 'both'

    return corrupted triple with random entity
    """

    s = triple[0:VECTOR_SIZE]
    p = triple[VECTOR_SIZE:VECTOR_SIZE * 2]
    o = triple[VECTOR_SIZE * 2:]

    # set up as the same
    s_corr = s[:]
    o_corr = o[:]

    if corrupt == 'subject':

        # corrupt only the subject
        while (s_corr == s).all():
            s_corr = choose(entities)
    elif corrupt == 'object':
        # corrupt only the object
        while (o_corr == o).all():
            o_corr = choose(entities)
    elif corrupt == 'random':
        # corrupt one or both randomly
        ch = np.random.randint(3)

        if ch == 0:
            while (s_corr == s).all():
                s_corr = choose(entities)
        if ch == 1:
            while (o_corr == o).all():
                o_corr = choose(entities)
        if ch == 2:

            while (s_corr == s).all() or (o_corr == o).all():
                s_corr = choose(entities)
                o_corr = choose(entities)
    else:

        while (s_corr == s).all() or (o_corr == o).all():
            s_corr = choose(entities)
            o_corr = choose(entities)

    return np.concatenate((s_corr, p, o_corr), axis=0)


def get_1_1_dataset_embedded(graph, entities, corrupt='random', vector_size=VECTOR_SIZE):
    """
    graph: numpy array of shape (samples,3*
    """

    if len(graph.shape) == 1:
        graph = np.expand_dims(graph, 0)
    # print(graph.shape)
    no_t = len(graph)
    corrupted_triples = [get_random_corrupted_triple_embedded(x, entities, corrupt=corrupt, vector_size=vector_size) for
                         x in graph]
    X = np.concatenate((graph, corrupted_triples), axis=0)

    Y = np.concatenate((np.ones(no_t), np.zeros(no_t))).astype(np.uint8)

    return X, Y


class Dataset11(torch.utils.data.Dataset):
    def __init__(self, graph, vec_mapping, entities, corrupt='random', vector_size=VECTOR_SIZE):
        """
        graph: graph to train on
        vec_mapping: function that returns vectos from URIs
        entities: iterable of all entities to build fake triples
        """

        self.entities = vec_mapping(np.array(entities))
        self.graph = get_vectors_fast(graph, vec_mapping)
        self.len = len(graph)
        self.vec_mapping = vec_mapping
        self.corrupt = corrupt
        self.vector_size = vector_size

    def __len__(self):
        return self.len

    def __getitem__(self, ix):
        return get_1_1_dataset_embedded(self.graph[ix], self.entities, self.corrupt, self.vector_size)


def get_1_1_dataset(graph, entities, entity_vec_mapping, corrupt='random'):
    original_triple_len = len(graph)
    # get triples
    X = list(graph)
    no_t = len(X)

    corrupted_triples = [get_random_corrupted_triple(x, entities, corrupt=corrupt) for x in X]
    X = X + corrupted_triples

    # convert uris to strings

    X = get_vectors_fast(X, entity_vec_mapping)

    # stack them

    Y = np.concatenate((np.ones(no_t), np.zeros(no_t))).astype(np.uint8)

    return X, Y


def fit_1_1(model, graph, word_vec_mapping, batch_size, entities, metrics=None, epochs=50, optimizer=None,
            lossF=torch.nn.BCEWithLogitsLoss(), graph_eval=None):
    """
    metrics: dictionary of {name:torchmetric} to track for training and evaluation.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # move metrics to device
    metrics = {name: metric.to(device) for (name, metric) in metrics.items()}

    # move model to device

    model = model.to(device)

    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters())

    history = defaultdict(list)
    loss_metric = torchmetrics.aggregation.MeanMetric().to(device)

    dataset = Dataset11(graph, word_vec_mapping, entities, 'random')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if graph_eval:
        dataset_eval = Dataset11(graph_eval, word_vec_mapping, entities, 'random')
        dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False)

    for ep in trange(epochs):
        # train
        for X, Y in dataloader:
            model.train()

            # shape is (bs,2,3*VECTORSIZE) because of dataset implementation
            # so I need to flatten the arrays
            X = torch.flatten(X, 0, 1)
            Y = torch.flatten(Y)

            X = X.to(device)
            Y = Y.to(device).double()

            predictions = model(X).squeeze()
            loss = lossF(predictions, Y)
            loss_metric(loss)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            for metric in metrics.values():
                metric(predictions, Y.long())

        for name, metric in metrics.items():
            history[name].append(metric.compute())
            metric.reset()
        history['loss'].append(loss_metric.compute())
        loss_metric.reset()

        # eval
        if graph_eval:
            with torch.no_grad():
                for X, Y in dataloader:
                    model.eval()

                    # shape is (bs,2,3*VECTORSIZE) because of dataset implementation
                    # so I need to flatten the arrays
                    X = torch.flatten(X, 0, 1)
                    Y = torch.flatten(Y)

                    X = X.to(device)
                    Y = Y.to(device).double()

                    predictions = model(X).squeeze()
                    loss = lossF(predictions, Y)
                    loss_metric(loss)

                    for metric in metrics.values():
                        metric(predictions, Y.long())

                for name, metric in metrics.items():
                    history[name + '_val'].append(metric.compute())
                    metric.reset()
                history['loss_val'].append(loss_metric.compute())
                loss_metric.reset()
                # tensor - > float conversion
    history = {k: [x.item() for x in v] for (k, v) in history.items()}
    return model, history
