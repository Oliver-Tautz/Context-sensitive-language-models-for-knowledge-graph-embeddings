from tqdm import trange, tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
import torch
import torchmetrics
import numpy as np

from utils_data import Dataset11, get_vectors_fast
from settings import VECTOR_SIZE
from utils import choose
from utils_graph import get_random_corrupted_triple


def fit_1_1(model, graph, word_vec_mapping, batch_size, entities, metrics=None, epochs=50, optimizer=None,
            lossF=torch.nn.BCEWithLogitsLoss(), graph_eval=None):
    """
    metrics: dictionary of {name:torchmetric} to track for training and evaluation.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
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
                for X, Y in dataloader_eval:
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


def get_bert_embeddings(entities, bert_model, tokenizer):
    entities = [tokenizer.encode(x) for x in np.array(entities)]
    embeddings = bert_model(torch.tensor(entities))
    embeddings = embeddings['last_hidden_state'][:, 1]

    # return numpy because of 'backwards_compatibility'
    # maybe make this optional
    return embeddings.detach().cpu().numpy()

