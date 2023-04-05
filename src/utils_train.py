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
from utils import verbprint
from profiler import Profiler


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

            # shape is (bs,2,3*VECTORSIZE) because of dataset_file implementation
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

                    # shape is (bs,2,3*VECTORSIZE) because of dataset_file implementation
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
    return model, optimizer, history


def train_bert_embeddings(model, epochs, dataset, dataset_eval, batchsize, optimizer, lossF, device, folder,
                          stop_early_patience=5, stop_early_delta=0.1):
    """
    Train a model on a dataset_file.

    """

    early_stopper = EarlyStopper(stop_early_patience, stop_early_delta)
    best_eval_loss = None

    dl = DataLoader(dataset, batch_size=batchsize, shuffle=True, pin_memory=True)
    dl_eval = DataLoader(dataset_eval, batch_size=batchsize, shuffle=False, pin_memory=True)

    optimizer = optimizer(model.parameters())

    loss_metric = torchmetrics.aggregation.MeanMetric().to(device)
    batchloss_metric = torchmetrics.aggregation.CatMetric().to(device)
    batchloss_metric_eval = torchmetrics.aggregation.CatMetric().to(device)
    history = defaultdict(list)

    verbprint(f"cuda_model: {next(model.parameters()).is_cuda}")
    verbprint("Starting training")

    profiler = Profiler(['batch', 'device', 'forward', 'loss', 'backward', 'metrics', 'eval'])

    for ep in trange(epochs):
        for inputs, batch_mask, batch_labels in dl:
            profiler.timer_start('batch')

            model.train()
            optimizer.zero_grad()
            batch_id = inputs[:, :, 0]
            profiler.timer_stop('batch')

            profiler.timer_start('device')
            batch_id = batch_id.to(device)
            batch_mask = batch_mask.to(device)
            profiler.timer_stop('device')

            profiler.timer_start('forward')
            out = model.forward(batch_id.to(device), batch_mask.to(device))
            profiler.timer_stop('forward')
            profiler.timer_start('loss')

            logits = out.logits

            # (batchsize, sequence_len, no_labels)
            logits_shape = logits.shape

            # (batchsize * sequence_len, no_labels)
            logits_no_sequence = logits.reshape(logits_shape[0] * logits_shape[1], logits_shape[2])

            # (batchsize)
            batch_labels_no_sequence = batch_labels.flatten().to(device)
            batch_mask = (inputs[:, :, 1] > 0).flatten().to(device)

            loss = lossF(logits_no_sequence[batch_mask], batch_labels_no_sequence[batch_mask])

            profiler.timer_stop('loss')

            profiler.timer_start('backward')

            loss.backward()
            optimizer.step()

            profiler.timer_stop('backward')

            # detach loss! Otherwise causes memory leakage

            profiler.timer_start('metrics')
            loss_metric(loss.detach())
            batchloss_metric(loss.detach())
            profiler.timer_stop('metrics')

        profiler.timer_start('metrics')
        history['loss'].append(loss_metric.compute().detach().item())
        loss_metric.reset()
        profiler.timer_stop('metrics')

        profiler.timer_start('eval')
        with torch.no_grad():
            model.eval()
            for inputs, batch_mask, batch_labels in dl_eval:
                optimizer.zero_grad()
                batch_id = inputs[:, :, 0]

                out = model.forward(batch_id.to(device), batch_mask.to(device))
                logits = out.logits

                # (batchsize, sequence_len, no_labels)
                logits_shape = logits.shape

                # (batchsize * sequence_len, no_labels)
                logits_no_sequence = logits.reshape(logits_shape[0] * logits_shape[1], logits_shape[2])

                # (batchsize)
                batch_labels_no_sequence = batch_labels.flatten().to(device)

                batch_mask = (inputs[:, :, 1] > 0).flatten().to(device)

                loss = lossF(logits_no_sequence[batch_mask], batch_labels_no_sequence[batch_mask])

                loss_metric(loss.detach())
                batchloss_metric_eval(loss.detach())

            eval_loss = loss_metric.compute().detach().item()
            if not best_eval_loss:
                best_eval_loss = eval_loss
            else:
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    model.save_pretrained(folder / "model_best_eval")


            history['loss_eval'].append(eval_loss)

            loss_metric.reset()
        print('current_loss=',history['loss'][-1],'\t', 'eval_loss=',history['loss_eval'][-1])


        profiler.timer_stop('eval')
        early_stopper.measure(eval_loss)
        if early_stopper.should_stop():
            break

    history['batchloss_metric'] = batchloss_metric
    history['batchloss_metric_eval'] = batchloss_metric_eval

    profiler.eval()
    return model, optimizer, history, profiler.get_profile()


def score_bert_model(model, device, dataset, batchsize, optimizer, lossF):
    with torch.no_grad():
        dl_test = DataLoader(dataset, batch_size=batchsize, shuffle=False, pin_memory=True)
        loss_metric = torchmetrics.aggregation.MeanMetric().to(device)
        batchloss_metric_eval = torchmetrics.aggregation.CatMetric().to(device)

        model.eval()
        loss_metric.reset()
        for inputs, batch_mask, batch_labels in dl_test:
            optimizer.zero_grad()
            batch_id = inputs[:, :, 0]

            out = model.forward(batch_id.to(device), batch_mask.to(device))
            logits = out.logits

            # (batchsize, sequence_len, no_labels)
            logits_shape = logits.shape

            # (batchsize * sequence_len, no_labels)
            logits_no_sequence = logits.reshape(logits_shape[0] * logits_shape[1], logits_shape[2])

            # (batchsize)
            batch_labels_no_sequence = batch_labels.flatten().to(device)

            batch_mask = (inputs[:, :, 1] > 0).flatten().to(device)

            loss = lossF(logits_no_sequence[batch_mask], batch_labels_no_sequence[batch_mask])

            loss_metric(loss.detach())
            batchloss_metric_eval(loss.detach())
    return loss_metric.compute().item()


class EarlyStopper():

    def __init__(self, patience, delta, type='basic'):
        self.patience = patience
        self.delta = delta
        self.counter = 1
        self.smallest_metric = None
        self.stop = False

    def measure(self, metric):
        print('patience:',self.counter)
        if not self.smallest_metric:
            self.smallest_metric = metric
        else:
            delta = self.smallest_metric - metric
            if delta > self.delta:
                self.counter = 1
                self.smallest_metric = metric
            else:
                if self.counter >= self.patience:
                    self.stop = True
                else:
                    self.counter += 1

    def should_stop(self):
        if not self.patience < 0:
            return False

        elif self.stop:
            return True
        else:
            return False


def get_bert_embeddings(entities, bert_model, tokenizer):
    entities = [tokenizer.encode(x) for x in np.array(entities)]
    embeddings = bert_model(torch.tensor(entities))
    embeddings = embeddings['last_hidden_state'][:, 1]

    # return numpy because of 'backwards_compatibility'
    # maybe make this optional
    return embeddings.detach().cpu().numpy()
