from collections import defaultdict

import numpy as np
import torch
import torchmetrics
from torch.utils.data import DataLoader
from tqdm import trange

from profiler import Profiler
from utils import verbprint


def train_bert_embeddings_mlm(model, epochs, dataset, dataset_eval, batchsize,
                              optimizer, lossF, device, folder,
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

    profiler = Profiler(['batch', 'device', 'forward', 'loss', 'backward',
                         'metrics', 'eval'])

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
            logits_no_sequence = logits.reshape(
                logits_shape[0] * logits_shape[1], logits_shape[2])

            # (batchsize)
            batch_labels_no_sequence = batch_labels.flatten().to(device)
            batch_mask = (inputs[:, :, 1] > 0).flatten().to(device)

            # print(logits_no_sequence[batch_mask][0])
            loss = lossF(
                logits_no_sequence[batch_mask],
                batch_labels_no_sequence[batch_mask])

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
                logits_no_sequence = logits.reshape(
                    logits_shape[0] * logits_shape[1], logits_shape[2])

                # (batchsize)
                batch_labels_no_sequence = batch_labels.flatten().to(device)

                batch_mask = (inputs[:, :, 1] > 0).flatten().to(device)

                loss = lossF(
                    logits_no_sequence[batch_mask],
                    batch_labels_no_sequence[batch_mask])

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
        print('current_loss=',
              history['loss'][-1], '\t', 'eval_loss=',
              history['loss_eval'][-1])

        profiler.timer_stop('eval')
        early_stopper.measure(eval_loss)
        if early_stopper.should_stop():
            break

    history['batchloss_metric'] = batchloss_metric
    history['batchloss_metric_eval'] = batchloss_metric_eval

    profiler.eval()
    return model, optimizer, history, profiler.get_profile()


def train_bert_embeddings_lm(model, epochs, dataset, dataset_eval, batchsize,
                             optimizer, device, folder,
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

    profiler = Profiler(['batch', 'device', 'forward', 'loss', 'backward',
                         'metrics', 'eval'])

    for ep in trange(epochs):
        for batch_ids, batch_mask, batch_type_ids, batch_labels in dl:
            profiler.timer_start('batch')

            model.train()
            optimizer.zero_grad()

            profiler.timer_stop('batch')

            profiler.timer_start('device')
            batch_ids = batch_ids.to(device)
            batch_mask = batch_mask.to(device)

            batch_type_ids = batch_type_ids.to(device)
            batch_labels = batch_labels.to(device)

            profiler.timer_stop('device')

            profiler.timer_start('forward')

            out = model.forward(batch_ids, batch_mask, batch_type_ids, labels=batch_labels)
            profiler.timer_stop('forward')
            profiler.timer_start('loss')

            loss = out.loss

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
            for batch_ids, batch_mask, batch_type_ids, batch_labels in dl_eval:
                optimizer.zero_grad()

                batch_ids = batch_ids.to(device)
                batch_mask = batch_mask.to(device)
                batch_type_ids = batch_type_ids.to(device)
                batch_labels = batch_labels.to(device)

                out = model.forward(batch_ids, batch_mask, batch_type_ids, labels=batch_labels)
                loss = out.loss

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
        print('current_loss=',
              history['loss'][-1], '\t', 'eval_loss=',
              history['loss_eval'][-1])

        profiler.timer_stop('eval')
        early_stopper.measure(eval_loss)
        if early_stopper.should_stop():
            break

    history['batchloss_metric'] = batchloss_metric
    history['batchloss_metric_eval'] = batchloss_metric_eval

    profiler.eval()
    return model, optimizer, history, profiler.get_profile()


def train_bert_embeddings_lp(model, epochs, dataset, dataset_eval, batchsize,
                             optimizer, lossF, device, folder,
                             stop_early_patience=5, stop_early_delta=0.1,
                             vector_size=200):
    """
    Train a model on a dataset_file.

    """
    # doing this because we get 1/2 fake input
    batchsize = int(batchsize / 2)
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

    profiler = Profiler(['batch', 'device', 'forward', 'loss', 'backward',
                         'metrics', 'eval'])

    classifier = torch.nn.Sequential(
        torch.nn.Linear(vector_size * 3, vector_size),
        torch.nn.Dropout(0.25),
        torch.nn.ReLU(),
        torch.nn.Linear(vector_size, int(vector_size / 2)),
        torch.nn.ReLU(),
        torch.nn.Linear(int(vector_size / 2), 1),

    )
    classifier.train()
    classifier = classifier.to(device)
    for ep in trange(epochs):
        for real_tp, fake_tp in dl:

            real_tp = real_tp.squeeze()
            fake_tp = fake_tp.squeeze()

            if len(fake_tp.shape) > 2:
                fake_tp = fake_tp.flatten(0, 1)

            tp = torch.cat((real_tp, fake_tp))

            profiler.timer_start('batch')

            attention_mask = torch.ones((len(real_tp) + len(fake_tp), 5))
            label = torch.cat((torch.ones(len(real_tp)),
                               torch.zeros(len(fake_tp))))

            model.train()
            classifier.train()
            optimizer.zero_grad()

            profiler.timer_stop('batch')

            profiler.timer_start('device')
            tp = tp.to(device)
            attention_mask = attention_mask.to(device)
            label = label.to(device)

            profiler.timer_stop('device')

            profiler.timer_start('forward')
            out = model.forward(tp, attention_mask)

            embeddings = out['last_hidden_state']
            # remove cls and sep embeddings and flatten
            classifiert_input = embeddings[:, 1:4].flatten(1, 2)

            predictions = classifier.forward(classifiert_input).squeeze()

            profiler.timer_stop('forward')
            profiler.timer_start('loss')

            loss = lossF(predictions, label)

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
            classifier.eval()
            for real_tp, fake_tp in dl_eval:

                real_tp = real_tp.squeeze()
                fake_tp = fake_tp.squeeze()

                optimizer.zero_grad()

                if len(fake_tp.shape) > 2:
                    fake_tp = fake_tp.flatten(0, 1)

                tp = torch.cat((real_tp, fake_tp))

                attention_mask = torch.ones((len(real_tp) + len(fake_tp), 5))
                label = torch.cat((torch.ones(len(real_tp)),
                                   torch.zeros(len(fake_tp))))

                tp = tp.to(device).squeeze()
                attention_mask = attention_mask.to(device)
                label = label.to(device)
                out = model.forward(tp, attention_mask)

                embeddings = out['last_hidden_state']
                # remove cls and sep embeddings and flatten
                classifiert_input = embeddings[:, 1:4].flatten(1, 2)

                predictions = classifier.forward(classifiert_input).squeeze()

                loss = lossF(predictions, label)

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
        print('current_loss=',
              history['loss'][-1], '\t', 'eval_loss=',
              history['loss_eval'][-1])

        profiler.timer_stop('eval')
        early_stopper.measure(eval_loss)
        if early_stopper.should_stop():
            break

    history['batchloss_metric'] = batchloss_metric
    history['batchloss_metric_eval'] = batchloss_metric_eval

    profiler.eval()
    return model, optimizer, history, profiler.get_profile(), classifier


def score_bert_model_mlm(model, device, dataset, batchsize, optimizer, lossF):
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
            logits_no_sequence = logits.reshape(
                logits_shape[0] * logits_shape[1], logits_shape[2])

            # (batchsize)
            batch_labels_no_sequence = batch_labels.flatten().to(device)

            batch_mask = (inputs[:, :, 1] > 0).flatten().to(device)

            loss = lossF(
                logits_no_sequence[batch_mask],
                batch_labels_no_sequence[batch_mask])

            loss_metric(loss.detach())
            batchloss_metric_eval(loss.detach())
    return loss_metric.compute().item()


def score_bert_model_lp(model, device, dataset, batchsize, optimizer, lossF,
                        classifier):
    with torch.no_grad():
        dl_test = DataLoader(dataset, batch_size=batchsize, shuffle=False, pin_memory=True)
        loss_metric = torchmetrics.aggregation.MeanMetric().to(device)
        batchloss_metric_eval = torchmetrics.aggregation.CatMetric().to(device)

        model.eval()
        classifier.eval()
        loss_metric.reset()
        for real_tp, fake_tp in dl_test:

            optimizer.zero_grad()

            real_tp = real_tp.squeeze()
            fake_tp = fake_tp.squeeze()

            if len(fake_tp.shape) > 2:
                fake_tp = fake_tp.flatten(0, 1)

            tp = torch.cat((real_tp, fake_tp))

            attention_mask = torch.ones((len(real_tp) + len(fake_tp), 5))
            label = torch.cat((torch.ones(len(real_tp)),
                               torch.zeros(len(fake_tp))))

            tp = tp.to(device).squeeze()
            attention_mask = attention_mask.to(device)
            label = label.to(device)

            out = model.forward(tp, attention_mask)

            embeddings = out['last_hidden_state']
            # remove cls and sep embeddings and flatten
            classifiert_input = embeddings[:, 1:4].flatten(1, 2)

            predictions = classifier.forward(classifiert_input).squeeze()

            loss = lossF(predictions, label)

            loss_metric(loss.detach())
            batchloss_metric_eval(loss.detach())
    return loss_metric.compute().item()


class EarlyStopper():
    # TODO: Add min epochs
    # look into ray tune ?

    """
    Set patience <0 to ignore early stopping.
    """

    def __init__(self, patience, delta, type='basic', min_epochs=10):
        self.patience = patience
        self.delta = delta
        self.counter = 1
        self.min_metric = None
        self.stop = False
        self.min_epochs = min_epochs
        self.log = {'counter': [], 'min_epochs': [], 'min_metric': [],
                    'metric' : [], 'stop': []}

    def __log__(self, metric):
        self.log['counter'].append(self.counter)
        self.log['min_epochs'].append(self.min_epochs)
        self.log['min_metric'].append(self.min_metric)
        self.log['metric'].append(metric)
        self.log['stop'].append(self.stop)

    def measure(self, metric):
        if self.min_epochs > 0:
            self.min_epochs -= 1
        else:
            if not self.min_metric:
                self.min_metric = metric

            else:
                delta = self.min_metric - metric
                if delta > self.delta:
                    self.counter = 1
                    self.min_metric = metric
                else:
                    if self.counter >= self.patience:
                        self.stop = True
                    else:
                        self.counter += 1
                        #print('++')
        self.__log__(metric)

    def should_stop(self):
        if self.patience < 0:
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
