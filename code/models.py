import torchmetrics
import torch
from torch import nn
from torch.utils.data import DataLoader
from settings import VECTOR_SIZE, BERT_SIMPLE_NAME,BERT_EPOCHS, CLASSIFIER_DROPOUT
from transformers import PreTrainedTokenizerFast, AutoModel, BertForTokenClassification
from utils_train import get_bert_embeddings

class ClassifierSimple(torch.nn.Module):
    def __init__(self, input_dim=3*VECTOR_SIZE, hidden_size=64,dropout=CLASSIFIER_DROPOUT):
        super(ClassifierSimple, self).__init__()

        self.layers = nn.Sequential()
            # flatten input if necessary
        self.layers.append(nn.Flatten())

        if dropout:
            self.layers.append(nn.Dropout(dropout))

        self.layers.append(nn.Linear(input_dim, hidden_size))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_size, 1))


        self.output_activation = nn.Sigmoid()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        return self.layers(x)

    def predict(self, x):
        x.to(self.device)

        return self.output_activation(self.layers(x))

    def predict_numpy(self, x):
        x = torch.tensor(x)
        x.to(self.device)
        return self.output_activation(self.layers(x)).detach().cpu().numpy()



class SimpleBertTokenClassifier(torch.nn.Module):

    def __init__(self,tokenizer,pretrained_config_name = 'prajjwal1/bert-tiny'):

        super(SimpleBertTokenClassifier, self).__init__()
        pretrained_model = AutoModel.from_pretrained(pretrained_config_name)
        pretrained_config = pretrained_model.config

        pretrained_config._name_or_path = "otautz/tiny_simple"

        # Encoder Only model
        pretrained_config.is_decoder = False
        pretrained_config.add_cross_attention = False

        # Predict vocab_size names
        pretrained_config.num_labels = tokenizer.get_vocab_size()

        # Embedding size
        pretrained_config.hidden_size = VECTOR_SIZE
        pretrained_config.max_position_embeddings = BERT_SIMPLE_MAXLEN

        del pretrained_model

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = BertForTokenClassification(pretrained_config)
        self.model.to(self.device)

    def forward(self,X):
        return self.model.forward(X)

    def train(self, dataset,dataset_eval = None, name=BERT_SIMPLE_NAME, lossF=torch.nn.CrossEntropyLoss(), batchsize=5000, optimizer=None, epochs=BERT_EPOCHS):

        if not optimizer:
            optimizer = torch.optim.Adam(self.model.parameters())

        dl = DataLoader(dataset, batch_size=batchsize, shuffle=True, pin_memory=True)

        if dataset_eval:
            dl_eval =  DataLoader(dataset_eval, batch_size=batchsize, shuffle=True, pin_memory=True)

        loss_metric = torchmetrics.aggregation.MeanMetric().to(self.device)
        batchloss_metric = torchmetrics.aggregation.CatMetric().to(self.device)
        batchloss_metric_eval = torchmetrics.aggregation.CatMetric().to(self.device)
        history = defaultdict(list)

        for ep in trange(epochs):
            self.model.train()
            for inputs, batch_mask, batch_labels in dl:
                optimizer.zero_grad()
                batch_id = inputs[:, :, 0]

                out = self.forward(batch_id.to(device), batch_mask.to(device))
                logits = out.logits

                # (batchsize, sequence_len, no_labels)
                logits_shape = logits.shape

                # (batchsize * sequence_len, no_labels)
                logits_no_sequence = logits.reshape(logits_shape[0] * logits_shape[1], logits_shape[2])

                # (batchsize)
                batch_labels_no_sequence = batch_labels.flatten().to(self.device)

                batch_mask = (inputs[:, :, 1] > 0).flatten().to(self.device)

                loss = lossF(logits_no_sequence[batch_mask], batch_labels_no_sequence[batch_mask])

                loss.backward()
                optimizer.step()

                loss_metric(loss)
                batchloss_metric(loss)

            history['loss'].append(loss_metric.compute().item())
            loss_metric.reset()

            if dataset_eval:
                with torch.no_grad():
                    self.model.eval()
                    for inputs, batch_mask, batch_labels in dl_eval:
                        optimizer.zero_grad()
                        batch_id = inputs[:, :, 0]

                        out = self.forward(batch_id.to(device), batch_mask.to(device))
                        logits = out.logits

                        # (batchsize, sequence_len, no_labels)
                        logits_shape = logits.shape

                        # (batchsize * sequence_len, no_labels)
                        logits_no_sequence = logits.reshape(logits_shape[0] * logits_shape[1], logits_shape[2])

                        # (batchsize)
                        batch_labels_no_sequence = batch_labels.flatten().to(self.device)

                        batch_mask = (inputs[:, :, 1] > 0).flatten().to(self.device)

                        loss = lossF(logits_no_sequence[batch_mask], batch_labels_no_sequence[batch_mask])

                        loss_metric(loss)
                        batchloss_metric_eval(loss)

                    history['loss_eval'].append(loss_metric.compute().item())
                    loss_metric.reset()


        return history



class SimpleBertEmbeddings(torch.nn.Module):

    def __init__(self,pretrained_path,tokenizer_path):
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        self.model = AutoModel.from_pretrained(pretrained_path)

    def forward(self,X):
        return get_bert_embeddings(X,self.model,self.tokenizer)




