import torchmetrics
import torch
from torch import nn


class ClassifierSimple(torch.nn.Module):
    def __init__(self, input_dim=300, hidden_size=64):
        super(ClassifierSimple, self).__init__()

        self.layers = nn.Sequential(
            # flatten input if necessary
            nn.Flatten(),
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

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
