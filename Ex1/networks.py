import torch
import torch.nn as nn
from consts import CIFAR_IMAGE_SIZE, CIFAR_CLASS_COUNT
import torch.nn.functional as F
import numpy as np


class FCN(nn.Module):
    def __init__(self, hidden_width, depth, dropout):
        super().__init__()
        self.hidden_width = hidden_width
        self.FC_layers = []
        self.layer1 = nn.Linear(CIFAR_IMAGE_SIZE, hidden_width)
        self.FC_layers.append(self.layer1)
        for i in range(depth - 1):
            self.FC_layers.append(nn.Linear(hidden_width, hidden_width))

        self.out = nn.Linear(hidden_width, CIFAR_CLASS_COUNT)
        self.dropout_layer = nn.Dropout(dropout)

    def init_weights(self, std):
        self.layer1.weight.data.normal_(mean=0.0, std=std)
        self.layer1.bias.data.normal_(mean=0.0, std=std)
        self.out.weight.data.normal_(mean=0.0, std=std)
        self.out.bias.data.normal_(mean=0.0, std=std)

    def init_weights_xavier(self):
        layer1_range = np.sqrt(6)/np.sqrt(CIFAR_IMAGE_SIZE + self.hidden_width)
        self.layer1.weight.data.uniform_(-layer1_range, layer1_range)
        self.layer1.bias.data.uniform_(-layer1_range, layer1_range)
        out_range = np.sqrt(6) / np.sqrt(self.hidden_width + CIFAR_CLASS_COUNT)
        self.out.weight.data.uniform_(-out_range, out_range)
        self.out.bias.data.uniform_(-out_range, out_range)

    def forward(self, X):
        self.dropout_layer(X)
        for layer in self.FC_layers:
            X = F.relu(layer(X))
        result = self.out(X)
        return result



