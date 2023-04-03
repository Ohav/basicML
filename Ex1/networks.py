import torch
import torch.nn as nn
from consts import CIFAR_IMAGE_SIZE, CIFAR_CLASS_COUNT
import torch.nn.functional as F
import numpy as np


class FCN(nn.Module):
    def __init__(self, hidden_width):
        super().__init__()
        self.hidden_width = hidden_width
        self.layer1 = nn.Linear(CIFAR_IMAGE_SIZE, hidden_width)
        self.out = nn.Linear(hidden_width, CIFAR_CLASS_COUNT)

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
        first_layer = F.relu(self.layer1(X))
        result = self.out(first_layer)
        return result



