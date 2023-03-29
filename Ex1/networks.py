import torch
import torch.nn as nn
from consts import CIFAR_IMAGE_SIZE, CIFAR_CLASS_COUNT


class FCN(nn.Module):
    def __init__(self, hidden_width):
        super(nn.Module).__init__()
        self.layer1 = nn.Linear(CIFAR_IMAGE_SIZE, hidden_width)
        self.out = nn.Linear(hidden_width, CIFAR_CLASS_COUNT)

    def forward(self, X):
        first_layer = self.layer1(X)
        result = self.out(first_layer)
        return result



