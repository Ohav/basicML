import torch
import torch.nn as nn
from consts import CIFAR_IMAGE_SIZE, CIFAR_CLASS_COUNT
import torch.nn.functional as F


class FCN(nn.Module):
    def __init__(self, hidden_width):
        super().__init__()
        self.layer1 = nn.Linear(CIFAR_IMAGE_SIZE, hidden_width)
        self.out = nn.Linear(hidden_width, CIFAR_CLASS_COUNT)

    def init_weights(self, std):
        self.layer1.weight.data.normal_(mean=0.0, std=std)
        self.layer1.bias.data.normal_(mean=0.0, std=std)
        self.out.weight.data.normal_(mean=0.0, std=std)
        self.out.bias.data.normal_(mean=0.0, std=std)

    def forward(self, X):
        first_layer = F.relu(self.layer1(X))
        result = self.out(first_layer)
        return result



