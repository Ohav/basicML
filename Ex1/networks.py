import torch
import torch.nn as nn
from consts import CIFAR_IMAGE_SIZE, CIFAR_CLASS_COUNT
import torch.nn.functional as F
import numpy as np


class FCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_width = 256
        self.layer1 = nn.Linear(CIFAR_IMAGE_SIZE, self.hidden_width)
        self.out = nn.Linear(self.hidden_width, CIFAR_CLASS_COUNT)

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
        X = torch.flatten(X, 1)  # flatten all dimensions except batch
        first_layer = F.relu(self.layer1(X))
        result = self.out(first_layer)
        return result

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)  # output - 30 * 30 * 64 (H x W x D)
        self.pool1 = nn.MaxPool2d(2, 2)  # output - 15 * 15 * 64
        self.conv2 = nn.Conv2d(64, 16, 3)  # output - 13 * 13 * 16
        self.pool2 = nn.MaxPool2d(2, 2)  # output - 7 * 7 * 16

        self.fc = nn.Linear(6*6*16, 784)
        self.out = nn.Linear(784, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc(x)
        x = self.out(x)
        return x

    def init_weights(self, std):
        self.conv1.weight.data.normal_(mean=0.0, std=std)
        self.conv1.bias.data.normal_(mean=0.0, std=std)  # TODO: do convolutional layers have bias?
        self.conv2.weight.data.normal_(mean=0.0, std=std)
        self.conv2.bias.data.normal_(mean=0.0, std=std)
        self.fc.weight.data.normal_(mean=0.0, std=std)
        self.fc.bias.data.normal_(mean=0.0, std=std)
        self.out.weight.data.normal_(mean=0.0, std=std)
        self.out.bias.data.normal_(mean=0.0, std=std)

    def init_weights_xavier(self):
        nn.init.xavier_uniform_(self.conv1.weight)
        #nn.init.xavier_uniform_(self.conv1.bias)
        nn.init.xavier_uniform_(self.conv2.weight)
        #nn.init.xavier_uniform_(self.conv2.bias)
        nn.init.xavier_uniform_(self.fc.weight)
        #nn.init.xavier_uniform_(self.fc.bias)
        nn.init.xavier_uniform_(self.out.weight)
        #nn.init.xavier_uniform_(self.out.bias)




