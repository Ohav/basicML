import torch
import torch.nn as nn
from consts import CIFAR_IMAGE_SIZE, CIFAR_CLASS_COUNT
import torch.nn.functional as F
import numpy as np
from consts import *


class FCN(nn.Module):
    def __init__(self, hidden_width=HIDDEN_WIDTH_FCN,
                 depth=DEPTH_FCN,
                 dropout=DROPOUT_FCN):
        super().__init__()
        self.hidden_width = hidden_width
        # Always a first layer
        self.FC_layers = [nn.Linear(CIFAR_IMAGE_SIZE, hidden_width)]
        for i in range(depth - 1):
            self.FC_layers.append(nn.Linear(hidden_width, hidden_width))

        self.out = nn.Linear(hidden_width, CIFAR_CLASS_COUNT)
        self.dropout_layer = nn.Dropout(dropout)

    def init_weights(self, std):
        for l in self.FC_layers:
            l.weight.data.normal_(mean=0.0, std=std)
            l.bias.data.normal_(mean=0.0, std=std)

        self.out.weight.data.normal_(mean=0.0, std=std)
        self.out.bias.data.normal_(mean=0.0, std=std)

    def init_weights_xavier(self):
        # fix init with depth
        layer1_range = np.sqrt(6) / np.sqrt(CIFAR_IMAGE_SIZE + self.hidden_width)
        self.FC_layers[0].weight.data.uniform_(-layer1_range, layer1_range)
        # self.FC_layers[0].bias.data.uniform_(-layer1_range, layer1_range)
        hidden_layers_range = np.sqrt(6) / np.sqrt(self.hidden_width * 2)
        for i in range(len(self.FC_layers) - 1):
            self.FC_layers[i + 1].weight.data.uniform_(-hidden_layers_range, hidden_layers_range)
            # self.FC_layers[i + 1].bias.data.uniform_(-hidden_layers_range, hidden_layers_range)

        out_range = np.sqrt(6) / np.sqrt(self.hidden_width + CIFAR_CLASS_COUNT)
        self.out.weight.data.uniform_(-out_range, out_range)
        # self.out.bias.data.uniform_(-out_range, out_range)

    def forward(self, X):
        X = torch.flatten(X, 1)
        self.dropout_layer(X)
        for layer in self.FC_layers:
            X = F.relu(layer(X))
        result = self.out(X)
        return result

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)  # output - 30 * 30 * 64 (H x W x D)
        self.conv1_output_size = 64*30*30
        self.conv2 = nn.Conv2d(64, 16, 3)  # output - 13 * 13 * 16
        self.conv2_output_size = 16 * 13 * 13
        self.pool = nn.MaxPool2d(2, 2)

        self.fc = nn.Linear(6*6*16, 784)
        self.hidden_width = 784
        self.out = nn.Linear(784, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
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
        # TODO: Change to our own Xavier, like in FCN
        # conv1_range = np.sqrt(6) / np.sqrt(CIFAR_IMAGE_SIZE + )
        # self.conv1.weight.data.uniform_()
        nn.init.xavier_uniform_(self.conv1.weight)
        #nn.init.xavier_uniform_(self.conv1.bias)
        nn.init.xavier_uniform_(self.conv2.weight)
        #nn.init.xavier_uniform_(self.conv2.bias)
        nn.init.xavier_uniform_(self.fc.weight)
        #nn.init.xavier_uniform_(self.fc.bias)
        nn.init.xavier_uniform_(self.out.weight)
        #nn.init.xavier_uniform_(self.out.bias)




