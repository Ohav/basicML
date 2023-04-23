import torch
import torch.nn as nn
from consts import CIFAR_IMAGE_SIZE, CIFAR_CLASS_COUNT
import torch.nn.functional as F
import numpy as np
import math
from consts import *


class FCN(nn.Module):
    NAME = "FCN"
    def __init__(self, hidden_width=HIDDEN_WIDTH_FCN,
                 depth=DEPTH_FCN,
                 dropout=DROPOUT_FCN):
        super().__init__()
        self.hidden_width = hidden_width
        # Always a first layer
        self.FC_layers = nn.ModuleList()
        self.FC_layers.append(nn.Linear(CIFAR_IMAGE_SIZE, hidden_width))
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
        for layer in self.FC_layers:
            X = self.dropout_layer(X)
            X = F.relu(layer(X))
        X = self.dropout_layer(X)
        result = self.out(X)
        return result


class CNN(nn.Module):
    NAME = 'CNN'
    def __init__(self, dropout=DROPOUT_CNN, number_of_filters_list=(64, 16), pool_layer_kernel_size=2, pool_layer_stride_size=2):
        super(CNN, self).__init__()

        self.number_of_filters_list = number_of_filters_list
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(nn.Conv2d(3, number_of_filters_list[0], kernel_size=3))
        self.conv_layers_output_height_and_width = H - 2
        self.conv_layers_output_height_and_width = math.ceil(self.conv_layers_output_height_and_width / 2)
        for i in range(1, len(number_of_filters_list)):
            self.conv_layers.append(nn.Conv2d(number_of_filters_list[i-1], number_of_filters_list[i], kernel_size=3))
            self.conv_layers_output_height_and_width = self.conv_layers_output_height_and_width - 2
            if i <= 1:
                self.conv_layers_output_height_and_width = math.ceil(self.conv_layers_output_height_and_width/2)

        #  Q3 1-6:
        # self.conv1_output_size = 64*30*30
        # self.conv2_output_size = 16 * 13 * 13

        self.pool = nn.MaxPool2d(pool_layer_kernel_size, pool_layer_stride_size, ceil_mode=True)

        self.hidden_width = (self.conv_layers_output_height_and_width ** 2) * number_of_filters_list[-1]
        self.fc = nn.Linear(self.hidden_width, 10)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        val = x
        for i, conv_layer in enumerate(self.conv_layers):
            val = F.relu(conv_layer(val))
            if i <= 1:
                val = self.pool(val)
            val = self.dropout_layer(val)

        val = torch.flatten(val, 1)  # flatten all dimensions except batch
        val = self.fc(val)
        return val

    def init_weights(self, std):
        for conv_layer in self.conv_layers:
            conv_layer.weight.data.normal_(mean=0.0, std=std)
            conv_layer.bias.data.normal_(mean=0.0, std=std)

        self.fc.weight.data.normal_(mean=0.0, std=std)
        self.fc.bias.data.normal_(mean=0.0, std=std)

    def init_weights_xavier(self):
        layer_range = np.sqrt(6) / np.sqrt(CIFAR_IMAGE_SIZE + self.number_of_filters_list[0] * 3 * 3)
        self.conv_layers[0].weight.data.uniform_(-layer_range, layer_range)
        for i in range(1, len(self.number_of_filters_list)):
            layer_range = np.sqrt(6) / np.sqrt(self.number_of_filters_list[i-1] * 3 * 3 +
                                               self.number_of_filters_list[i] * 3 * 3)
            self.conv_layers[i].weight.data.uniform_(-layer_range, layer_range)

        out_range = np.sqrt(6) / np.sqrt(self.hidden_width + CIFAR_CLASS_COUNT)
        self.fc.weight.data.uniform_(-out_range, out_range)


