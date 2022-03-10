#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Module with models to train an animal classifier"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "03/22"

# Third-party modules
from torch import nn, flatten
from torchvision import models

class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()
        # Mantains output size
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels=8, 
            kernel_size=(3,3), 
            stride=(1,1),
            padding=(1,1)
            )
        # Reduce output half the output size
        self.pool = nn.MaxPool2d(
            kernel_size=(2,2),
            stride=(2,2))
        # 
        self.conv2 = nn.Conv2d(
            in_channels=8, 
            out_channels=16, 
            kernel_size=(3,3), 
            stride=(1,1),
            padding=(1,1)
            )
        self.fc1 = nn.Linear(16*32*32, num_classes)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.pool(x)
        x = self.act(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x


models = {
    "cnn": CNN(in_channels=3, num_classes=10),
    "resnet18": models.resnet18(pretrained=False)
}