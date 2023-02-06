import numpy as np
import torch
from torch import nn

import keras
from keras.datasets import mnist
from keras import layers


class CoreNetwork(nn.Module):

    def __init__(self):
        super(CoreNetwork, self).__init__()

        self.layerOne = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )

        self.layerTwo = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=28 * 28 * 5, out_features=10),
        )


    # Runs a single iteration of forward propagation on the created network.
    def forward(self, sample):
        layerOne = self.layerOne(sample)
        logits = self.layerTwo(layerOne)
        return logits

    # Creates and runs a data augmentation network on the provided dataset
    def augment_data(self, data):
        augment = keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.1)
            ]
        )
        return augment(data)


