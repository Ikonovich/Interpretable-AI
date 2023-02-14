import torch
from torch import nn
import torch.nn.functional as func
import torch.optim as optim

import keras
from keras import layers
from torch.autograd import Variable
from torch.utils.data import DataLoader


class CoreNetwork(nn.Module):

    def __init__(self):
        super(CoreNetwork, self).__init__()

        self.loss_func = torch.nn.CrossEntropyLoss()  # Softmax is internally computed.
        self.optimizer = None

        self.convOne = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.ReLU()
        )

        self.denseOne = nn.Linear(in_features=27556 * 10, out_features=1000, bias=True)
        self.denseTwo = nn.Linear(in_features=1000, out_features=10, bias=True)
        torch.nn.init.xavier_uniform_(self.denseOne.weight)  # initialize parameters
        torch.nn.init.xavier_uniform_(self.denseTwo.weight)

        #
        # self.loss_function = torch.nn.CrossEntropyLoss()  # Softmax is internally computed.
        # self.optimizer = torch.optim.SGD(params=self.parameters(), lr=learning_rate)
        #
        # self.optimizer.zero_grad()  # Initialize gradients

        ############
        # L1 ImgIn shape=(?, 28, 28, 1)
        # Conv -> (?, 28, 28, 32)
        # Pool -> (?, 14, 14, 32)
        # self.convOne = torch.nn.Sequential(
        #     torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
        #     torch.nn.ReLU(),
        #     torch.nn.MaxPool2d(kernel_size=2, stride=2),
        #     torch.nn.Dropout(p=1 - keep_prob))
        # # L2 ImgIn shape=(?, 14, 14, 32)
        # # Conv      ->(?, 14, 14, 64)
        # # Pool      ->(?, 7, 7, 64)
        # self.layerTwo = torch.nn.Sequential(
        #     torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        #     torch.nn.ReLU(),
        #     torch.nn.MaxPool2d(kernel_size=2, stride=2),
        #     torch.nn.Dropout(p=1 - keep_prob))
        # # L3 ImgIn shape=(?, 7, 7, 64)
        # # Conv ->(?, 7, 7, 128)
        # # Pool ->(?, 4, 4, 128)
        # self.layerThree = torch.nn.Sequential(
        #     torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        #     torch.nn.ReLU(),
        #     torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        #     torch.nn.Dropout(p=1 - keep_prob))
        #
        # # L4 FC 4x4x128 inputs -> 625 outputs
        # self.denseOne = torch.nn.Linear(4 * 4 * 128, 625, bias=True)
        # torch.nn.init.xavier_uniform(self.denseOne.weight)
        # self.layer4 = torch.nn.Sequential(
        #     self.denseOne,
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(p=1 - keep_prob))
        # # L5 Final FC 625 inputs -> 10 outputs
        # self.fc2 = torch.nn.Linear(625, 10, bias=True)
        # torch.nn.init.xavier_uniform_(self.fc2.weight)  # initialize parameters

    def train(self, dataloader: DataLoader, learning_rate=0.01):
        # Trains the network with a given data loader.

        # Weight decay represents a constant of L2 regularization rather than actual weight decay
        # i.e  weight_decay=1e-5
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)

        # Stores the number of correct predictions
        total_correct = 0
        iteration = 1
        print("Beginning training.")
        for i, (batch_X, batch_Y) in enumerate(dataloader):
            X = Variable(batch_X)  # image is already size of (28x28), no reshape
            Y = Variable(batch_Y)  # label is not one-hot encoded

            result = self.iterate(X, Y, training=True)

            if result == Y.data:
                total_correct += 1
            if iteration % 5 == 1:
                accuracy = total_correct/iteration
                print("\t Iteration: {},\t accuracy = {}             \r".format(iteration, accuracy))

            iteration += 1

    def iterate(self, X, Y, training=False):
        # Runs a single iteration of the network.
        self.optimizer.zero_grad()  # <= initialization of the gradients

        # forward propagation
        hypothesis = self(X)

        if training == True:
            cost = self.loss_func(hypothesis, Y)  # <= compute the loss function
            # Backward propagation
            cost.backward()  # <= compute the gradient of the loss/cost function
            self.optimizer.step()  # <= Update the gradients

        # Return the prediction
        return hypothesis.data.max(dim=1)[1]

    # Runs a single iteration of forward propagation on the created network.
    def forward(self, sample):
        # layerOne = self.layerOne(sample)
        # layerTwo = self.layerTwo(layerOne)
        # return layerTwo
        out = self.convOne(sample)
        out = out.view(out.size(0), -1)   # Flatten them for FC
        out = self.denseOne(out)
        out = self.denseTwo(out)
        return out


    # Creates and runs a data augmentation network on the provided dataset
    def augment_data(self, data):
        augment = keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.1)
            ]
        )
        return augment(data)


