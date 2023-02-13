import torch
from torch import nn
import torch.nn.functional as func
import torch.optim as optim

import keras
from keras import layers
from torch.autograd import Variable


class CoreNetwork(nn.Module):

    def __init__(self, learning_rate=0.5):
        super(CoreNetwork, self).__init__()

        # hyperparameters
        batch_size = 32
        keep_prob = 1

        self.convOne = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )

        self.denseOne = nn.Linear(in_features=500 * 500 * 5, out_features=10, bias=True)

        torch.nn.init.xavier_uniform_(self.denseOne.weight)  # initialize parameters
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

    # Runs a single training batch
    def run_batch(self, batch_x, batch_y):
        train_cost = []
        accurate = 0
        iterations = 0
        avg_cost = 0


        if len(batch_x) != len(batch_y):
            raise IndexError

        for x, y in zip(batch_x, batch_y):
            iterations += 1

            # Run forward propagation
            result = self(x)
            correct = torch.zeros(10)
            correct[int(y)] = 1
            cost = self.loss_function(result, correct)

            # Backward propagation
            cost.backward()  # Compute the gradient of the loss function
            self.optimizer.step()  # Update the gradients

            # Print some performance to monitor the training
            prediction = torch.argmax(result)
            if prediction == int(y.data): accurate += 1
            train_cost.append(cost.item())
            if iterations % 200 == 0:
                print("Epoch= {},\t batch = {},\t cost = {:2.4f},\t accuracy = {}".format(1, iterations, train_cost[-1],
                                                                                          accurate / iterations))

            avg_cost += cost.data / iterations

        print("[Epoch: {:>4}], averaged cost = {:>.9}".format(1, avg_cost))

    # Runs a single iteration of forward propagation on the created network.
    def forward(self, sample):
        # layerOne = self.layerOne(sample)
        # layerTwo = self.layerTwo(layerOne)
        # return layerTwo
        out = self.convOne(sample)
        #out = self.layerTwo(out)
        #out = self.layerThree(out)
        out = out.view(out.size(0), -1)   # Flatten them for FC
        out = self.denseOne(out)
        #out = self.fc2(out)
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


