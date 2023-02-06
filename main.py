import numpy as np
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from keras.datasets import mnist
from torch.autograd import Variable

import DataHandler
from CoreModel import CoreNetwork


def mnist_baseline(network, batch_size=1):
    mnist_train = dsets.MNIST(root='MNIST_data/',
                              train=True,
                              transform=transforms.ToTensor(),
                              download=True)

    mnist_test = dsets.MNIST(root='MNIST_data/',
                             train=False,
                             transform=transforms.ToTensor(),
                             download=True)

    # dataset loader
    data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                              batch_size=batch_size,
                                              shuffle=True)

    print("Running network.")

    for i, (batch_X, batch_Y) in enumerate(data_loader):
        x = Variable(batch_X)  # image is already size of (28x28), no reshape
        y = Variable(batch_Y)  # label is not one-hot encoded

        optimizer.zero_grad()  # <= initialization of the gradients
        y_pred = network(x)
        print(f"Predicted class: {y_pred}")
        break;



if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = CoreNetwork().to(device)
    print("Model: " + str(model))

    learning_rate = 0.001
    criterion = torch.nn.CrossEntropyLoss()  # Softmax is internally computed.
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    mnist_baseline(model)


    # data = DataHandler.unpickle(1)
    #
    # for row in data[b'data']:
    #     print(len(row))
    #     break


