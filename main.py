import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from SortedFolderImageDataset import SortedFolderImageDataset


def mnist_baseline(network, batch_size=32):

    # MNIST dataset
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




if __name__ == "__main__":

    paths = list()
    labels = list()

    paths.append("Data\\Training\\diver-n10565667")
    labels.append("Scuba Diver")

    data = SortedFolderImageDataset(paths, labels)
    #
    #
    # batch_size = 32
    #
    # # MNIST dataset
    # mnist_train = dsets.MNIST(root='MNIST_data/',
    #                           train=True,
    #                           transform=transforms.ToTensor(),
    #                           download=True)
    #
    # mnist_test = dsets.MNIST(root='MNIST_data/',
    #                          train=False,
    #                          transform=transforms.ToTensor(),
    #                          download=True)
    #
    # # dataset loader
    # data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
    #                                           batch_size=batch_size,
    #                                           shuffle=True)
    #
    #
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using {device} device")
    #
    # model = CoreNetwork().to(device)
    # print("Model: " + str(model))
    #
    # learning_rate = 0.001
    # loss_func = torch.nn.CrossEntropyLoss()  # Softmax is internally computed.
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    #
    # print('Training the Deep Learning network ...')
    # train_cost = []
    # train_accu = []
    #
    # training_epochs = 15
    # total_batch = len(mnist_train) // batch_size
    #
    # print('Size of the training dataset is {}'.format(mnist_train.data.size()))
    # print('Size of the testing dataset'.format(mnist_test.data.size()))
    # print('Batch size is : {}'.format(batch_size))
    # print('Total number of batches is : {0:2.0f}'.format(total_batch))
    # print('\nTotal number of epochs is : {0:2.0f}'.format(training_epochs))
    #
    # for epoch in range(training_epochs):
    #     avg_cost = 0
    #     for i, (batch_X, batch_Y) in enumerate(data_loader):
    #         X = Variable(batch_X)  # image is already size of (28x28), no reshape
    #         Y = Variable(batch_Y)  # label is not one-hot encoded
    #
    #         optimizer.zero_grad()  # <= initialization of the gradients
    #
    #         # forward propagation
    #         hypothesis = model(X)
    #         cost = loss_func(hypothesis, Y)  # <= compute the loss function
    #
    #         # Backward propagation
    #         cost.backward()  # <= compute the gradient of the loss/cost function
    #         optimizer.step()  # <= Update the gradients
    #
    #         # Print some performance to monitor the training
    #         prediction = hypothesis.data.max(dim=1)[1]
    #         train_accu.append(((prediction.data == Y.data).float().mean()).item())
    #         train_cost.append(cost.item())
    #         if i % 200 == 0:
    #             print("Epoch= {},\t batch = {},\t cost = {:2.4f},\t accuracy = {}".format(epoch + 1, i, train_cost[-1],
    #                                                                                       train_accu[-1]))
    #
    #         avg_cost += cost.data / total_batch
    #
    #     print("[Epoch: {:>4}], averaged cost = {:>.9}".format(epoch + 1, avg_cost.item()))
    #
    # print('Learning Finished!')



