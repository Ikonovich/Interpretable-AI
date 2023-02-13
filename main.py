import torch
import torchvision.transforms as transforms
from torch.autograd import Variable


from CoreModel import CoreNetwork
from DataHandler import get_tiny_imagenet_loader, mnist_baseline


if __name__ == "__main__":

    batch_size = 1

    # # Load mnist dataset
    # train_dataloader, test_dataloader = mnist_baseline(transform=transforms.ToTensor())

    # Load ImageNet dataset
    train_dataset, test_dataset = get_tiny_imagenet_loader(
        shuffle=False, transform=transforms.ToTensor())

    # dataset loaders
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)

    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)

    # Initialize the network
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = CoreNetwork().to(device)
    print("Model: " + str(model))

    learning_rate = 0.001
    loss_func = torch.nn.CrossEntropyLoss()  # Softmax is internally computed.
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    print('Training the Deep Learning network ...')
    train_cost = []
    train_accu = []

    training_epochs = 15
    total_batch = len(train_dataset) // batch_size

    print('Size of the training dataset is {}'.format(len(test_dataset)))
    print('Size of the testing dataset'.format(len(test_dataset)))
    print('Batch size is : {}'.format(batch_size))
    print('Total number of batches is : {0:2.0f}'.format(total_batch))
    print('\nTotal number of epochs is : {0:2.0f}'.format(training_epochs))

    for epoch in range(training_epochs):
        avg_cost = 0
        for i, (batch_X, batch_Y) in enumerate(train_dataloader):
            X = Variable(batch_X)  # image is already size of (28x28), no reshape
            Y = Variable(batch_Y)  # label is not one-hot encoded

            optimizer.zero_grad()  # <= initialization of the gradients

            # forward propagation
            hypothesis = model(X)
            cost = loss_func(hypothesis, Y)  # <= compute the loss function

            # Backward propagation
            cost.backward()  # <= compute the gradient of the loss/cost function
            optimizer.step()  # <= Update the gradients

            # Print some performance to monitor the training
            prediction = hypothesis.data.max(dim=1)[1]
            train_accu.append(((prediction.data == Y.data).float().mean()).item())
            train_cost.append(cost.item())
            if i % 200 == 0:
                print("Epoch= {},\t batch = {},\t cost = {:2.4f},\t accuracy = {}".format(epoch + 1, i, train_cost[-1],
                                                                                          train_accu[-1]))

            avg_cost += cost.data / total_batch

        print("[Epoch: {:>4}], averaged cost = {:>.9}".format(epoch + 1, avg_cost.item()))

    print('Learning Finished!')



