import torch
import torchvision
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
        shuffle=False)

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

    # model = CoreNetwork().to(device)

    # Pre-trained resnet
    model = torchvision.models.resnet50(weights="DEFAULT")


    print("Model: " + str(model))

    training_epochs = 15
    total_batch = len(train_dataset) // batch_size

    print('Size of the training dataset is {}'.format(len(train_dataset)))
    print('Size of the testing dataset is {}'.format(len(test_dataset)))

    model.train(dataloader=train_dataloader)

    print('Learning Finished!')
