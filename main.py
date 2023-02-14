import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader

from tqdm import tqdm


from CoreModel import CoreNetwork
from DataHandler import get_tiny_imagenet_loader, mnist_baseline


def run_resnet(dataloader: DataLoader):
    model = torchvision.models.resnet50(weights="DEFAULT")
    #model.eval().cuda()  # Needs CUDA, don't bother on CPUs

    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            #y_pred = model(x.cuda())
            y_pred = model(x)

            correct += (y_pred.argmax(axis=1) == y.data).sum().item()
            total += len(y)
    print(correct / total)

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
    run_resnet(test_dataloader)

    print('Size of the training dataset is {}'.format(len(train_dataset)))
    print('Size of the testing dataset is {}'.format(len(test_dataset)))


    print('Learning Finished!')
