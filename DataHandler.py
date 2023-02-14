from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms


from SortedFolderImageDataset import SortedFolderImageDataset

def mnist_baseline(transform):

    # MNIST dataset
    mnist_train = dsets.MNIST(root='MNIST_data/',
                              train=True,
                              transform=transform,
                              download=True)

    mnist_test = dsets.MNIST(root='MNIST_data/',
                             train=False,
                             transform=transform,
                             download=True)


    return mnist_train, mnist_test

def get_tiny_imagenet_loader(shuffle: bool = True):
    transform = transforms.ToTensor()


    paths = list()
    labels = list()

    paths.append("Data\\Training\\diver-n10565667")
    labels.append("Scuba Diver")
    paths.append("Data\\Training\\gazelle-n02423022")
    labels.append("Gazelle")
    paths.append("Data\\Training\\husky-n02110185")
    labels.append("Husky")
    paths.append("Data\\Training\\jacko-n03590841")
    labels.append("Jack-O-Lanter")
    paths.append("Data\\Training\\library-n03661043")
    labels.append("Library")
    paths.append("Data\\Training\\necklace-n03814906")
    labels.append("necklace")
    paths.append("Data\\Training\\pepper-n07720875")
    labels.append("Bell Pepper")

    train_set = SortedFolderImageDataset(paths, labels, percent_range=(0.0, 0.8), transform=transform, shuffle=shuffle)
    test_set = SortedFolderImageDataset(paths, labels, percent_range=(0.8, 1.0), transform=transform, shuffle=shuffle)


    return train_set, test_set
