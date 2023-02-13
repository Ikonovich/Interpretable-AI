import math
import os
import random

from torchvision.io import read_image
from torch.utils.data import Dataset


class SortedFolderImageDataset(Dataset):
    '''
    Used to access an image dataset that has been pre-sorted into folders, as a Pytorch dataset.
    '''

    def __init__(self, folder_paths: list[str], folder_labels: list, percent_range: tuple[float, float] = (0, 1)):
        '''
        Indexes of labels must match the appropriate folder path's index in the corresponding list.
        :param folder_paths: A list of folder paths to be used as the dataset.
        :param folder_labels: A list of labels for the provided folders.
         :param range: The range of image paths that will be loaded from the provided folder, as
         fractions of 1.
        '''

        if len(folder_paths) != len(folder_labels):
            raise IndexError("Number of folder paths and number of labels must match.")


        self.folders = folder_paths
        self.folder_labels = folder_labels

        # Stores each individual sample-label pair.
        self.samples = list()

        # Reads each of the image paths into a list and adds the appropriate label to s corresponding list.
        images = list()
        labels = list()

        for i in range(len(folder_paths)):
            folder_images, folder_labels = self._process_folder(folder_paths[i], folder_labels[i], percent_range)
            images += folder_images
            labels += folder_labels

        # Zip the images and labels into a single list of tuples and randomize the order.
        self.samples = list(zip(images, labels))
        random.shuffle(self.samples)

        #

    def _process_folder(self, folder_path: str, label, percent_range: tuple[float, float]):
        '''
        This helper function returns two lists containing all of paths in a folder and a corresponding
        list of labels, for convenience in init.
        :param folder_path: The relative path to the folder.
        :param label: The label of the provided folder.
        :return: A tuple containing (list of image paths in folder, list of labels)
        '''
        images = list()
        labels = list()
        parent_dir = os.getcwd()
        files = os.listdir(os.path.join(parent_dir, folder_path))

        startPercent = percent_range[0]
        endPercent = percent_range[1]

        startIndex = math.floor(percent_range[0] * len(files))
        endIndex = 0


        for filepath in files:
            img = os.path.join(folder_path, filepath)
            images.append(img)
            labels.append(label)

        return images, labels


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        '''
        :param index: The index of the data sample to return.
        :return: The image referenced by the samples at the provided index, as a tensor, and the label of the image.
        '''
        sample = self.samples[index]
        path = sample[0]
        label = sample[1]
        image = read_image(path)

        return image, label
