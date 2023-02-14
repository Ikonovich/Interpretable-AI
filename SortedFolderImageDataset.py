import math
import os
import random

import torch
from torchvision.io import read_image
from torch.utils.data import Dataset

from PIL import Image, ImageOps

class SortedFolderImageDataset(Dataset):
    '''
    Used to access an image dataset that has been pre-sorted into folders, as a Pytorch dataset.
    '''

    def __init__(self, folder_paths: list[str], folder_labels: list, percent_range: tuple[float, float]=(0, 1), transform=None, shuffle: bool=True):
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

        # Stores a transformation function that should be applied to samples on retrieval.
        self.transform = transform

        # Reads each of the image paths into a list and adds the appropriate label to corresponding list.
        # Labels correspond to the index of the associated folder path in self.folder_labels
        images = list()
        labels = [i for i in range(len(self.folder_labels))]

        for i in range(len(folder_paths)):
            new_images, new_labels = self._process_folder(folder_paths[i], labels[i], percent_range)
            images += new_images
            labels += new_labels

        # Zip the images and labels into a single list of tuples and randomize the order.
        self.samples = list(zip(images, labels))

        if shuffle == True:
            random.shuffle(self.samples)


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

        startIndex = math.ceil(percent_range[0] * len(files))
        endIndex = min(math.ceil(percent_range[1] * len(files)), len(files))


        for index in range(startIndex, endIndex):
            filepath = files[index]
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
        image = Image.open(path)
        image = image.convert('RGB')
        # Resize and pad the image
        #image = self._resize(image)

        #image = self._pad(image)

        # Convert to grayscale
        #image = ImageOps.grayscale(image)

        # If self.transform isn't set to None, apply it to the image.
        if self.transform != None:
            image = self.transform(image)


        return image, torch.tensor(label, dtype=torch.long)

    def _resize(self, image, max_size: int = 500):

        width, height = image.size

        if (width > height):
            scale_factor = max_size / width
            final_height = int(height * scale_factor)
            new_size = (max_size, final_height)
        else:
            scale_factor = max_size / height
            final_width = int(width * scale_factor)
            new_size = (final_width, max_size)

        return image.resize(new_size)

    def _pad(self, image, size: tuple[int, int] = (500, 500)):

        # Generates a new white padding image
        result = Image.new(image.mode, (size[0], size[1]), 0)

        # Pastes the original image over it from the upper left corner.
        result.paste(image, (0, 0))

        return result