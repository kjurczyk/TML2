"""Dataset setting and data loader for Fashion1000."""

import logging
import os
import tarfile
import torch
import torch.utils.data as data
import torchvision
from PIL import Image
from torch.utils.data import Dataset
#import keras
import matplotlib.pyplot as plt
import gzip
import numpy as np
from torchvision import transforms
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class_labels = [
    '0',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
]


class FASHION1000(Dataset):
    r"""fashion first 1000 Dataset.
    Args:
        root (string): Root directory of dataset where dataset file exist.
        train (bool, optional): If True, use the training split.
        download (bool, optional): If true, downloads the dataset
            from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that takes in
            an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``

    """


    def __init__(self, root, train=True, transform=None, download=False):
        # init params
        self.root = os.path.expanduser(root)
        ## put these files under data/fashion, AKA the root
        self.train_image = 'fourth_1000_train_images.gz'
        self.train_label = 'fourth_1000_train_labels.gz' 
        
        self.test_image = 't10k-images-idx3-ubyte.gz'
        self.test_label = 't10k-labels-idx1-ubyte.gz'
        self.train = train
        # Num of Train = 1000, Num ot Test 10000
        self.transform = transform
        

        # download dataset.
        if download:
            try:
                self.download_and_extract()
            except FileExistsError:
                pass
        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        self.images, self.labels = self.load_samples()
        
        

    def __getitem__(self, index):
         
        img, label = self.images[index], self.labels[index]
        if self.transform is not None:
            torch_tensor = torch.Tensor(img).float()
            torch_tensor = torch_tensor.view(1,28,28) #reshape image tensor to (Channel, Height, Weight)
            torch_tensor = self.transform(torch_tensor)
        label = np.int64(label).item()
        return torch_tensor, label

    def __len__(self):
        """Return size of dataset."""
        return self.dataset_size

    def _check_exists(self):
        """Check if dataset is download and in right place."""
        return os.path.exists(os.path.join(self.root, self.train_image))

    def load_samples(self):
        """Load dataset."""
        #files = [
        #   'first_1000_train_labels.gz', 'first_1000_train_images.gz',
        #   'second_1000_train_labels.gz', 'second_1000_train_images.gz',
        #   'third_1000_train_labels.gz', 'third_1000_train_images.gz',
        #   'fourth_1000_train_labels.gz', 'fourth_1000_train_images.gz',
        #   't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
        #   ]
        """Load sample images from dataset."""
        
       
        if self.train: 
            image = os.path.join(self.root, self.train_image)
            label = os.path.join(self.root, self.train_label)
            
        else: # load test .gz files
            image = os.path.join(self.root, self.test_image)
            label = os.path.join(self.root, self.test_label)
            
            
        f = gzip.open(image, "rb")
        l = gzip.open(label, "rb")
        
        if self.train:
            labels = np.frombuffer(l.read(), np.uint8)
        
            data_set = np.frombuffer(f.read(), np.uint8).reshape(len(labels), 28, 28)
            
        else:
            labels = np.frombuffer(l.read(), np.uint8, offset=8)
            data_set = np.frombuffer(f.read(), np.uint8, offset=16).reshape(len(labels), 28, 28)

        images = data_set
        
        
        self.dataset_size = labels.shape[0]
        f.close()
        l.close()
        
        return images, labels
