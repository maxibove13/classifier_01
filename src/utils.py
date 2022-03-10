#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Module with utilities classes and functions"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "03/22"

# Built-in modules
import os

# Third-party modules
import pandas as pd
from torch.utils.data import Dataset
import torch
from skimage import io
import yaml

categories = ['sheep', 'cat', 'cow', 'butterfly', 'dog', 'squirrel', 'chicken', 'spider', 'elephant', 'horse']

class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(os.path.join(root_dir, csv_file))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, 'images', self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)