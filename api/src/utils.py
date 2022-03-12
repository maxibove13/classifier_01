#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Module with utilities classes and functions"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "03/22"

# Built-in modules
import os
import csv

# Third-party modules
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
import albumentations as A

import torch
from torch import nn
from skimage import io
import yaml

# Local modules
from src.models import models

# read yaml file
with open('config.yaml') as file:
    config = yaml.safe_load(file)

categories = ['sheep', 'cat', 'cow', 'butterfly', 'dog', 'squirrel', 'chicken', 'spider', 'elephant', 'horse']

test_transform = A.Compose(
    [
        A.Normalize(mean=[0.5206, 0.5008, 0.4145], std=[0.2676, 0.2633, 0.2806]),
        ToTensorV2(),
    ]
) 

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

def create_csv(set):
    processed_data_dir = os.path.join(config['data']['rootdir'], 'processed', config['data']['dataset'], set)
    images = os.listdir(os.path.join(processed_data_dir, 'images'))

    with open(os.path.join(processed_data_dir, 'annotations.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'category'])
        for im in images:
            writer.writerow([im , im[-5]])

def get_mean_std(dataset, batch_size, num_workers):
    # Load data
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # Initialize sums
    channels_sum, channel_squared_sum, num_batches = 0, 0, 0
    # Iterate over data and sum
    for num_batches, (data, _) in enumerate(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channel_squared_sum += torch.mean(data**2, dim=[0, 2, 3])

    # Calculate mean and std
    train_mean = channels_sum / num_batches
    train_std = (channel_squared_sum / num_batches - train_mean ** 2) ** 0.5

    return train_mean, train_std

def initialize_model(model_name, device):
    print("Initializing model...")
    
    # Define model
    model = models[model_name]
    if model_name == 'resnet18':
        model.fc = nn.Linear(512, 10)
    # Send it to device
    model.to(device)

    return model

def load_checkpoint(checkpoint_file, model, optimizer, lr, device):
    """
    Function to load a checkpoint.
    """
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # Update lr (otherwise it will have lr of old checkpoint)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

