#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Script to train a classifier for computer vision datasets"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "03/22"

# Built-in modules
import argparse
import os

# Third-party modules
import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from torchvision.models import googlenet
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import yaml

# Local modules
from utils import ImageDataset
from models import CNN, models

# read yaml file
with open('config.yaml') as file:
    config = yaml.safe_load(file)

learning_rate = config['train']['learning_rate']
num_epochs = config['train']['num_epochs']
batch_size = config['train']['batch_size']
num_workers = config['train']['num_workers']
split = config['train']['split']

def train_model(model_name, learning_rate, batch_size, num_epochs, num_workers, split):

    # Define device
    device = torch.device(config['train']['device']) if torch.cuda.is_available() else 'cpu'
    print(
        f"Using device: {torch.cuda.get_device_name()}"
        f" with {torch.cuda.get_device_properties(0).total_memory/1024/1024/1024:.0f} GiB")


    # Define dataset
    dataset = ImageDataset(
        csv_file = 'annotations.csv', 
        root_dir = os.path.join(config['data']['rootdir'], 'processed', config['data']['dataset']),
        transform = transforms.ToTensor()
        )
    print(f"Defining dataset {config['data']['dataset']} with {len(dataset)} samples")


    # Calculate dataset mean and std
    print("Calculating mean and std of dataset...")
    # Load dataset
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    channels_sum, channel_squared_sum, num_batches = 0, 0, 0
    for num_batches, (data, _) in enumerate(train_loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channel_squared_sum += torch.mean(data**2, dim=[0, 2, 3])

    train_mean = channels_sum / num_batches
    train_std = (channel_squared_sum / num_batches - train_mean ** 2) ** 0.5



    # Define transforms
    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_mean, std=train_std)
    ])


    # Create dataset with new transforms
    print("Applying transforms...")
    dataset = ImageDataset(
        csv_file = 'annotations.csv', 
        root_dir = os.path.join(config['data']['rootdir'], 'processed', config['data']['dataset']),
        transform = trans
        )

    # Divide training and testing dataset
    train_set, test_set = random_split(dataset, [int(float(split)*len(dataset)), len(dataset) - int(float(split)*len(dataset))])
    print(
        f"Training / Testing split: {split}\n"
        f"Training {len(train_set)} and testing {len(test_set)} samples. ")

    # Load data
    print("Loading data...")
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Model
    print("Initializing model...")
    model = models[model_name]
    if model_name == 'resnet18':
        model.fc = nn.Linear(512, 10)
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = 

    # Training loop
    print(f"Starting training: \n")
    print(
        f"  Model: {model_name}\n"
        f"  Training samples: {len(train_set)}\n"
        f"  Number of epochs: {num_epochs}\n"
        f"  Mini batch size: {batch_size}\n"
        f"  Number of batches: {len(train_loader)}\n"
        f"  Learning rate: {learning_rate}\n"
    )

    fig, axs = plt.subplots(2,1, sharex=True)
    axs[0].set_title('Training loss')
    axs[0].set(ylabel='loss')
    axs[0].grid()
    axs[1].set_title('Training and validation accuracy')
    axs[1].set(xlabel='epochs')
    axs[1].set(ylabel='accuracy')
    axs[1].grid()

    accs_train = []
    accs_test = []
    losses = []
    for epoch in range(num_epochs):
        for idx, (data, targets) in enumerate(train_loader):
            running_loss = 0.0
            data = data.to(device)
            # Send data to device
            targets = targets.to(device=device)

            # Forward
            scores = model(data)
            loss = criterion(scores, targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            # Gradient descent (update weights)
            optimizer.step()
            # Print GPU memory allocation
            if idx == 1 and epoch == 0:
                print(f" GPU memory usage: {torch.cuda.memory_allocated(device=1)/1024/1024:.2f} MiB")
        
            running_loss += loss.item() * batch_size
            del loss
        # Calculate average loss per image in the epoch and append it to list of losses
        epoch_loss = running_loss / len(train_set)
        losses.append(epoch_loss)
        # Print progress every epoch


        # Calculate training and testing accuracy and append it to list of accuracies
        accs_train.append(check_accuracy(train_loader, model, device))
        accs_test.append(check_accuracy(test_loader, model, device))

        x = np.arange(0, epoch+1)
        axs[0].plot(x, losses, 'b', label='Training loss')
        axs[1].plot(x, accs_train, 'r', label='Training accuracy')
        axs[1].plot(x, accs_test, 'b', label='Testing accuracy')
        if epoch == 0:
            axs[0].legend(loc='upper right')
            axs[1].legend(loc='lower right')
            axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
            axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.savefig(os.path.join(config['figures']['dir'], 'loss_acc_evol.png'))

        print(
            f"Epoch [{epoch+1:03d}/{num_epochs} - "
            f"Loss: {epoch_loss:.4f}] - "
            f"Train Acc: {accs_train[-1]:.2f} - "
            f"Test Acc: {accs_test[-1]:.2f}"
            )
        
        # Save model at the end of each epoch
        if config['model']['save']:
            torch.save({
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }, os.path.join(config['models']['rootdir'], config['models']['name']))

def check_accuracy(loader, model, device):
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)

            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        
        acc = float(num_correct)/float(num_samples)

        # print(f"Got {num_correct} / {num_samples} with accuracy {acc*100:.2f}")

    model.train()
    return acc


if __name__ == "__main__":
    # Initialize Argument Parser
    parser = argparse.ArgumentParser()
    # Add arguments we want to parse
    parser.add_argument("--model", type=str)
    # Read arguments from command line
    args = parser.parse_args()
    # Run function
    train_model(
        args.model,
        learning_rate, 
        batch_size, 
        num_epochs, 
        num_workers,
        split)
