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
from torch.utils.data import random_split, DataLoader, sampler
import torchvision.transforms as transforms
import pandas as pd
import yaml

# Local modules
from utils import ImageDataset, get_mean_std, initialize_model, categories

# read yaml file
with open('config.yaml') as file:
    config = yaml.safe_load(file)

learning_rate = config['train']['learning_rate']
num_epochs = config['train']['num_epochs']
batch_size = config['train']['batch_size']
num_workers = config['train']['num_workers']

def train_model(model_name, learning_rate, batch_size, num_epochs, num_workers):

    # Define device
    device = torch.device(config['train']['device']) if torch.cuda.is_available() else 'cpu'
    print(
        f"Using device: {torch.cuda.get_device_name()}"
        f" with {torch.cuda.get_device_properties(0).total_memory/1024/1024/1024:.0f} GiB")


    # Define training dataset
    train_dataset = ImageDataset(
        csv_file = 'annotations.csv', 
        root_dir = os.path.join(config['data']['rootdir'], 'processed', config['data']['dataset'], 'train'),
        transform = transforms.ToTensor()
        )
    print(f"Defining training dataset {config['data']['dataset']} with {len(train_dataset)} samples")


    # Calculate dataset mean and std
    print("Calculating mean and std of training dataset...")
    train_mean, train_std = get_mean_std(train_dataset, batch_size, num_workers)
    train_mean = train_mean
    train_std = train_std
    print(f"mean: {train_mean}, std: {train_std}")

    # Define transforms
    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_mean, std=train_std)
    ])

    # Define validation dataset
    val_dataset = ImageDataset(
        csv_file = 'annotations.csv', 
        root_dir = os.path.join(config['data']['rootdir'], 'processed', config['data']['dataset'], 'val'),
        transform = trans
        )

    test_indices = list(range(len(val_dataset)))
    np.random.shuffle(test_indices)
    test_idx = test_indices[:]
    test_sampler = sampler.SubsetRandomSampler(test_idx)
    print(f"Defining validation dataset {config['data']['dataset']} with {len(test_sampler)} samples")

    # Recreate train_dataset dataset with new transforms
    print("Applying transforms...")
    train_dataset = ImageDataset(
        csv_file = 'annotations.csv', 
        root_dir = os.path.join(config['data']['rootdir'], 'processed', config['data']['dataset'], 'train'),
        transform = trans
        )
    train_indices = list(range(len(train_dataset)))
    np.random.shuffle(train_indices)
    train_idx = train_indices[:]
    train_sampler = sampler.SubsetRandomSampler(train_idx)



    # Load data
    print("Loading data...")
    train_loader = DataLoader(dataset=train_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(dataset=val_dataset, sampler=test_sampler, batch_size=batch_size, num_workers=num_workers)

    # Model
    model = initialize_model(model_name, device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Stochastic Gradient Descent
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                      momentum=0.9, weight_decay=5e-4)

    # Training loop
    print(f"Starting training: \n")
    print(
        f"  Model: {model_name}\n"
        f"  Training samples: {len(train_sampler)}\n"
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
    accs_val = []
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
        # Calculate average loss per image in the epoch and append it to list of losses
        epoch_loss = running_loss / len(train_dataset)
        losses.append(epoch_loss)
        del loss

        # Save model at the end of each epoch
        if config['models']['save']:
            torch.save({
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }, os.path.join(config['models']['rootdir'], config['models']['name']))

        # Calculate training and validation accuracy and append it to list of accuracies
        model.eval()
        accs_train.append(check_accuracy(train_loader, model, device))
        accs_val.append(check_accuracy(val_loader, model, device))
        model.train()

        x = np.arange(0, epoch+1)
        axs[0].plot(x, losses, 'b', label='Training loss')
        axs[1].plot(x, accs_train, 'r', label='Training accuracy')
        axs[1].plot(x, accs_val, 'b', label='Validation accuracy')
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
            f"Val Acc: {accs_val[-1]:.2f}"
            )
        

def check_accuracy(loader, model, device):
    num_correct = 0
    num_samples = 0
    
    with torch.no_grad():
        for ind, (x, y) in enumerate(loader):
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = torch.max(scores, 1)
            # print("in:",predictions)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        
        acc = float(num_correct)/float(num_samples)

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
        num_workers)
