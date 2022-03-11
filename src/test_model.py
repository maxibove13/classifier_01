#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Script to predict a image class using a trained model"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "03/22"

# Built-in modules
import argparse
import os

# Third-party modules
import numpy as np
from torchvision import datasets
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import sampler, DataLoader
from torchvision import transforms
import torch
from torch import tensor
from torch.autograd import Variable
import yaml

# Local modules
from process_images import resize_image
from utils import initialize_model, load_checkpoint, test_transform, categories

# read yaml file
with open('config.yaml') as file:
    config = yaml.safe_load(file)

def test_model(model_name, set):

    # Path to test images from
    image_path = os.path.join(
            config['data']['rootdir'],
            'processed',
            config['data']['dataset'],
            set,
            'images',
        )

    # Define device
    device = torch.device(config['train']['device']) if torch.cuda.is_available() else 'cpu'
    print(
    f"Using device: {torch.cuda.get_device_name()}"
    f" with {torch.cuda.get_device_properties(0).total_memory/1024/1024/1024:.0f} GiB")

    print(f"Testing model on {image_path} images path")

    # Initialize model
    model = initialize_model(model_name, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['train']['learning_rate'],
                    momentum=0.9, weight_decay=5e-4)
    # Load checkpoint
    print("Loading config['models']['name'] checkpoint")
    load_checkpoint(
        os.path.join(config['models']['rootdir'], config['models']['name']),
        model,
        optimizer,
        config['train']['learning_rate'],
        device
    )

    # Put model in evaluation mode
    model.eval()

    num_correct = 0
    num_samples = 0
    for im in os.listdir(image_path):
        image = Image.open(os.path.join(image_path, im))
        image_arr = np.asarray(image)

        with torch.no_grad():
            # Apply transform to image, unsqueeze send to device
            x = test_transform(image=image_arr)["image"].unsqueeze(0).to(device)
            # Apply model to x
            scores = model(x)
            # Get prediction
            _, predictions = torch.max(scores, 1)
            # Sum correct inputs
            if (int(predictions.cpu()) == int(im[-5])):
                print(f"{categories[int(predictions.cpu())]} predicted correctly")
                num_correct += 1
            else:
                print("ALERT !!!")
                print(f"{categories[int(im[-5])]} predicted incorrectly")
            num_samples += predictions.size(0)
    # Print accuracy
    print(float(num_correct)/float(num_samples))


if __name__ == "__main__":
    # Initialize Argument Parser
    parser = argparse.ArgumentParser()
    # Add arguments we want to parse
    parser.add_argument("--model", type=str)
    parser.add_argument("--set", type=str)
    # Read arguments from command line
    args = parser.parse_args()
    # Run function
    test_model(args.model, args.set)