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
from utils import initialize_model, load_checkpoint, test_transform

# read yaml file
with open('config.yaml') as file:
    config = yaml.safe_load(file)

def infer(model_name, num_samples):

    processed_image_path = os.path.join(
            config['data']['rootdir'],
            'processed',
            config['data']['dataset'],
            'infer',
        )
    for im in os.listdir(processed_image_path):
        os.remove(os.path.join(processed_image_path, im))
    

    # Define device
    device = torch.device(config['train']['device']) if torch.cuda.is_available() else 'cpu'
    print(
    f"Using device: {torch.cuda.get_device_name()}"
    f" with {torch.cuda.get_device_properties(0).total_memory/1024/1024/1024:.0f} GiB")

    # Define data directory
    data_dir = os.path.join(config['data']['rootdir'], 'raw', config['data']['dataset'],'test')
    # Load data
    data = datasets.ImageFolder(
        data_dir,
        transform=test_transform)
    # Define classes
    classes = data.classes

    # Shuffle the images list
    indices = list(range(len(data)))
    np.random.shuffle(indices)

    # Choose number of infering samples from dataset
    idx = indices[:num_samples]

    # Iterate over sample and resize them
    for i in idx:
        # Open image
        image_path = data.samples[i][0]
        # Resize image
        image_proc = resize_image(image_path, 256)
        # Save image
        image = image_proc.convert("RGB")
        image_fn = os.path.join(processed_image_path, f"test_{i}_{data.samples[i][1]}.png")
        # print(i,classes[data.samples[i][1]])
        image.save(image_fn)

     # Initialize model
    model = initialize_model(model_name, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['train']['learning_rate'],
                    momentum=0.9, weight_decay=5e-4)
    load_checkpoint(
        os.path.join(config['models']['rootdir'], config['models']['name']),
        model,
        optimizer,
        config['train']['learning_rate'],
        device
    )

    model.eval() # Put model in evaluation mode

    num_correct = 0
    num_samples = 0
    for im in os.listdir(processed_image_path):
        image = Image.open(os.path.join(processed_image_path, im))
        image_arr = np.asarray(image)

        with torch.no_grad():
            x = test_transform(image=image_arr)["image"].unsqueeze(0).to(device)
            scores = model(x)
            _, predictions = torch.max(scores, 1)
            print(classes[int(predictions)], classes[int(im[-5])])
            if (int(predictions.cpu()) == int(im[-5])):
                num_correct += 1
            num_samples += predictions.size(0)

    print(float(num_correct)/float(num_samples))


if __name__ == "__main__":
    # Initialize Argument Parser
    parser = argparse.ArgumentParser()
    # Add arguments we want to parse
    parser.add_argument("--samples", type=int)
    parser.add_argument("--model", type=str)
    # Read arguments from command line
    args = parser.parse_args()
    # Run function
    num_samples = 3
    infer(args.model, args.samples)