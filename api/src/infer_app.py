#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Module to make inference"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "03/22"

# Built-in modules
import io
import os
import sys

# Third-party modules
import numpy as np
import torch
import yaml
from PIL import Image

sys.path.append('./training/src')

# Local modules
from training.src.process_images import resize_image
from training.src.utils import load_checkpoint, test_transform, initialize_model

# read yaml file
with open('config.yaml') as file:
    config = yaml.safe_load(file)

def transform_image(image, size):
    image = Image.open(io.BytesIO(image))
    # Resize image
    aspect = image.size[0]/image.size[1]
    w, h = image.size
    if w > h:
        image = image.resize((int(aspect * size),size))
        image = image.crop((0, 0, size, image.size[1]))
    else:
        image = image.resize((size, int(1/aspect * size)))
        image = image.crop((0, 0, image.size[0], size))
    image = np.asarray(image)
    return test_transform(image=image)["image"].unsqueeze(0)


def get_prediction(image_tensor):
    # device
    device = 'cpu'

    # Initialize model
    model = initialize_model("resnet18", torch.device(device))
    optimizer = torch.optim.SGD(model.parameters(), lr=config['train']['learning_rate'], momentum=0.9, weight_decay=5e-4)

    # Load model
    print("loading model")
    load_checkpoint(
            os.path.join(config['models']['rootdir'], config['models']['name']),
            model,
            optimizer,
            config['train']['learning_rate'],
            device
        )

    # Put model in evaluation mode
    model.eval()

    # Apply model to image
    outputs = model(image_tensor.to(device))
    _, predictions = torch.max(outputs.data, 1)

    return predictions

