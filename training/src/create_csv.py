#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Script to process images. In this case we resize and crop the image to a selected square size mantaining aspect ratio."""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "03/22"

# Built-in modules
import os
import multiprocessing

# Third-party modules
import csv
from PIL import Image
import pandas as pd
import yaml

# read yaml file
with open('./api/config.yaml') as file:
    config = yaml.safe_load(file)

def create_csv():
    processed_data_dir = os.path.join(config['data']['rootdir'], 'processed', config['data']['dataset'])
    images = os.listdir(os.path.join(processed_data_dir, 'images'))

    with open(os.path.join(processed_data_dir, 'annotations.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'category'])
        for im in images:
            writer.writerow([im , im[-5]])


if __name__ == "__main__":
    create_csv()