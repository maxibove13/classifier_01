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
from PIL import Image
import pandas as pd
import yaml

# read yaml file
with open('config.yaml') as file:
    config = yaml.safe_load(file)

size = config['data']['size']

def resize_images(size, multiprocess=True):
    # Raw data directory
    data_dir = os.path.join(config['data']['rootdir'], 'raw',config['data']['dataset'])
    # List of categories directories
    categories_dir = os.listdir(data_dir)

    # Create 'processed' and dataset folder if they dont exist
    if 'processed' not in os.listdir(os.path.join(config['data']['rootdir'])):
        os.mkdir(os.path.join(config['data']['rootdir'], 'processed'))
    if config['data']['dataset'] not in os.listdir(os.path.join(config['data']['rootdir'], 'processed')):
        os.mkdir(os.path.join(config['data']['rootdir'], 'processed', config['data']['dataset']))
    if 'images' not in os.listdir(os.path.join(config['data']['rootdir'], 'processed', config['data']['dataset'])):
        os.mkdir(os.path.join(config['data']['rootdir'], 'processed', config['data']['dataset'], 'images'))

    # Initialize categories and images list
    categories = []
    images = []
    i = 0
    # Iterate over all categories directories
    for idx_cat, category_dir in enumerate(categories_dir):
        # Images in this category
        images_per_category = os.listdir(os.path.join(data_dir, category_dir))
        # iterate over all images in this category directory (use different cores)
        processes = []
        if multiprocess:
            pool = multiprocessing.Pool()
            for idx_im, im in enumerate(images_per_category):
                # Define the process and target
                pool.apply_async(process_image, args=(category_dir, im, size, idx_cat, idx_im, data_dir, images_per_category))
            pool.close()
            pool.join()
        else:
            for idx_im, im in enumerate(images_per_category):
                process_image(category_dir, im, size, idx_cat, idx_im, data_dir, images_per_category)
        
        
def process_image(category_dir, im, size, idx_cat, idx_im, data_dir, images_per_category):
    image = Image.open(os.path.join(data_dir, category_dir,im)) # Open image
    aspect = image.size[0]/image.size[1] # Define aspect ratio
    w, h = image.size
    # Force the min dimension to be size, adjust the other one to mantain dimension
    # Crop from the top or right to reduce dimensions to (size, size)
    size = config['data']['size']
    if w > h:
        image = image.resize((int(aspect * size),size))
        image = image.crop((0, 0, size, image.size[1]))
    else:
        image = image.resize((size, int(1/aspect * size)))
        image = image.crop((0, 0, image.size[0], size))

    image_fn = os.path.join(config['data']['rootdir'], 'processed', config['data']['dataset'], 'images',f"animals_{idx_im}_{idx_cat}.png")
    print(f"Saving processed {idx_im}/{len(images_per_category)} {category_dir} image in: {image_fn}")
    # Save image
    image = image.convert('RGB') # Make sure the imae is in RGB
    image.save(image_fn)

if __name__ == "__main__":
    resize_images(size, multiprocess=True)
