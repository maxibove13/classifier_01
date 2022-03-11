#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Script that defines a flask endpoint"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "03/22"

# Built-in modules
import random

# Third-party model
from flask import Flask, request, jsonify

# Local modules
from app.infer_app import transform_image, get_prediction
from app.utils import categories

app = Flask(__name__)

allowed_extensions = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# Create endpoint to infer data
@app.route('/infer', methods=["POST"])
def infer():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supporte'})

    try:
        # Load image
        img_bytes = file.read()
        # Transform iamge to tensor
        tensor = transform_image(img_bytes, 256)
        # Make prediction
        prediction = get_prediction(tensor)
        # print(prediction)
        # Return json data
        data = {'prediction': prediction.item(), 'class_name': categories[prediction.item()]}
        return jsonify(data)
    except:
        return jsonify({'error': 'prediction error'})

if __name__ == "__main__":
    port = 5175
    print(port)
    app.run(use_reloader=False, debug=True, port=port)