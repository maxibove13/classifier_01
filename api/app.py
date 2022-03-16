#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Run flask app and define endpoints"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "03/22"

# Built-in modules
import random

# Third-party model
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

# Local modules
from src.infer_app import transform_image, get_prediction
from src.utils import categories

# Instance of Flask class.
app = Flask(__name__)

# In order to make a fetch request we have to use CORS that is an HTTP-header based mechanism that allows a server
# to indicate any origins other than its own from which the browser should permit loading resources.
cors = CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'

allowed_extensions = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# # Create endpoint to infer data
@app.route('/infer', methods=["POST"])
@cross_origin()
def infer():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})
    # Make inference on image
    try:
        # Load image
        img_bytes = file.read()
        # Transform image to tensor
        tensor = transform_image(img_bytes, 256)
        # Make prediction
        prediction = get_prediction(tensor)
        # Return json data
        data = {'prediction': prediction.item(), 'class_name': categories[prediction.item()]}
        return jsonify(data)
    except:
        return jsonify({'error': 'prediction error'})


@app.route('/members', methods=["GET"])
def test():
   return {"data": ["data1", "data2", "data3"]}

if __name__ == "__main__":
    port = 5175
    app.run(use_reloader=False, debug=True, port=port)