#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Script that defines a flask endpoint"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "03/22"

# Third-party model
from flask import Flask, request, jsonify

app = Flask(__name__)

# Create endpoint to infer data
@app.route('/infer', methods=["POST"])
def infer():
    # 1. load image
    # 2. Transform image to tensor
    # 3. Make inference
    # 4 return json data
    return jsonify({'results': 1})

