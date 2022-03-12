#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Script to test the infer endpoint"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "03/22"

# Third-party modules
import requests

# Heroku URL
# https://animal-classifier01.herokuapp.com/infer



resp = requests.post("http://localhost:5000/infer", files={'file': open('./data/download.jpeg', 'rb')})

print(resp.text)