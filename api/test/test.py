#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Script to test the infer endpoint"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "03/22"

# Built-in module
import argparse

# Third-party modules
import requests


def test_endpoint(server):
    if server == 'local':
        url = 'http://127.0.0.1:5175'
    elif server == 'heroku':
        url = 'https://animal-classifier01.herokuapp.com'

    resp = requests.post(url + '/infer', files={'file': open('./data/download.jpeg', 'rb')})

    print(resp.text)


if __name__ == "__main__":
    # Initialize Argument Parser
    parser = argparse.ArgumentParser()
    # Add arguments we want to parse
    parser.add_argument("--server", type=str)
    # Read arguments from command line
    args = parser.parse_args()
    # Run function
    test_endpoint(args.server)
