import os
import cv2
import torch
import json
import time
import base64
import requests
import transformers
import numpy as np

from torchvision import transforms
from PIL import Image
from flask import Flask, jsonify, request

from lerobot.common.datasets.lerobot_dataset import MultiDatasetforDistTraining

def dict_json_serlizable(d:dict):
    #convert all numpy ndarray in dict to list recursively
    for k,v in d.items():
        if isinstance(v, np.ndarray):
            d[k] = v.tolist()
            print("converted numpy ndarray:{} to list".format(k))
        elif isinstance(v, dict):
            dict_json_serlizable(v)
        elif isinstance(v, list):
            for i in range(len(v)):
                if isinstance(v[i], np.ndarray):
                    v[i] = v[i].tolist()
                    print("converted numpy ndarray: {} to Obj.list".format(k))
    return d

def numpy_to_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist() # Convert NumPy array to a list
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def decode_b64_image(b64image):
    # Decode the base64 image string
    str_decode = base64.b64decode(b64image)
    np_image = np.frombuffer(str_decode, np.uint8)
    image_cv2 = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    # Get h,w of image
    h, w, _ = image_cv2.shape
    #Crop the image retain the mid area to be a square
    if h > w:
        image_cv2 = image_cv2[int((h-w)/2):int((h+w)/2), :, :]
    elif w > h:
        image_cv2 = image_cv2[:, int((w-h)/2):int((w+h)/2), :]
    return image_cv2

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Cuda realtime vla model inference service!'