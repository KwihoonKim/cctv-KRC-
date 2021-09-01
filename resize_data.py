# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 23:54:27 2021

@author: Kwihoon
"""

import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

# image, mask load and save
image_directory = 'dataset_path/krc/13/'
save_directory = 'dataset_path/krc/42110A013910/'

SIZE = 224
image_dataset = []
train_label = []

images = os.listdir(image_directory)

for i, image_name in enumerate(images):
    image = cv2.imread(image_directory+image_name, 1)
    image = Image.fromarray(image)
    image = image.resize((SIZE, SIZE))
    image.save(save_directory+image_name.split('.')[0]+'.'+image_name.split('.')[1]+'.png')
    print(i)
