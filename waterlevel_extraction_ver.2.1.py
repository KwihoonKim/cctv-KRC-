# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 22:30:05 2021

@author: Kwihoon
"""

import pandas as pd
import os

image_directory = 'dataset_path/krc/2/42110A001302/'
# =============================================================================
# image_directory = 'dataset_path/krc/test/'
# =============================================================================
images = os.listdir(image_directory)

level_total = []
for i, image_name in enumerate(images):   
    a = image_name.split('.')
    b = len(a)
    if b == 3:
        image_name = image_name.split('_')[1]
        level = image_name.split('.')[0]+'.'+image_name.split('.')[1]
        level_total.append(level)

level_repre = set(level_total)

from matplotlib import pyplot as plt
xtick = [0.6,0.7,0.8,0.9,1.0,1.1]
xtick = [0,7,21,28,32]
ytick = [0,50,100,150,200,250,300,350]

plt.gca().invert_yaxis()
plt.plot(level_total)
plt.yticks(xtick)

plt.hist(level_total, bins = 12, rwidth = 0.9)
plt.xticks(xtick)
plt.gca().invert_xaxis()
# =============================================================================
# 
# k = 0
# for i in level_total:
#     if float(i) >= 0.91:
#         k += 1
# =============================================================================
    
