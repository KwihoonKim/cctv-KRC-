# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 15:07:48 2021

@author: Kwihoon
"""

import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import tifffile

num = '4'
# image, mask load and save
image_directory = 'dataset_path/krc/'+num+'/'+num+'_images/'
annotation_directory = 'dataset_path/krc/'+num+'/'+num+'_annot_img/'
save_directory = 'dataset_path/krc/'+num+'/'+num+'_annot_gen/'

SIZE = 224

images = os.listdir(image_directory)
annot = os.listdir(annotation_directory)

'수위값 개수 세기'
image_dataset = []
for i, image_name in enumerate(images):
    image_name = image_name.split('_')[1]
    image_name = image_name.split('.')[0]+'.'+image_name.split('.')[1]
    image_dataset.append(image_name)
    print(i)
image_dataset_num = set(image_dataset)

annot_dataset = []
for i, image_name in enumerate(annot):
    image_name = image_name.split('_')[1]+'.'+image_name.split('_')[2]
    annot_dataset.append(image_name)
    print(i)
annot_dataset_num = set(annot_dataset)

    
'annotation 파일 복제'    
for i, image_name in enumerate(images):
    image_name0 = image_name.split('_')[0]
    image_name1 = image_name.split('_')[1]
    image_name2 = image_name1.split('.')[0]+'.'+image_name1.split('.')[1]   
    for j, annotation_name in enumerate(annot):
        image = cv2.imread(annotation_directory+annotation_name, 0)      
        annotation_name = annotation_name.split('_')[1]+'.'+annotation_name.split('_')[2]
        image[image>0] = 1
        if float(annotation_name) == float(image_name2):
            with tifffile.TiffWriter(save_directory+image_name0+'_'+image_name2+'.tiff') as tiff:
                tiff.save(image, compress=6,)# photometric='rgb')
            break
    print(i)
  

image_directory = 'dataset_path/krc/4/4_annot_sample/'
annotation_directory = 'dataset_path/krc/4/4_annot_img/'
images = os.listdir(image_directory)
annot = os.listdir(annotation_directory)

for i, annotation_name in enumerate(annot):
    image = cv2.imread(annotation_directory+annotation_name, 0)
    plt.figure(figsize=(32,16))
    plt.imshow(image)
    plt.title(annotation_name, fontsize=250)
    print(i)
          




