# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 14:00:15 2021

@author: Kwihoon
"""

import pandas as pd
import os
import cv2
from matplotlib import pyplot as plt
from PIL import Image

'일련번호'
num = '2/'
'시설코드'
code = '42110A001302/'
'방수문계측데이터 파일제목'
name = '강원_강릉_산북2방수문.xls'

df = pd.read_excel('dataset_path/krc/방수문계측데이터/'+name)

name = df.columns.tolist()
name.remove('관측일시')
name.remove('유입수위')

df_new = df.drop(name,axis=1)

for i in range(df.shape[0]):
    b = str(df_new.loc[i,'관측일시']).split(' ')[0]
    a = str(df_new.loc[i,'관측일시']).split(' ')[1]    
    df_new.loc[i,'date'] = b[0]+b[1]+b[2]+b[3]+b[5]+b[6]+b[8]+b[9]+a[0]+a[1]+a[3]+'0'
    print(i)
df = pd.DataFrame()
    
image_directory = 'dataset_path/krc/'+code
save_directory = 'dataset_path/krc/'+num

images = os.listdir(image_directory)

for i, image_name in enumerate(images):
    image = cv2.imread(image_directory+image_name,1)
    image = Image.fromarray(image)
    plt.imshow(image)
    image_name = image_name.split('.')[0]
    code = image_name.split('_')[0]
    image_name = image_name.split('_')[1]+image_name.split('_')[2]
    date_list = list(df_new['date'])

    if image_name in date_list:
        index = date_list.index(image_name)
        level = df_new['유입수위'][index]
        image_name = image_name+'_'+str(level)
        image.save(save_directory+code+image_name+'.jpg')
        
    print(i)
    
        
