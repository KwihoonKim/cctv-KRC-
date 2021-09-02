# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 23:32:42 2021

@author: USESR
"""

from resnet_unet_model import resnet18_unet_model, resnet34_unet_model,\
                                resnet50_unet_model, resnet101_unet_model,\
                                resnet152_unet_model, vgg16_unet_model, vgg16_unet_model,\
                                se_resnet18_unet_model, se_resnet34_unet_model,\
                                se_resnet101_unet_model, se_resnet152_unet_model,\
                                resnext50_unet_model, resnext101_unet_model,\
                                senet154
                                
                       
model = resnet34_unet_model()

import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

# image, mask load and save
image_directory = 'image/'
mask_directory = 'annotation/'

SIZE = 256
image_dataset = []
mask_dataset = []

images = os.listdir(image_directory)
masks = os.listdir(mask_directory)

for i, image_name in enumerate(images):
    if (image_name.split('.')[2] == 'jpg'):
        image = cv2.imread(image_directory+image_name, 1)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        image_dataset.append(np.array(image))
        
for i, image_name in enumerate(masks):
    if (image_name.split('.')[2] == 'tiff'):
        image = cv2.imread(mask_directory+image_name, 0)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        image = np.array(image)
        mask_dataset.append(image)
        
waterlevel_dataset = []
for i, image_name in enumerate(masks):
    image = cv2.imread(mask_directory+image_name, 0)
    image = Image.fromarray(image)
    image = image.resize((SIZE, SIZE))
    width = int(sum(list(np.array(image)[:,128])))
    waterlevel = image_name.split('_')[1]
    waterlevel = waterlevel.split('.')[0] + '.' + waterlevel.split('.')[1]
    waterlevel = float(waterlevel)
    b = (waterlevel, width)
    waterlevel_dataset.append(b)

'Zipping mask image and waterlevel'
zip_mask_waterlevel = []
image_dataset = np.array(image_dataset)
data_length = image_dataset.shape[0]
for i in range(data_length):
    mask_waterlevel = (mask_dataset[i], waterlevel_dataset[i])
    zip_mask_waterlevel.append(mask_waterlevel)

zip_mask_waterlevel = np.array(list(zip_mask_waterlevel), dtype = object)


'Train Test Validation Split'
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(image_dataset, zip_mask_waterlevel, test_size = 0.30)#, random_state = 1)
X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size = 0.33, random_state = 0)

'Mask_dataset Regression Equation'
Y_test, waterlevel_test = np.array(Y_test).T
Y_val, waterlevel_val = np.array(Y_val).T
Y_train, waterlevel_train = np.array(Y_train).T

Y_train = np.array(list(Y_train))
waterlevel_train = np.array(list(waterlevel_train)).T
Y_train = np.expand_dims(Y_train, axis=3)

Y_test = np.array(list(Y_test))
waterlevel_test = np.array(list(waterlevel_test)).T
Y_test = np.expand_dims(Y_test, axis=3)

Y_val = np.array(list(Y_val))
waterlevel_val = np.array(list(waterlevel_val)).T
Y_val = np.expand_dims(Y_val, axis=3)

from numpy.polynomial.polynomial import polyfit
y_train, x_train = waterlevel_train
b,m = polyfit(x_train, y_train, 1)
y_hat_train = b + m*x_train

from sklearn.metrics import r2_score
r2_train = round(r2_score(y_train, y_hat_train),2)

'픽셀 vs. 수위 회귀식 Train'
plt.plot(x_train, y_train, '.')
plt.plot(x_train, y_hat_train, '-', label = 'test')
plt.annotate('R2='+str(r2_train), xy=(160,1.2))
plt.xlabel('Pixels')
plt.ylabel('Waterlevel (m)')
plt.legend(['R2 = ' +str(r2_train)])
plt.show()



# =============================================================================
# 'Data Augmentation'
# from skimage.transform import rotate
# from skimage.util import random_noise
# from skimage.filters import gaussian
# 
# image_dataset1 = []
# mask_dataset1 = []
# waterlevel_dataset1 = []
# 
# data_length = len(image_dataset)
# 
# for i in range (data_length):
#     for j in range(1):
#         image_data_list = rotate(image_dataset[i], angle = j*10, mode = 'edge')
#         mask_data_list = rotate(mask_dataset[i], angle = j*10, mode = 'edge')
#         image_dataset1.append(image_data_list)
#         mask_dataset1.append(mask_data_list)
# =============================================================================


image_dataset_final = np.array(image_dataset)*255
mask_dataset_final = np.array(mask_dataset)*255

image_dataset_final = np.uint8(image_dataset_final)
mask_dataset_final = np.uint8(mask_dataset_final)

image_dataset_final = np.float32(image_dataset_final)
mask_dataset_final = np.float32(mask_dataset_final)

history = model.fit(X_train, Y_train, batch_size = 16, verbose = 1, epochs = 100, validation_data = (X_val, Y_val), shuffle = False)

# =============================================================================
# model.save('unet_'+model_name+'_epoch200_normx.hdf5')
# =============================================================================


# Evaluate model
# =============================================================================
# _, f1 = model.evaluate(X_test, Y_test)
# print('f1_score = ', (f1*100.0))
# =============================================================================

# plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label = 'Training loss')
plt.plot(epochs, val_loss, 'r', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['f1-score']
val_acc = history.history['val_f1-score']
plt.plot(epochs, acc, 'y', label = 'Training f1')
plt.plot(epochs, val_acc, 'r', label='Validation f1')
plt.title('Training and validation f1')
plt.xlabel('Epochs')
plt.ylabel('f1')
plt.legend()
plt.show()


for i in range(1):
    train_img = X_train[i]
    train_img_input = np.expand_dims(train_img, 0)
    
    ground_truth = Y_train[i]
    
    plt.figure(figsize=(16, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(train_img/256.)
    
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(ground_truth, cmap='gray')

    
'테스트 이미지 평가'
iou_score_list = []
f1_score_list = []
waterlevel_hat_list = []
width_test_list = []
waterlevel_answer, pixel_test = waterlevel_test
num = len(waterlevel_answer)

for i in range(num):
    '평가지표 계산'
    test_img = X_test[i]
    test_img_input = np.expand_dims(test_img, 0)
    prediction = (model.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)
    prediction = np.expand_dims(prediction, 2)
    ground_truth = Y_test[i]
    list1 = prediction - ground_truth
    list1 = np.array(list1).flatten().tolist()
    list1 = list(list1)
    c0 = len(list1)
    c1 = list1.count(255)
    c2 = list1.count(1)
    list2 = prediction*ground_truth
    c3 = np.sum(list2)

    iou_score = c3/(c1+c2+c3)
    iou_score = round(iou_score,2)
    f1_score = 2*c3/(2*c3+c1+c2)
    f1_score = round(f1_score,2)
    
    iou_score_list.append(iou_score)
    f1_score_list.append(f1_score)
    
    '그림 출력'
    plt.figure(figsize=(16, 8))
    
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img/256.)
    
    plt.subplot(232)
    plt.title('Testing Label (ground-truth)')
    plt.imshow(ground_truth, cmap='gray')
    
    plt.subplot(233)
    plt.title('Modeled Image (f1:'+str(f1_score)+')')
    plt.imshow(prediction, cmap='gray')
    
    fig = plt.gcf()
    fig.savefig('resnet34_'+str(i+1)+'.png', dpi = fig.dpi)
    plt.close()
    
    '수위계산'
    width = int(sum(list(np.array(prediction)[:,128])))
    width_test_list.append(width)
    waterlevel_hat = b + width*m
    waterlevel_hat_list.append(waterlevel_hat)
    print(i)
    
'픽셀 vs. 수위 회귀식 Test'
plt.plot(width_test_list, waterlevel_hat_list, '.')
r2_test = round(r2_score(width_test_list, waterlevel_hat_list),2)
# =============================================================================
# plt.plot(x, waterlevel_hat_list, '-', label = 'test')
# =============================================================================
plt.annotate('R2='+str(r2_test), xy=(160,1.2))
plt.xlabel('Pixels')
plt.ylabel('Waterlevel (m)')
plt.legend(['R2 = ' +str(r2_test)])
plt.show()    

'Scatter plot 수위 (Observed vs. Modeled)'    
from sklearn.metrics import r2_score, mean_absolute_error
x = [0,0.8]
r2_scatter = round(r2_score(waterlevel_answer, waterlevel_hat_list),2)
r2_mae = round(mean_absolute_error(waterlevel_answer, waterlevel_hat_list),2)
plt.figure(figsize=(5,5))
plt.plot(waterlevel_answer, waterlevel_hat_list, '.')
plt.xlabel('Observed (m)')
plt.ylabel('Modeled (m)')
plt.plot(x, x, '-', label = 'test')
plt.legend(['R2 = ' +str(r2_scatter)+'  /  '+ 'MAE = ' +str(r2_mae)])






 

# =============================================================================
# 
# '픽셀 수위 최적 회귀식 라인 테스트'
# from numpy.polynomial.polynomial import polyfit
# from sklearn.metrics import r2_score, mean_absolute_error
# for k in range(1):
#     waterlevel_dataset = []
#     for i, image_name in enumerate(masks):
#         image = cv2.imread(mask_directory+image_name, 0)
#         image = Image.fromarray(image)
#         image = image.resize((SIZE, SIZE))
#         width = int(sum(list(np.array(image)[:,128+k])))
#         waterlevel = image_name.split('_')[1]
#         waterlevel = waterlevel.split('.')[0] + '.' + waterlevel.split('.')[1]
#         waterlevel = float(waterlevel)
#         bb = (waterlevel, width)
#         waterlevel_dataset.append(bb)
#     
#     test = np.array(waterlevel_dataset).T
#     x = test[1]
#     y = test[0]
#     b,m = polyfit(x,y,1)
#     y_hat = b+m*x
#     r2 = r2_score(y,y_hat)
#     plt.title(r2)
#     plt.plot(x,y, '.')
#     plt.plot(x, y_hat, '-', label='test')
#     plt.xlabel('Pixels')
#     plt.ylabel('Waterlevel (m)')
#     plt.show()
# =============================================================================
