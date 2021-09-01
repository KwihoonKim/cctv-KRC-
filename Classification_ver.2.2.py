# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 23:32:42 2021

@author: USESR
"""


import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

# image, mask load and save

number_group = 8
SIZE = 224
image_directory = 'dataset_path/42110A013910/'
images = os.listdir(image_directory)


label_classification = []
import jenkspy
for i, image_name in enumerate(images):
    waterlevel = image_name.split('_')[1]
    level = waterlevel.split('.')[0]+'.'+waterlevel.split('.')[1]
    level = float(level)
    label_classification.append(level)
breaks = jenkspy.jenks_breaks(label_classification, nb_class=number_group)


train_label = []
for i, image_name in enumerate(images):
    label = [0 for i in range(number_group)]
    waterlevel = image_name.split('_')[1]
    level = waterlevel.split('.')[0]+'.'+waterlevel.split('.')[1]
    level = float(level)
    k = 0        
    for j, lev in enumerate(breaks):
        if j == 0:
            continue
        elif level <= float(lev):
            k += 1
    label[number_group-k] = 1
    train_label.append(label)
    
    
image_dataset = []
for i, image_name in enumerate(images):
    if (image_name.split('.')[2] == 'png'):
        image = cv2.imread(image_directory+image_name, 1)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        image_dataset.append(np.array(image))
        
'Model Construction'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Activation, Dense, BatchNormalization
from tensorflow.keras.layers import Flatten, Convolution2D, MaxPooling2D
res_model = tf.keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

model = Sequential()
model.add(res_model)
model.add(Dense(number_group,activation = 'softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

'Train Test Split'
from sklearn.model_selection import train_test_split
image_dataset = np.array(image_dataset)
train_label = np.array(train_label)

X_train, X_test, Y_train, Y_test = train_test_split(image_dataset, train_label, test_size = 0.30, random_state = 0)
X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size = 0.33, random_state = 0)

history = model.fit(X_train, Y_train, batch_size = 16, epochs=100, validation_data=(X_val,Y_val))


# =============================================================================
# model.save('unet_'+model_name+'_epoch200_normx.hdf5')
# =============================================================================
predict = model.predict(X_test)

# Evaluate model
loss, acc = model.evaluate(X_test, Y_test)

print('Accuracy = ', (acc*100.0), '%')

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

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label = 'Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation f1')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

num = len(X_test)
Y_test_hat = []

for i in range(num):
    '평가지표 계산'
    test_img = X_test[i]
    test_label = Y_test[i]
    test_img_input = np.expand_dims(test_img, 0)
    test_label_hat = model.predict(test_img_input)
    Y_test_hat.append(test_label_hat)

Y_test_hat = np.round(Y_test_hat,2)

'그림 출력'
fig = plt.figure()
for i in range(16):
    ax = fig.add_subplot(4,4,i+1)
    plt.subplots_adjust(hspace=0.35)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.set_title(Y_test_hat[i],size=10)
    ax.imshow(X_test[i])    
    print(i)
