# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 2021
Created by Kwihoon Kim

<CCTV research project for KRC>
ver.1
"""


import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

# image, mask load and save
image_directory = 'dataset_path/Channel_data/'

SIZE = 224
image_dataset = []
train_label = []

images = os.listdir(image_directory)

for i, image_name in enumerate(images):
    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread(image_directory+image_name, 1)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        image_dataset.append(np.array(image))
        
for i, image_name in enumerate(images):
    label = [0 for i in range(6)]
    waterlevel = image_name.split('_')[0]
    level = waterlevel[0]+waterlevel[1]
    
    if level == '02':
        label[0] = 1
    elif level == '07':
        label[1] = 1
    elif level == '23':
        label[2] = 1
    elif level == '24':
        label[3] = 1
    elif level == '25':
        label[4] = 1
    elif level == '29':
        label[5] = 1 
    train_label.append(label)


'Model Construction'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Activation, Dense, BatchNormalization
from tensorflow.keras.layers import Flatten, Convolution2D, MaxPooling2D
res_model = tf.keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=6)

model = Sequential()
model.add(res_model)
model.add(Dense(6,activation = 'softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

'Train Test Split'
from sklearn.model_selection import train_test_split
image_dataset = np.array(image_dataset)
train_label = np.array(train_label)

X_train, X_test, Y_train, Y_test = train_test_split(image_dataset, train_label, test_size = 0.20, random_state = 0)
X_train, X_val, Y_train, Y_val = train_test_split(image_dataset, train_label, test_size = 0.20, random_state = 0)

history = model.fit(X_train, Y_train, batch_size = 16, epochs=5, validation_data=(X_val,Y_val))


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
plt.ylabel('f1')
plt.legend()
plt.show()



