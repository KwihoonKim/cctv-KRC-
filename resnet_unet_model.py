# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 10:44:47 2021

@author: USESR
"""

import segmentation_models as sm

def resnet18_unet_model():
    BACKBONE = 'resnet34'
    model = sm.Unet(BACKBONE, input_shape = (None, None, 3), encoder_weights='imagenet', classes=1)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'])
    model.summary()
    return model

# =============================================================================
# def resnet34_unet_model():
#     BACKBONE = 'resnet34'
#     model = sm.Unet(BACKBONE, input_shape = (None, None, 3), encoder_weights='imagenet', classes=1)
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'])
#     model.summary()
#     return model
# =============================================================================
from segmentation_models.metrics import FScore
from segmentation_models.losses import DiceLoss

def resnet34_unet_model():
    BACKBONE = 'resnet34'
    model = sm.Unet(BACKBONE, input_shape = (None, None, 3), encoder_weights='imagenet', classes=1)
    metric = FScore()
# =============================================================================
#     loss = DiceLoss()
#     loss = JaccardLoss()
# =============================================================================
    loss = 'binary_crossentropy'
    model.compile(optimizer='adam', loss= loss, metrics=[metric])
    model.summary()
    return model

def resnet50_unet_model():
    BACKBONE = 'resnet50'
    model = sm.Unet(BACKBONE, input_shape = (None, None, 3), encoder_weights='imagenet', classes=1)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'])
    model.summary()
    return model

def resnet101_unet_model():
    BACKBONE = 'resnet101'
    model = sm.Unet(BACKBONE, input_shape = (None, None, 3), encoder_weights='imagenet', classes=1)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'])
    model.summary()
    return model

def resnet152_unet_model():
    BACKBONE = 'resnet152'
    model = sm.Unet(BACKBONE, input_shape = (None, None, 3), encoder_weights='imagenet', classes=1)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'])
    model.summary()
    return model

def vgg16_unet_model():
    BACKBONE = 'vgg16'
    model = sm.Unet(BACKBONE, input_shape = (None, None, 3), encoder_weights='imagenet', classes=1)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'])
    model.summary()
    return model

def vgg19_unet_model():
    BACKBONE = 'vgg19'
    model = sm.Unet(BACKBONE, input_shape = (None, None, 3), encoder_weights='imagenet', classes=1)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'])
    model.summary()
    return model

def se_resnet18_unet_model():
    BACKBONE = 'seresnet18'
    model = sm.Unet(BACKBONE, input_shape = (None, None, 3), encoder_weights='imagenet', classes=1)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'])
    model.summary()
    return model

def se_resnet34_unet_model():
    BACKBONE = 'seresnet34'
    model = sm.Unet(BACKBONE, input_shape = (None, None, 3), encoder_weights='imagenet', classes=1)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'])
    model.summary()
    return model

def se_resnet50_unet_model():
    BACKBONE = 'seresnet50'
    model = sm.Unet(BACKBONE, input_shape = (None, None, 3), encoder_weights='imagenet', classes=1)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'])
    model.summary()
    return model

def se_resnet101_unet_model():
    BACKBONE = 'seresnet101'
    model = sm.Unet(BACKBONE, input_shape = (None, None, 3), encoder_weights='imagenet', classes=1)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'])
    model.summary()
    return model

def se_resnet152_unet_model():
    BACKBONE = 'seresnet152'
    model = sm.Unet(BACKBONE, input_shape = (None, None, 3), encoder_weights='imagenet', classes=1)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'])
    model.summary()
    return model

def resnext50_unet_model():
    BACKBONE = 'resnext50'
    model = sm.Unet(BACKBONE, input_shape = (None, None, 3), encoder_weights='imagenet', classes=1)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'])
    model.summary()
    return model

def resnext101_unet_model():
    BACKBONE = 'resnext101'
    model = sm.Unet(BACKBONE, input_shape = (None, None, 3), encoder_weights='imagenet', classes=1)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'])
    model.summary()
    return model

def senet154():
    BACKBONE = 'senet154'
    model = sm.Unet(BACKBONE, input_shape = (None, None, 3), encoder_weights='imagenet', classes=1)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'])
    model.summary()
    return model

