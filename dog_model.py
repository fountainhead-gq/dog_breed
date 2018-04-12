# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from tqdm import tqdm
import zipfile, shutil
import cv2,h5py
import os, sys, glob
import itertools
from sklearn.model_selection import train_test_split

from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Flatten, Input, Activation, Lambda
from keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D
from keras.optimizers import RMSprop, Adam, Adadelta, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, Callback, CSVLogger, History, ModelCheckpoint, EarlyStopping

from keras.applications.inception_v3 import InceptionV3, preprocess_input as inception_v3_preprocess_input
from keras.applications.xception import Xception, preprocess_input as xception_preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input as inception_resnet2_preprocess_input



def get_features(ft_model, preprocess_input, data, img_height=None, batch_size=64):
    if not img_height:
        img_height = 299
        
    inputs = Input(shape=(img_height, img_height, 3))
    input_tensor = Lambda(preprocess_input)(inputs)
    base_model = ft_model(input_tensor=input_tensor, weights='imagenet', include_top=False)
    
    x = GlobalAveragePooling2D()(base_model.output)
    model = Model(inputs=inputs, outputs=x)
    data_feature = model.predict(data, batch_size=batch_size)

    return data_feature

def inception_resnetv2(img_width, n_class, dropout=0.5):
    input_tensor = Input(shape=(img_width, img_width, 3))
    input_tensor = Lambda(inception_resnet2_preprocess_input)(input_tensor)  
    base_model = InceptionResNetV2(input_tensor=input_tensor, weights='imagenet', include_top=False)
    
    #for layers in base_model.layers:
        #layers.trainable = False
    
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(dropout)(x)
    x = Dense(n_class, activation='softmax')(x)
    # x = Dense(n_class, activation='softmax', kernel_initializer='he_normal')(x)
    
    model_inceptionResNetV2 = Model(inputs=base_model.input, outputs=x)
    model_inceptionResNetV2.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
    # model_inceptionResNetV2.summary()
    return model_inceptionResNetV2


def visual_image(X, y, img_num, dog_breed):
    plt.figure(figsize=(15, 7))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        rand = random.randint(0, img_num-1)
        img = X[rand][:,:,::-1]
        plt.title(y[rand])
        plt.imshow(img)
        plt.title(dog_breed[y[rand].argmax()])


def show_loss(his_Model):
    fig, ax = plt.subplots(2,1)
    history = his_Model.history
    ax[0].plot(history['loss'], color='b', label="loss")
    ax[0].plot(history['val_loss'], color='r', label="val_loss",axes =ax[0])
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history['acc'], color='g', label="acc")
    ax[1].plot(history['val_acc'], color='c',label="val_acc")
    legend = ax[1].legend(loc='best', shadow=True)       