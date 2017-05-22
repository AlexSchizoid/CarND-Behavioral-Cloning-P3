import os
import argparse
import json
import cv2
import numpy as np
import csv
import pandas as pd
import math
from copy import deepcopy
from sklearn.utils import shuffle
from keras.models import Sequential, model_from_json
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, MaxPooling2D, Cropping2D
from keras.regularizers import l2, activity_l2
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation, Reshape
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

def read_image(index, X_center, X_left, X_right, Y_train, camera = 1):
    #read left/center/right image
    steering = Y_train[index]
    correction = 0.27 #apply correction to left/rigth image
    if camera == 0:
        image = plt.imread('./data/IMG/'+X_left[index].split('/')[-1])
        steering += correction
    elif camera == 1:
        image = plt.imread('./data/IMG/'+X_center[index].split('/')[-1])
    elif camera == 2:
        image = plt.imread('./data/IMG/'+X_right[index].split('/')[-1])
        steering += -correction

    return (image, steering)

def crop_and_resize(image):
    #crop top and bottom pixels and resize to 64x64
    cropped = cv2.resize(image[60:140,:], (64,64))
    return cropped
    
    
def augment_random_brightness(image):
    new_image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    factor = .24 + np.random.uniform()
    new_image[:,:,2] = new_image[:,:,2] * factor
    new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)
    return new_image

def random_shear(image, steering, shear_range):
    #Source is https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713
    
    (rows, cols, ch) = image.shape

    dx = np.random.randint(-shear_range,shear_range+1)
    random_point = [cols/2+dx,rows/2]

    #source points for a trapeze plan
    #destionation points have an offset. warp image
    pts1 = np.float32([[0, rows],[cols,rows],[cols/2, rows/2]])
    pts2 = np.float32([[0, rows],[cols,rows], random_point])

    dsteering = dx /(rows / 2) * 360 / (2*np.pi*25.0) / 6.0    

    M = cv2.getAffineTransform(pts1,pts2)
    image = cv2.warpAffine(image, M, (cols,rows),borderMode=1)
    
    steering += dsteering

    return (image, steering)

def random_flip_image(image,steering):
    #flip the image horizontally and adapt steering angle
    chance = np.random.randint(0,2)
    if chance == 0:
        (image,steering) = cv2.flip(image, 1), -steering
        
    return image,steering
        
def process_image(X_center, X_left, X_right, Y_train):
    #Get a random example
    index = np.random.randint(0,len(Y_train))
    #chose at random one of the 3 cameras
    camera = np.random.randint(0,3)
    image, steering = read_image(index, X_center, X_left, X_right, Y_train, camera)
    
    #first stage apply shearing
    image,steering = random_shear(image,steering,shear_range=90)
    
    #second stage crop top and bottom pixels and resize to 64x64
    image = crop_and_resize(image)
    
    #third stage - flip image horizontally
    image,steering = random_flip_image(image,steering)
    steering = steering * 1.2

    #fourth stage - apply brightness factor to the image
    image = augment_random_brightness(image)
    
    return (image, steering)

def training_generator(X_center, X_left, X_right, Y_train, batch_size = 128):
    
    batch_images = np.zeros((batch_size, 64, 64, 3))
    batch_steering = np.zeros(batch_size)
    #run forever
    while 1:
        for index in range(batch_size):
            #process image
            image, steering = process_image(X_center, X_left, X_right, Y_train)
            batch_images[index] = image
            batch_steering[index] = steering
        yield batch_images, batch_steering
        
def process_validation_samples(X_valid, Y_valid):
    X = np.zeros((len(X_valid),64,64,3))
    Y = np.zeros(len(X_valid))
    
    for index in range(len(X_valid)):
        #get a center camera image
        image, steering = read_image(index, X_valid, X_valid, X_valid, Y_valid)
        X[index] = crop_and_resize(image)
        Y[index] = steering 
    return X,Y

def comma_ai_model():
    model = Sequential()

    model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(64,64,3)))
    model.add(Convolution2D(32, 8,8 ,border_mode='same', subsample=(4,4)))
    model.add(ELU())

    model.add(Convolution2D(64, 8,8 ,border_mode='same',subsample=(4,4)))
    model.add(ELU())

    model.add(Convolution2D(128, 4,4,border_mode='same',subsample=(2,2)))
    model.add(ELU())

    model.add(Convolution2D(128, 2,2,border_mode='same',subsample=(1,1)))
    model.add(ELU())

    model.add(Flatten())

    model.add(Dropout(0.2))
    model.add(Dense(128))
    model.add(ELU())

    model.add(Dropout(0.5))
    model.add(Dense(128))

    model.add(Dense(1))
    
    model.compile(optimizer = Adam(lr=1e-4), loss='mse')

    return model

#use pandas library to read data
data_path = './data'
data_csv = '/driving_log.csv'
training_data = pd.read_csv(data_path + data_csv, names=None)

#prepare sample arrays
X_center = training_data['center']
X_left  = training_data['left']
X_right = training_data['right']
Y_train = training_data['steering']
Y_train = Y_train.astype(np.float32)

X_center = X_center.apply(lambda x: data_path + '/' + x)

X_valid = X_center
Y_valid = Y_train

batch_size=256
train_generator = training_generator(X_center, X_left, X_right, Y_train, batch_size)
#get validation samples
X_valid, Y_valid = process_validation_samples(X_valid, Y_valid)

model = comma_ai_model()

history_object = model.fit_generator(train_generator, samples_per_epoch=batch_size * 100, nb_epoch=5, validation_data=(X_valid, Y_valid), verbose=1)
model.save('./models/model.h5')
