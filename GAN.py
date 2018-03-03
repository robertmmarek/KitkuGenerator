# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 11:48:52 2018

@author: ROBEMARE
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, InputLayer, Reshape, Conv2D
from keras.layers import Conv2DTranspose, MaxPooling2D, BatchNormalization
from keras.layers import Activation, LeakyReLU
from keras import optimizers

import numpy as np

from scipy.misc import imsave

image_width = 100
image_height = 100
nb_of_channels = 1

input_noise_len = 100

def D():
    model = Sequential()
    model.add(InputLayer(batch_input_shape=(None, image_height, image_width, nb_of_channels)))
    
    model.add(Conv2D(filters=64, kernel_size=(5,5)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(filters=128, kernel_size=(3,3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(filters=256, kernel_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    
    model.add(Flatten())

    model.add(Dense(units=32))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=1, activation="sigmoid"))
    
    model.compile(optimizer=optimizers.SGD(),
                  loss='binary_crossentropy')
    
    return model
    
def G():
    model = Sequential()
    model.add(InputLayer(batch_input_shape=(None, input_noise_len)))
    model.add(Dense(units=image_width*image_height))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Reshape(target_shape=(image_height, image_width, 1)))
    model.add(Dropout(0.3))
    
    model.add(Conv2DTranspose(filters=64, kernel_size=(5,5), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2DTranspose(filters=64, kernel_size=(5,5), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2DTranspose(filters=64, kernel_size=(5,5), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2DTranspose(filters=1, kernel_size=(5,5), padding="same"))
    model.add(Activation('sigmoid'))

    
    model.compile(optimizer=optimizers.SGD(),
                  loss='mean_squared_error')
    
    return model

def Combined(D, G):
    model = Sequential()
    model.add(G)
    model.add(D)
    model.compile(optimizer=optimizers.SGD(),
                  loss='binary_crossentropy')
    
    return model
    
def DataGenerator():
    datagen = ImageDataGenerator()
    return datagen

def GenerateNoise(batch):
    return np.random.randn(batch, input_noise_len)

def train_D_t(train_data, D):
    labels = np.ones(shape=[train_data.shape[0]])
    training_loss = D.train_on_batch(train_data, labels)
    return training_loss, D

def train_D_f(train_noise, D, G):
    labels = np.zeros(shape=[train_noise.shape[0]])
    train_data = G.predict(train_noise)
    training_loss = D.train_on_batch(train_data, labels)
    return training_loss, D

def train_G(train_noise, D, Combined):
    temp_d = D.get_weights()
    labels = np.ones(shape=[train_noise.shape[0]])
    training_loss = Combined.train_on_batch(train_noise, labels)
    D.set_weights(temp_d)
    return training_loss, Combined

#training params
batch_size = 5
epoch_size = 200
n_epochs = 1000
save_each = 1

train_path = "./train/"
out_path = "./out/"

G = G()
D = D()
C = Combined(D, G)
data_gen = DataGenerator()
train_data_gen = data_gen.flow_from_directory(train_path, 
                                              batch_size=batch_size,
                                              target_size=(image_height, image_width),
                                              color_mode='grayscale')
record_file = open('log', 'w')
print("epoch mean_dlosst mean_dlossf mean_gloss", file=record_file)
skip_d_training = False
dlosst = []
dlossf = []
gloss = []
for i in range(n_epochs):
    
    if skip_d_training == False:
        dlossf = []
        dlosst = []
    
    gloss = []
    for j in range(epoch_size):
        train_noise = GenerateNoise(batch_size)
        train_noise_2 = GenerateNoise(batch_size)
        train_data = train_data_gen.next()[0]

        if not skip_d_training:
            d_training_loss_f, _ = train_D_f(train_noise_2, D, G)
            dlossf.append(d_training_loss_f)
            d_training_loss_t, _ = train_D_t(train_data, D)
            dlosst.append(d_training_loss_t)
            
        g_training_loss, _ = train_G(train_noise, D, C)
        gloss.append(g_training_loss)
        
        
    skip_d_training = False
    mean_dlosst = np.mean(dlosst)
    mean_dlossf = np.mean(dlossf)
    mean_gloss = np.mean(gloss)
    if(mean_gloss/mean_dlossf > 4.):
        skip_d_training = True
    
    print("%d, mean_dlosst: %f, mean_dlossf: %f, mean_gloss: %f" % (i, mean_dlosst, mean_dlossf, mean_gloss))
    print("%i %f %f %f" % (i, mean_dlosst, mean_dlossf, mean_gloss), file=record_file)
    if i % save_each == 0:
        pred = G.predict(train_noise)
        imsave(out_path+"_"+str(i)+".png", pred[0, :, :, 0])
        G.save_weights('G')
        D.save_weights('D')
   
record_file.close()