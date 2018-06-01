# -*- coding: utf-8 -*-
"""
Created on Wed May 30 14:54:38 2018

@author: chama
"""

from __future__ import print_function
import keras

from tqdm import tqdm

# For custom metrics
import keras.backend as K
import gc
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os

import numpy as np
import scipy.sparse as sps

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import random
import writeSubmission
    
load_size = 50000
batch_size = 32
num_classes = 228
num_channels=3;
epochs =10
data_augmentation = True
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_fashion_trained_model_150.h5'
image_dim = 64
Runrange=load_size*16

#####################Load eval data
x_eval = np.load('D:/PrivacyPreservingDistributedDL/Filters/x_eval_all_64.npy')
y_eval = np.load('D:/PrivacyPreservingDistributedDL/Filters/y_eval_all_64.npy')
print('reading eval done')

#load test data
x_Test = np.load('D:/PrivacyPreservingDistributedDL/Filters/x_TestSub_64.npy')
print('reading test done')

############### load train labels
train_labels = sps.load_npz("D:/PrivacyPreservingDistributedDL/Filters/train_image_mat.npz").todense()
train_labels = train_labels[0:Runrange+1]
train_set_size = train_labels.shape[0]-1
y_train =train_labels[1:train_set_size+1,1:num_classes+1]

x_eval = x_eval.astype('float32')
x_eval /= 255
x_Test = x_Test.astype('float32')
x_Test /= 255
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

#################CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',input_shape=(image_dim,image_dim,3))) #conv1
model.add(Activation('relu'))
#print('x_train shape for Conv1:', x_train.shape[1:])
    
model.add(Conv2D(32, (3, 3)))                                              #conv2
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))                             #conv3                                      
model.add(Activation('relu'))
    
model.add(Conv2D(64, (3, 3)))                                            #conv4
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
    
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('sigmoid'))
    
    
# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)


def _sd(y_true, y_pred):
        return 1-K.mean(K.square(K.abs(y_true-K.round(K.abs(y_pred)))))*5
    
def _ap(y_true,y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    
    tp = K.sum(y_pos * y_pred_pos)
    #tn = K.sum(y_neg * y_pred_neg,axis=1)
    
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
        
    precision = tp/(tp+fp);
    recall = tp/(tp+fn);
        
    return (precision+recall)/2
        
def _f1(y_true,y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    
    tp = K.sum(y_pos * y_pred_pos)
    #tn = K.sum(y_neg * y_pred_neg,axis=1)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
        
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
        
    return 2*(precision*recall)/(precision+recall)
    
def _sumLabels(y_true, y_pred):
    return K.sum(y_true,axis=1)
def abs_KL_div(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), None)
    y_pred = K.clip(y_pred, K.epsilon(), None)
    return K.sum( K.abs( (y_true- y_pred) * (K.log(y_true / y_pred))), axis=-1)

# Optimizer
model.compile(loss='binary_crossentropy',optimizer=opt,metrics=[_ap,_f1] )
tbCallBack = keras.callbacks.TensorBoard(log_dir='D:/PrivacyPreservingDistributedDL/Filters/Graph1', histogram_freq=0, write_graph=True, write_images=True)

print('Using real-time data augmentation.')

for e in tqdm(range(epochs)):
    print('\nEpoch', e)
    n_load_batch = int(np.ceil(train_set_size/load_size))
    index = np.arange(1,n_load_batch+1)
    random.shuffle(index)
            
    for load_batch in tqdm(range(1,n_load_batch+1)):
        if index[load_batch-1] ==n_load_batch:
            load_sizel=train_set_size - load_size*(index[load_batch-1]-1)
        else:
            load_sizel = load_size
                
        r_start =(index[load_batch-1]-1)*load_size
                
        print('\nreading batch {}'.format(index[load_batch-1]))     
        X_load_batch = np.load('D:/dataset/BATCHES/X_batch50000_64_{}.npy'.format(index[load_batch-1]))
        Y_load_batch =y_train[r_start:r_start+load_sizel,]
        X_load_batch = X_load_batch.astype('float32')
        X_load_batch /= 255
        model.fit(X_load_batch, Y_load_batch, batch_size=batch_size,epochs=1,validation_data=(x_eval,y_eval),verbose=1,callbacks=[tbCallBack])
        preds = model.predict(x_Test)
        preds[preds>= 0.5] = 1
        preds[preds<0.5] = 0
        print('\n Epoch: ', e)  
        writeSubmission.writeSub(preds)
        #np.save('D:/PrivacyPreservingDistributedDL/Filters/y_TestSub_size50000_64_all{}.npy'.format(e*100000+load_batch),preds)
        print("Test matrix saved.")
        del X_load_batch
        del Y_load_batch
        del preds
        gc.collect()
if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)
    
    # Score trained model.
    #scores = model.evaluate(x_eval, y_eval, verbose=1)
    #print('Test loss:', scores[0])
    #print('Test accuracy:', scores[1])
print(model.summary())    
        
       



            