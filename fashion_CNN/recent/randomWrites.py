# -*- coding: utf-8 -*-
"""
Created on Tue May 29 17:35:28 2018

@author: chama
"""


from __future__ import print_function
import keras
import cv2
from tqdm import tqdm
from skimage import img_as_ubyte
# For custom metrics

import matplotlib.image as mpimg
import numpy as np
import scipy.sparse as sps
import random










load_size = 50000
batch_size =32
num_classes = 228
num_channels=3
image_dim =64
#####################################################22

 ### creating the filename and label vectors
train_labels = sps.load_npz("D:/PrivacyPreservingDistributedDL/Filters/train_image_mat.npz").todense()
############### loading validation image mat
eval_labels = sps.load_npz("D:/PrivacyPreservingDistributedDL/Filters/eval_image_mat.npz").todense()
    




train_labels = train_labels[0:200001,]
 



print(train_labels)
    
train_set_size = train_labels.shape[0]-1
#eval_set_size = eval_labels.shape[0]-1
label_size = train_labels.shape[1]-1
    
print('train_set_size ',train_set_size )
#print('eval_set_size ',eval_set_size )
print('label_size ',label_size )

index = np.arange(1,train_set_size)
random.shuffle(index)


X_load_batch = np.zeros((load_size,image_dim,image_dim,num_channels),dtype='uint8')
Y_load_batch = np.zeros((load_size,num_classes),dtype='uint8')


y_train =train_labels[1:train_set_size+1,1:229]

 
n_load_batch = int(np.ceil(train_set_size/load_size))
for load_batch in tqdm(range(1,n_load_batch+1)):
    
    if load_batch ==n_load_batch:
        load_sizel=train_set_size - load_size*(load_batch-1)
    else:
        load_sizel = load_size
        #######load the images, resize them
    r_start =(load_batch-1)*load_size
    #r_end = r_start+load_sizel
    #with tf.device("/cpu:0"):
                    
    for i in tqdm(range(0,load_sizel)):
        img = mpimg.imread('D:/dataset/inputs/{}.jpg'.format(index[r_start+i]))
        X_load_batch[i,]=img_as_ubyte(cv2.resize(img,(image_dim,image_dim), interpolation = cv2.INTER_AREA))
        batches = 0
        Y_load_batch[i,]=y_train[index[r_start+i],]
    print('\n saving loadBatch number: ', load_batch)
    np.save('D:/dataset/BATCHER/X_batch50000_64_{}.npy'.format(load_batch),X_load_batch[0:load_sizel,])
    np.save('D:/dataset/BATCHER/Y_batch50000_64_{}.npy'.format(load_batch),Y_load_batch[0:load_sizel,])

"""
y_train =train_labels[0:train_set_size+1,1:229]
index = np.arange(1,train_set_size)
random.shuffle(index)

 
n_load_batch = int(np.ceil(train_set_size/load_size))
for load_batch in tqdm(range(1,n_load_batch+1)):
    
    if load_batch ==n_load_batch:
        load_sizel=train_set_size - load_size*(load_batch-1)
    else:
        load_sizel = load_size
        #######load the images, resize them
    r_start =(load_batch-1)*load_size
    #r_end = r_start+load_sizel
    #with tf.device("/cpu:0"):
                    
    for i in range(0,load_sizel):
        img = mpimg.imread('D:/dataset/inputs/{}.jpg'.format(index[r_start+i]))
        X_load_batch[i,]=img_as_ubyte(cv2.resize(img,(image_dim,image_dim), interpolation = cv2.INTER_AREA))
        batches = 0
        Y_load_batch[i,]=y_train[index[r_start+i],]
    print('\n saving loadBatch number: ', load_batch)
    np.save('D:/dataset/BATCHESR/X_batch64_512r__{}.npy'.format(load_batch),X_load_batch[0:load_sizel,])
    np.save('D:/dataset/BATCHESR/Y_batch64_512r__{}.npy'.format(load_batch),Y_load_batch[0:load_sizel,])
"""