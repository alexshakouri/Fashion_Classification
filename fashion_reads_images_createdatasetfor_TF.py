# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 13:39:41 2018

@author: chama
"""
## import opencv

#opencv2
import cv2

import time


from IPython.core.display import HTML
from IPython.display import Image
#from collections import Counter
import pandas as pd 
import json
import gc

from plotly.offline import init_notebook_mode, iplot
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import plotly.graph_objs as go

from plotly import tools
import seaborn as sns


import tensorflow as tf
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sps
## read the dataset 

path = "D:\Chamain\Academic\Spring 2018\EEC 289Q Computer Engineering\Project/inputs/train.json"

inp = open(path).read()
inp = json.loads(inp)



# how many images 
ntrain_images = len(inp['images'])
train_annotations = inp['annotations']
#train_labels = train_annotations['label_id']

# how many labels 
all_annotations = []
for each in inp['annotations']:
    all_annotations.extend(each['labelId'])
total_labels = len(set(all_annotations))
print(all_annotations[0])
print ("Total Images in the dataset: ", ntrain_images)
print ("Total Labels in the dataset: ", total_labels)


## view images

#iterate over the traing samples and display images and labels
"""
for image in inp['annotations']:
    print('image No:',image['imageId'])
    print('Labels: ',image['labelId'])
    img=mpimg.imread('D:\dataset\inputs\{}.jpg'.format(image['imageId']))
    plt.imshow(img)
    plt.show()
    time.sleep(1)
"""

###### creating panda dataframes
'''
train_imgs_df = pd.DataFrame.from_records(inp["images"])
train_imgs_df["url"] = train_imgs_df["url"]  ## maybe better to get rid of the urls
train_labels_df = pd.DataFrame.from_records(inp["annotations"])
#train_labels_df = train_labels_df["labelId"].apply(lambda x: [int(i) for i in x])
train_df = pd.merge(train_imgs_df,train_labels_df,on="imageId",how="outer")
train_df["imageId"] = train_df["imageId"].astype(np.int)
print(train_df.head())
print(train_df.dtypes)
'''
del inp
gc.collect()

############### creating data matrix
'''
train_image_arr = train_df[["imageId","labelId"]].apply(lambda x: [(x["imageId"],int(i)) for i in x["labelId"]], axis=1).tolist()
train_image_arr = [item for sublist in train_image_arr for item in sublist]
train_image_row = np.array([d[0] for d in train_image_arr]).astype(np.int)
train_image_col = np.array([d[1] for d in train_image_arr]).astype(np.int)
train_image_vals = np.ones(len(train_image_col))
train_image_mat = csr_matrix((train_image_vals, (train_image_row, train_image_col)))
print(train_image_mat.shape)
'''
################## Saving the data matrix
'''
sps.save_npz('./inputs/train_image_mat.npz', train_image_mat)
print('train matrix saved')
'''


train_image_mat = sps.load_npz("D:\Chamain\Academic\Spring 2018\EEC 289Q Computer Engineering\Project/inputs/train_image_mat.npz")
print('train matrix shape, after loading: ',train_image_mat.shape);



labels = train_image_mat.sum(0).astype(np.int)
print(labels)
print(labels.argmax(axis=1))

## Class distribution.
plt.figure(figsize=(15,5))
labels_inds = np.arange(len(labels.tolist()[0]))
sns.barplot(labels_inds,  labels.tolist()[0])
plt.xlabel('label id', fontsize=0.5)
plt.ylabel('Count', fontsize=16)
plt.title("Distribution of labels", fontsize=10)

############# display images correspoinding to each label
"""
for col in range(1,train_image_mat.shape[1]):
    print('Label : ',col)
    image_id = np.array(np.nonzero(train_image_mat[:,col]))
    print(image_id[0,:])
    print('-'*100)
    for i in image_id[0,0:4]:
        print('Label: ',col,'Image: ',i)
        img=mpimg.imread('D:\dataset\inputs\{}.jpg'.format(i))
        plt.imshow(img)
        plt.show()
        time.sleep(1)
    time.sleep(2)
"""
### creating the filename and label vectors
current_train_mat = train_image_mat[1:50001,1:229]
#current_train_mat = train_image_mat[1:50001,1]
print(current_train_mat.shape)
train_filenames = np.asarray(['D:\dataset\inputs\{}.jpg'.format(i) for i in range(1,50001)]).reshape(50000,1)
train_labels = current_train_mat.todense().reshape((50000,228))
#train_labels = current_train_mat.todense().reshape((50000,1))
print('Shape of file names : ',train_filenames.shape)
print('Shape of labels: ',train_labels.shape)


print(train_filenames[0:10])
print(train_labels[0:10])

"""
def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_image(image_string)
  image_resized = tf.image.resize_images(image_decoded, [10, 10,3])
  return image_resized, label

# A vector of filenames.
filenames = tf.constant(['D:\dataset\inputs\1.jpg', 'D:\dataset\inputs\2.jpg'])

# `labels[i]` is the label for the image in `filenames[i].
labels = tf.constant([1,2])

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(_parse_function)
"""

#reading with opencv
def _read_py_function(filename, label):
  image_decoded = cv2.imread(filename.decode(), cv2.IMREAD_COLOR)
  return image_decoded, label

# Use standard TensorFlow operations to resize the image to a fixed shape.
def _resize_function(image_decoded, label):
  image_decoded.set_shape([None, None, None])
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label

filenames = train_filenames
labels = train_labels

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(
    lambda filename, label: tuple(tf.py_func(
        _read_py_function, [filename, label], [tf.uint8, label.dtype])))
dataset = dataset.map(_resize_function)

