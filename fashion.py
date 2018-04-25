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
from wordcloud import WordCloud
from plotly import tools
import seaborn as sns
from PIL import Image

import tensorflow as tf
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sps
## read the dataset 

path = './inputs/train.json'

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


train_image_mat = sps.load_npz('./inputs/train_image_mat.npz')
print('train matrix shape, after loading: ',train_image_mat.shape);



labels = train_image_mat.sum(0).astype(np.int)
print(labels)

## Class distribution.
plt.figure(figsize=(15,5))
labels_inds = np.arange(len(labels.tolist()[0]))
sns.barplot(labels_inds,  labels.tolist()[0])
plt.xlabel('label id', fontsize=0.5)
plt.ylabel('Count', fontsize=16)
plt.title("Distribution of labels", fontsize=10)

############# display images correspoinding to each label
for col in range(train_image_mat.shape[1]):
    print('Label : ',col+1)
    image_id = np.nonzero(train_image_mat[:,col+1]).toarray()
    for i in image_id:
        print('Label: ',col+1,'Image: ',i)
        img=mpimg.imread('D:\dataset\inputs\{}.jpg'.format(i))
        plt.imshow(img)
        plt.show()
        time.sleep(1)
    time.sleep(2)
     


