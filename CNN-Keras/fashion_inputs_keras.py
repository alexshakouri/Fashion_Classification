#opencv2
import cv2
import time
from tqdm import tqdm
from skimage import img_as_ubyte
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
import copy

import tensorflow as tf
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sps
## read the dataset 


train_image_mat = sps.load_npz("D:\Chamain\Academic\Spring 2018\EEC 289Q Computer Engineering\Project/inputs/train_image_mat.npz")
print('train matrix shape, after loading: ',train_image_mat.shape);



labels = train_image_mat.sum(0).astype(np.int)
### creating the filename and label vectors
current_train_mat = train_image_mat[1:50001,1:229]
current_eval_mat = train_image_mat[50001:60001,1:229]
#current_train_mat = train_image_mat[1:50001,1]

#train_filenames = np.asarray(['D:/dataset/inputs/{}.jpg'.format(i) for i in range(1,50001)]).reshape(50000,1)
train_labels = current_train_mat.todense().reshape((50000,228))
#train_labels = current_train_mat.todense().reshape((50000,1))
#print('Shape of file names : ',train_filenames.shape)
print('Shape of labels: ',train_labels.shape)

eval_filenames = np.asarray(['D:/dataset/inputs/{}.jpg'.format(i) for i in range(50001,60001)]).reshape(10000,1)
eval_labels = current_eval_mat.todense().reshape((10000,228))
#train_labels = current_train_mat.todense().reshape((50000,1))
print('Shape of file names : ',eval_filenames.shape)
print('Shape of labels: ',eval_labels.shape)

#######################training input preperation
#read resixe and store
x_train = np.zeros(shape=(50000,32,32,3), dtype='uint8')
y_train = np.zeros((50000,228),dtype='uint8')

x_test = np.zeros(shape=(10000,32,32,3), dtype='uint8')
y_test = np.zeros((10000,228),dtype='uint8')
#print(str(train_filenames[0,:]))
"""
for i in tqdm(range(0,50000-1)):
    img=mpimg.imread('D:/dataset/inputs/{}.jpg'.format(i+1))
    img=img_as_ubyte(cv2.resize(img,(32,32), interpolation = cv2.INTER_AREA))
    x_train[i,:,:,:] = copy.copy(img)
    y_train[i,:]=copy.copy(train_labels[i,:])
    
    #plt.imshow(x_train[i,:,:,:])
    #plt.show()
    #time.sleep(2);
    
#np.save('D:/PrivacyPreservingDistributedDL/Filters/x_train.npy',x_train)
#np.save('D:/PrivacyPreservingDistributedDL/Filters/y_train.npy',y_train)   
    
print('train read done!')
"""
for i in tqdm(range(0,10000-1)):
    img=mpimg.imread('D:/dataset/inputs/{}.jpg'.format(i+50000+1))
    img=img_as_ubyte(cv2.resize(img,(32,32), interpolation = cv2.INTER_AREA))
    x_test[i,:,:,:] = copy.copy(img)
    y_test[i,:]=copy.copy(eval_labels[i,:])
    print(y_test[i,:])
    plt.imshow(x_test[i,:,:,:])
    plt.show()
    time.sleep(2);
#np.save('D:/PrivacyPreservingDistributedDL/Filters/x_test.npy',x_test)
#np.save('D:/PrivacyPreservingDistributedDL/Filters/y_test.npy',y_test)
print('testing read done')