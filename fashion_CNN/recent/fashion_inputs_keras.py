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

train_image_range = 100000
validation_set_size = 20000
def _create_Test_image_mat(path):
    #path = "D:\Chamain\Academic\Spring 2018\EEC 289Q Computer Engineering\Project/inputs/train.json"
    
    inp = open(path).read()
    inp = json.loads(inp)
    
    
    
    # how many images 
    nTest_images = len(inp['images'])
    
    
    
    print ("Total Images in the Test dataset: ", nTest_images)
    
    
    ## view images
    
    #iterate over the traing samples and display images and labels
    """
    for image in inp['images']:
        print('image No:',image['imageId'])
        #print('Labels: ',image['labelId'])
        #img=mpimg.imread('D:\dataset\inputs\{}.jpg'.format(image['imageId']))
        #plt.imshow(img)
        #plt.show()
        #time.sleep(1)
    """
    
    ###### creating panda dataframes
    
    Test_imgs_df = pd.DataFrame.from_records(inp["images"])
    Test_imgs_df["url"] = Test_imgs_df["url"]  ## maybe better to get rid of the urls
    #train_labels_df = pd.DataFrame.from_records(inp["annotations"])
    #train_labels_df = train_labels_df["labelId"].apply(lambda x: [int(i) for i in x])
    #train_df = pd.merge(Test_imgs_df,train_labels_df,on="imageId",how="outer")
    Test_df = Test_imgs_df
    Test_df["imageId"] = Test_df["imageId"].astype(np.int)
    print(Test_df.head())
    print(Test_df.dtypes)
    
    del inp
    gc.collect()
    
    ############### creating data matrix
    """
    train_image_arr = train_df[["imageId","labelId"]].apply(lambda x: [(x["imageId"],int(i)) for i in x["labelId"]], axis=1).tolist()
    train_image_arr = [item for sublist in train_image_arr for item in sublist]
    train_image_row = np.array([d[0] for d in train_image_arr]).astype(np.int)
    train_image_col = np.array([d[1] for d in train_image_arr]).astype(np.int)
    train_image_vals = np.ones(len(train_image_col))
    train_image_mat = csr_matrix((train_image_vals, (train_image_row, train_image_col)))
    print(train_image_mat.shape)
    
    ################## Saving the data matrix
    
    sps.save_npz('./inputs/train_image_mat.npz', train_image_mat)
    print('train matrix saved')
    """

#_create_Test_image_mat('D:/dataset/test.json')
train_image_mat = sps.load_npz("D:\Chamain\Academic\Spring 2018\EEC 289Q Computer Engineering\Project/inputs/train_image_mat.npz")
print('train matrix shape, after loading: ',train_image_mat.shape);



labels = train_image_mat.sum(0).astype(np.int)
### creating the filename and label vectors
current_train_mat = train_image_mat[1:train_image_range+1,1:229]
current_eval_mat = train_image_mat[train_image_range+1:(train_image_range + validation_set_size + 1),1:229]
#current_test_mat = train_image_mat[80001:90001,1:229]
#current_train_mat = train_image_mat[1:50001,1]

#train_filenames = np.asarray(['D:/dataset/inputs/{}.jpg'.format(i) for i in range(1,50001)]).reshape(50000,1)
train_labels = current_train_mat.todense().reshape((train_image_range,228))
#train_labels = current_train_mat.todense().reshape((50000,1))
#print('Shape of file names : ',train_filenames.shape)
print('Shape of labels: ',train_labels.shape)

#eval_filenames = np.asarray(['D:/dataset/inputs/{}.jpg'.format(i) for i in range(50001,60001)]).reshape(10000,1)
eval_labels = current_eval_mat.todense().reshape((validation_set_size,228))
#train_labels = current_train_mat.todense().reshape((50000,1))
#print('Shape of file names : ',eval_filenames.shape)
print('Shape of labels: ',eval_labels.shape)

#test_labels = current_test_mat.todense().reshape((10000,228))
#train_labels = current_train_mat.todense().reshape((50000,1))
#print('Shape of file names : ',eval_filenames.shape)
#print('Shape of labels: ',test_labels.shape)


#######################training input preperation
#read resixe and store
x_train = np.zeros(shape=(train_image_range,32,32,3), dtype='uint8')
y_train = np.zeros((train_image_range,228),dtype='uint8')

x_eval = np.zeros(shape=(validation_set_size,32,32,3), dtype='uint8')
y_eval = np.zeros((validation_set_size,228),dtype='uint8')

#x_test = np.zeros(shape=(10000,32,32,3), dtype='uint8')
#y_test = np.zeros((10000,228),dtype='uint8')
#print(str(train_filenames[0,:]))

"""
for i in tqdm(range(0,train_image_range-1)):
    img=mpimg.imread('D:/dataset/inputs/{}.jpg'.format(i+1))
    img=img_as_ubyte(cv2.resize(img,(32,32), interpolation = cv2.INTER_AREA))
    x_train[i,:,:,:] = copy.copy(img)
    y_train[i,:]=copy.copy(train_labels[i,:])
    
    #plt.imshow(x_train[i,:,:,:])
    #plt.show()
    #time.sleep(2);
    
np.save('D:/PrivacyPreservingDistributedDL/Filters/x_train_2.npy',x_train)
np.save('D:/PrivacyPreservingDistributedDL/Filters/y_train_2.npy',y_train)   
"""    
print('train read done!')

for i in tqdm(range(0,validation_set_size-1)):
    img=mpimg.imread('D:/dataset/inputs/{}.jpg'.format(i+train_image_range+1))
    img=img_as_ubyte(cv2.resize(img,(32,32), interpolation = cv2.INTER_AREA))
    x_eval[i,:,:,:] = copy.copy(img)
    y_eval[i,:]=copy.copy(eval_labels[i,:])
    #print(y_test[i,:])
    #plt.imshow(x_test[i,:,:,:])
    #plt.show()
    #time.sleep(2);
np.save('D:/PrivacyPreservingDistributedDL/Filters/x_eval_2.npy',x_eval)
np.save('D:/PrivacyPreservingDistributedDL/Filters/y_eval_2.npy',y_eval)
print('eval read done')
"""
for i in tqdm(range(0,10000-1)):
    img=mpimg.imread('D:/dataset/inputs/{}.jpg'.format(i+90000+1))
    img=img_as_ubyte(cv2.resize(img,(32,32), interpolation = cv2.INTER_AREA))
    x_test[i,:,:,:] = copy.copy(img)
    y_test[i,:]=copy.copy(test_labels[i,:])
    #print(y_test[i,:])
    #plt.imshow(x_test[i,:,:,:])
    #plt.show()
    #time.sleep(2);
np.save('D:/PrivacyPreservingDistributedDL/Filters/x_test.npy',x_test)
np.save('D:/PrivacyPreservingDistributedDL/Filters/y_test.npy',y_test)
print('test read done')
"""
"""
##################comaring first 10,000 training and eval labels
for i in range(0,10000-1):
    print("Difference of sample {} :".format(i+1),np.sum(np.square(y_train[i,:]-y_test[i,:])))
"""
"""
#################### Test preperation
x_Test = np.zeros(shape=(39706,32,32,3), dtype='uint8')
for i in tqdm(range(0,39706-1)):
    img=mpimg.imread('D:/dataset/test/{}.jpg'.format(i+1))
    img=img_as_ubyte(cv2.resize(img,(32,32), interpolation = cv2.INTER_AREA))
    x_Test[i,:,:,:] = copy.copy(img)
    #print(y_test[i,:])
    #plt.imshow(x_test[i,:,:,:])
    #plt.show()
    #time.sleep(2);
np.save('D:/PrivacyPreservingDistributedDL/Filters/x_TestSub.npy',x_Test)
"""
