import numpy as np
import keras
import time
import cv2
from skimage import img_as_ubyte
import matplotlib.image as mpimg
import copy

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, image_list, labels, batch_size=64, dim=(64,64), n_channels=3,
                 n_classes=228, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.image_list = image_list
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype= 'uint8')
        self.y = np.empty((self.batch_size,self.n_classes), dtype='uint8')

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.image_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
         
        limit = min(self.batch_size,len(self.image_list)-self.batch_size*(index))
        
        
        indexes = self.indexes[index*self.batch_size:index*self.batch_size+limit]
        # Find list of IDs
        image_list_temp = indexes
        # Generate data
        X, y = self.__data_generation(image_list_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(1,len(self.image_list)+1)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, image_list_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
  
        

        # Generate data
        #start_batch = time.time()
        for i, ID in enumerate(image_list_temp):
            # Store sample
            #print('printing image list temp',len(image_list_temp))
            #print(image_list_temp)
            
            img = mpimg.imread('D:/dataset/inputs/{}.jpg'.format(ID))
            self.X[i,] = img_as_ubyte(cv2.resize(img,self.dim, interpolation = cv2.INTER_AREA))
            # Store class
            self.y[i,] = self.labels[i,]
        #end_batch = time.time()

        #print('Time for resizeing batch')
        #print(end_batch - start_batch)

        return self.X[0:len(image_list_temp),:], self.y[0:len(image_list_temp),:]

