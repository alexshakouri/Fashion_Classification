# -*- coding: utf-8 -*-
"""
Created on Wed May  2 18:19:20 2018
@author: CHAMAIN AND ALEX (mostly ALEX ;D)
"""

import csv
import numpy as np
import pandas as pd
y_Test = np.load('y_Test.npy')
print('y_Test loaded with size',y_Test.shape)
print(y_Test[1,:])

imageId = np.arange(1, y_Test.shape[0]+1)

labelId = np.zeros(y_Test.shape[0], dtype=object)

for i in range(0, y_Test.shape[0]):
    label_indices = np.array(np.where(y_Test[i,:]>0))
    
    #This joins together all the indices that we marked correctly
    labelId[i] = ''.join(str(label_indices))
    #Gets rid of the extra [] that gets put into the string
    labelId[i] = labelId[i].strip('[]')
    #print(labelId[i])

print (labelId)

#saves the two types of data into columns in the file
submission = pd.DataFrame({'image_id' : imageId, 'label_id' : labelId})
submission.to_csv("submission.csv", index=False)

