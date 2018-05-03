# -*- coding: utf-8 -*-
"""
Created on Wed May  2 18:19:20 2018

@author: chama
"""

import csv
import numpy as np
y_Test = np.load('D:/PrivacyPreservingDistributedDL/Filters/y_Test.npy')
print('y_Test loaded with size',y_Test.shape)
#print(y_Test[66,:])


with open('submission.csv', 'w') as csvfile:
    fieldnames = ['image_id', 'label_id']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for i in range(0,y_Test.shape[0]-1):
        label_indices = np.asarray(np.where(y_Test[i,:]>0))
        print(label_indices)
        label_str = np.array2string(label_indices,separator=' ')
        writer.writerow({'image_id': '{}'.format(i+1), 'label_id':label_str })
