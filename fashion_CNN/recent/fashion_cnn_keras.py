'''Train a simple deep CNN on the CIFAR10 small images dataset.
It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
# For custom metrics
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os

import time
import copy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf


batch_size = 128
num_classes = 228
epochs = 1
data_augmentation = True
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_fashion_trained_model.h5'

# The data, split between train and test sets:
x_train = np.load('D:/PrivacyPreservingDistributedDL/Filters/x_train.npy')
y_train = np.load('D:/PrivacyPreservingDistributedDL/Filters/y_train.npy')
print('reading train data done')
x_eval = np.load('D:/PrivacyPreservingDistributedDL/Filters/x_vali.npy')
y_eval = np.load('D:/PrivacyPreservingDistributedDL/Filters/y_vali.npy')
print('reading eval data done')
x_test = np.load('D:/PrivacyPreservingDistributedDL/Filters/x_test.npy')
y_test = np.load('D:/PrivacyPreservingDistributedDL/Filters/y_test.npy')
#x_Test = np.load('D:/PrivacyPreservingDistributedDL/Filters/x_TestSub_size_512.npy')
print('reading test done')




print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
#print(x_Test.shape[0], 'Test samples')

"""
for i in range(0,x_train.shape[0]-1):
    image = x_train[i,:,:,:]
    plt.imshow(image)
    plt.show()
    time.sleep(1)
"""


# Convert class vectors to binary class matrices.
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',input_shape=x_train.shape[1:])) #conv1
model.add(Activation('relu'))
print('x_train shape for Conv1:', x_train.shape[1:])

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



# Let's train the model using adam
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=[_ap,_f1])

x_train = x_train.astype('float32')
x_eval = x_eval.astype('float32')
#x_Test = x_Test.astype('float32')
x_train /= 255
x_eval /= 255
#x_Test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=True,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4)
    """
    preds = model.predict(x_Test)
    preds[preds>= 0.5] = 1
    preds[preds<0.5] = 0
    """
    
    #print('Shape of y_test: ',y_eval.shape)
    #print('Shape of preds: ',preds.shape)
    #print ('msd accuracy =',1-np.mean(np.square(y_eval-preds)))
    ############## save Test matrix
    #np.save('D:/PrivacyPreservingDistributedDL/Filters/y_TestSub_size_512.npy',preds)
    #print("eval matrix saved.")
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_eval, y_eval, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
#print(model.summary())
preds = model.predict(x_eval)
print('=='*100)
sess=tf.Session()
with sess.as_default():
    print(_f1(tf.convert_to_tensor(y_eval,dtype=tf.float32),tf.convert_to_tensor(preds,dtype=tf.float32)).eval())