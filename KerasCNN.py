import plaidml.keras
plaidml.keras.install_backend()  #use AMD GPU
'''
import cv2
import os
import random
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import confusion_matrix
import itertools
'''
import numpy as np

import pandas as pd
import keras
from sklearn.model_selection import train_test_split

from keras.layers import Conv2D, Dense, Dropout, MaxPool2D, Flatten
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint


#Establishing the data directory
trainDataDir = '/Users/zhanmingwang/OneDrive/Programming/kaggle-digitRecognizer/data/train.csv'
testDataDir = '/Users/zhanmingwang/OneDrive/Programming/kaggle-digitRecognizer/data/test.csv'

#read data from CSV files
train = pd.read_csv(trainDataDir)
test = pd.read_csv(testDataDir)

#create the training data
y_train = train['label']
x_train = train.drop('label', axis=1)

#free up memory
del train

#Check for null and missing values
x_train.isnull().any().describe()
test.isnull().any().describe() #in this case, there is none

#normalization
x_train = x_train/ 255.0
test = test/ 255.0

#reshaping (reshape image in 3 dimensions)
x_train = x_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)

#OneHot encode the labels --> [0, 0, 0, 0, 1, 0, 0 ...]
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes= 10)

#split train and test
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)



#build CNN
model = Sequential()
model.add(Conv2D(filters= 32, kernel_size=(5,1), padding= 'Same',
                 activation='relu', input_shape= (28,28,1)))
model.add(Conv2D(filters= 32, kernel_size=(1,5), padding= 'Same',
                 activation='relu', input_shape= (28,28,1)))
model.add(MaxPool2D(pool_size=(1,1)))
model.add(Conv2D(filters= 32, kernel_size=(5,1), padding='Same',
                 activation='relu'))
model.add(Conv2D(filters= 32, kernel_size=(1,5), padding='Same',
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same',
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same',
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(filters=128, kernel_size=(2,2), padding='Same',
                 activation='relu'))
model.add(Conv2D(filters=128, kernel_size=(2,2), padding='Same',
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(10, activation='softmax'))

#Define the optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)     #RMSprop decreases gradient decent oscillation

#Compile the model
model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])

#Set a learning rate annealer (basically decreasing the lr dynamically with steps)
learning_rate_reduction = ReduceLROnPlateau(monitor= 'val_acc',   #this will be use in the model.fit/ model.fit_generator
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00000001)            #ReduceLROnPlateau decreases the learning rate when the model stops improving

epochs = 100   #theroetically it should get to 0.9967 accuracy
batch_size = 85


##DATA AUGMENTATION (prevent overfitting)
datagen = ImageDataGenerator(   # used below in model.fit_generator
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=10,  #randomly roate image my 10 degrees
    zoom_range= 0.1,  #randomly zoon some images by 10%
    width_shift_range=0.1, #randomly shift image to 10% of the left and right
    height_shift_range=0.1, #randomly shift image to 10% of the height
    horizontal_flip=False, #Not flipping because of 6 and 9
    vertical_flip=False   #Same reason why
)

datagen.fit(x_train) #horizontal or vertical flip might result in misclassifying symetircal numbers such as 6 and 9


#TRAIN!
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                              epochs=epochs, validation_data=(x_val, y_val),
                              verbose=2, steps_per_epoch=x_train.shape[0] // batch_size,
                              callbacks=[learning_rate_reduction])  #fit_generator runs the training and image augmentation in parallel

#Evaluation
    #Trainig and validation curves


#predict results
results = model.predict(test)

#Select the index with the maximum probability
results = np.argmax(results, axis = 1)

results = pd.Series(results, name='Label')

submission = pd.concat([pd.Series(range(1, 28001), name='ImageId'), results],
                       axis=1)

submission.to_csv('cnn_mnist_datagen.csv', index=False)


