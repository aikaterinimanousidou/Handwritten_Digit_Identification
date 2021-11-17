# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 15:47:29 2020

@author: Aikaterini Manousidou and Johanna Sundberg
"""

import tensorflow as tf
from tensorflow.keras import models, layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""Load datafiles"""
trainData = pd.read_csv('c:/Users/46729/Documents/Python Scripts/Artificiell Intelligens/trainset.csv')
testData = pd.read_csv('c:/Users/46729/Documents/Python Scripts/Artificiell Intelligens/testset.csv')
nrOfPixels = 28
colorValue = 255

"""Preprocessing training and validation data"""
valSize = int(len(trainData) * 0.2)
validationI = np.random.choice(trainData.shape[0], size = valSize, replace=False)
validationIndex = trainData.index.isin(validationI)
validation = trainData.iloc[validationIndex]

training = trainData.iloc[~validationIndex]

training = training.to_numpy()
train_label = training[:,0]
trainImages = training[:,1:785]
trainImages = trainImages/colorValue
trainImages = trainImages.reshape((len(trainImages),28,28,1))

validation = validation.to_numpy()
valid_label = validation[:,0]
validImages = validation[:,1:785]
validImages = validImages/colorValue
validImages = validImages.reshape((len(validImages),28,28,1))

"""Preprocessing testdata"""
testImages = np.array(testData.iloc[:,:])
testImages = testImages/colorValue
testImages = testImages.reshape((len(testImages),28,28,1))


"""Create network model"""
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), padding ='same', activation='relu', input_shape=(nrOfPixels, nrOfPixels, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), padding ='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (4, 4), padding ='same', activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation="softmax"))


"""Freezing the first layers"""
model.layers[0].trainable = True
model.layers[1].trainable = True
model.layers[2].trainable = False
model.layers[3].trainable = False
model.layers[4].trainable = False
model.layers[5].trainable = True
model.layers[6].trainable = True
model.layers[7].trainable = True
  
"""Training the top layers"""
model.summary()
opt_func = 'rmsprop'
loss_func = tf.keras.losses.SparseCategoricalCrossentropy()
model.compile(optimizer=opt_func, loss=loss_func, metrics=['accuracy'])
history1 = model.fit(trainImages, train_label, epochs=6,
                    validation_data=(validImages, valid_label))

"""Training all the layers"""
model.trainable = True
model.summary()
model.compile(optimizer=opt_func, loss=loss_func, metrics=['accuracy'])
history2 = model.fit(trainImages, train_label, epochs=10,
                    validation_data=(validImages, valid_label))


"""Evaluate model with testImages"""
evaluations = np.array(model.predict(testImages))

"""Load predictions into csv file"""
datalabels = np.arange(1, 14001, 1)
psychicPredictions = list()
for i in range(0,len(evaluations)):
    psychicPredictions.append(evaluations[i,:].argmax())
    
dataAsDict = {'ImageId': datalabels, 'Label': psychicPredictions}
datafile = pd.DataFrame(data = dataAsDict)
datafile.to_csv('predictions.csv', index = False)

"""Plots"""
plt.title('Convergence of accuracy')
plt.plot(history1.history['accuracy'], label='accuracy_1')
plt.plot(history1.history['val_accuracy'], label = 'val_accuracy_1')
plt.plot(history2.history['accuracy'], label='accuracy_2')
plt.plot(history2.history['val_accuracy'], label = 'val_accuracy_2')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
