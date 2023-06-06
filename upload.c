import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re
import cv2
from datetime import datetime, timedelta
import sys
from numpy import asarray
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

np.random.seed(7)

from google.colab import drive
drive.mount('/content/drive')
  
x = sorted(glob.glob("/content/drive/MyDrive/뿌리 데이터(크기변환)/test/root1_2209*/*.jpg"))
x = np.array([np.array(Image.open(fname).convert("L")) for fname in x])
x = x.astype('float32') /255.

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets, with 80% for training and 20% for testing
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
  
input_shape = (174, 320, 1)
  
num_classes = 7
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same',
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
  
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
  
batch_size = 150
epochs = 9
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
hist = model.fit(x_train, y_train,
batch_size=batch_size, epochs=epochs,
verbose=1, validation_data=(x_test, y_test))
  
import random
predicted_result = model.predict(x_test) 
predicted_labels = np.argmax(predicted_result, axis=1) 
test_labels = np.argmax(y_test, axis=0)
wrong_result = []
for n in range(0, len(test_labels)):
  if predicted_labels[n] != test_labels[n]: 
    wrong_result.append(n)
samples = random.choices(population=wrong_result, k=16) 
count = 0
nrows = ncols = 4
plt.figure(figsize=(12,8))
for n in samples:
  count += 1
  plt.subplot(nrows, ncols, count)
  plt.imshow(x_test[n].reshape(28, 28), cmap='Greys', interpolation='nearest')
  tmp = "Label:" + str(test_labels[n]) + ", Prediction:" + str(predicted_labels[n]) 
  plt.title(tmp)
plt.tight_layout() 
plt.show()
