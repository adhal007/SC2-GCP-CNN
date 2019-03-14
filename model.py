import keras  # Keras 2.1.2 and TF-GPU 1.8.0
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv3DTranspose
from keras.callbacks import TensorBoard
import numpy as np
import os
import random
import task
def SC2_model():

      model = Sequential()
      model.add(Conv2D(32, kernel_size=(3,3), padding='same',
      input_shape=(176, 200, 3),
      activation='relu'))
      model.add(MaxPooling2D(pool_size=(2, 2)))
      model.add(Conv2D(64, (5, 5), activation='relu', padding='valid'))
      model.add(MaxPooling2D(2, 2))
      model.add(Conv2D(128, (7, 7), activation='relu', padding='valid'))
      model.add(MaxPooling2D(2, 2))
      model.add(Flatten())
      model.add(Dropout(0.25))
      model.add(Dense(512, activation='relu'))

      model.add(Dense(4, activation='softmax'))

      model.summary()
      learning_rate = 0.0001
      opt = keras.optimizers.adam(lr=learning_rate, decay=1e-6)

      model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
      return model
# tensorboard = TensorBoard(log_dir="logs/stage1")
