
#import libraries

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import utils
import os
# %matplotlib inline

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Input, Dropout,Flatten, Conv2D
from keras.layers import BatchNormalization, Activation, MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils import plot_model

from IPython.display import SVG, Image
from livelossplot import PlotLossesKeras
import scipy
import tensorflow as tf
#print("Tensorflow version:", tf.__version__)

#generate training and validation batches
img_size = 48
batch_size = 64

datagen_train = ImageDataGenerator(horizontal_flip=True)
train_generator = datagen_train.flow_from_directory("train/", 
target_size=(img_size,img_size),
color_mode='grayscale',
batch_size=batch_size,
class_mode='categorical',
shuffle=True )

datagen_validation = ImageDataGenerator(horizontal_flip=True)
validation_generator = datagen_train.flow_from_directory("validation/", 
target_size=(img_size,img_size),
color_mode='grayscale',
batch_size=batch_size,
class_mode='categorical',
shuffle=True )


#create cnn model

model = Sequential()

#1 - cnn
model.add(Conv2D(64,(3,3),padding='same',input_shape=(48,48,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#2 - cnn
model.add(Conv2D(128,(5,5),padding='same',))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#3 - cnn
model.add(Conv2D(512,(3,3),padding='same',input_shape=(48,48,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#4 - cnn
model.add(Conv2D(512,(3,3),padding='same',input_shape=(48,48,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(5,activation='softmax'))

opt = Adam(lr=0.0005)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

#train model
epochs = 15
steps_per_epoch = train_generator.n//train_generator.batch_size
validation_steps = validation_generator.n//validation_generator.batch_size

checkpoint = ModelCheckpoint("model_weights.h5" , monitor = 'val_accuracy' ,
save_weights_only=True , mode = 'max' , verbose=1)

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss' , factor=0.1 , patience=2 , min_lr=0.00001 , model = 'auto')
callbacks = [PlotLossesKeras() , checkpoint,reduce_lr]

history = model.fit(
    x = train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=callbacks
)

#represent model as json
model_json = model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)
    