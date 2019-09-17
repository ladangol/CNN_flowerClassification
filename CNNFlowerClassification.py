
import numpy as np
import matplotlib.pyplot as plt
import os, glob
import PIL
from PIL import Image
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint


def load_imagedata(path):
    X=[]
    y=[]
    #note that the photos do not come in order with glob.glob (that means your target value for the first file is not necessirily class 1)
    #however the photos from 1- 80 belong to one class.
    for f_name in glob.glob(path+"*.jpg"):
        class_index = (int(f_name.split('image_')[1][0:4])-1)/80
        #class_index_one_hot = keras.utils.to_categorical(class_index, 17)[0]
        class_index = int(class_index)
        image = Image.open(f_name)
        image = image.resize((128,128))
        image = np.asarray(image).astype('float32')/255
        X.append(image)
        y.append(class_index)
    X,y = np.array(X), np.array(y)
    y = keras.utils.to_categorical(y, len(np.unique(y)))       #one hot encoding
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state = 42)
    return X_train, X_test,y_train, y_test


path = "flowers17_dataset/"

X_train, X_test, y_train, y_test = load_imagedata(path)

print('X_train shape:', X_train.shape)
print('X_test shape', X_test.shape)
print('y_train shape', y_train.shape)


def create_CNNmodel(bath_size, epochs, num_classes):

    model = Sequential()
    model.add(Conv2D(32, (3,3), padding = 'same', input_shape = X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (5,5), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (5,5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    opt = keras.optimizers.Adam(lr = 0.0001)
    model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

    model.summary()
    return model

batch_size = 64
epochs = 100
num_classes = 17
model = create_CNNmodel(bath_size = batch_size, epochs= epochs, num_classes=num_classes)
#saving the best model

#datagen = ImageDataGenerator(rescale=1./255,
#        shear_range=0.2,
#        zoom_range=0.2,
#        horizontal_flip=True)

#train_set = datagen.fit(X_train)
#test_datagen = ImageDataGenerator(rescale=1./255)
#test_Set = test_datagen.fit(X_test)


filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
# Fit the model
model.fit(X_train, y_train,
          validation_data = (X_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=callbacks_list,
          verbose=0, shuffle = True)
