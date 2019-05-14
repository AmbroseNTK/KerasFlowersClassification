from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from matplotlib import cm
import matplotlib.pylab as plt
from onnxmltools.utils import save_model
from tensorflow.core.framework import graph_pb2
import winmltools
import tf2onnx
import onnxmltools
from keras.preprocessing import image
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.layers import GlobalAveragePooling2D
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Layer
from keras.layers import Lambda
from keras.layers import Input
from keras.models import Sequential
from keras.models import Model
from keras import backend as K
import keras

import numpy as np
import sys
import os
import argparse


def createApplicationModel():
    base_model = keras.applications.InceptionV3(
        False, input_shape=(256, 256, 3))
    x = base_model.output

    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)

    y = Dense(5, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=y)
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def prepareDataset():
    train_datagen = ImageDataGenerator(
        rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    return (train_datagen.flow_from_directory(training_dir), train_datagen.flow_from_directory(test_dir))
