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


###################################################################
# Variables                                                       #
# When launching project or scripts from Visual Studio,           #
# input_dir and output_dir are passed as arguments.               #
# Users could set them from the project setting page.             #
###################################################################

input_dir = None
output_dir = None
log_dir = None


#################################################################################
# Keras configs.                                                                #
# Please refer to https://keras.io/backend .                                    #
#################################################################################


# K.set_floatx('float32')
# String: 'float16', 'float32', or 'float64'.

# K.set_epsilon(1e-05)
# float. Sets the value of the fuzz factor used in numeric expressions.

# K.set_image_data_format('channels_first')
# data_format: string. 'channels_first' or 'channels_last'.


#################################################################################
# Keras imports.                                                                #
#################################################################################


training_dir = "../dataset/flowersv2/training"
test_dir = "../dataset/flowersv2/test"


def loss_plot(fit_history):
    plt.figure(figsize=(18, 4))

    plt.plot(fit_history.history['loss'], label='train')
    plt.plot(fit_history.history['val_loss'], label='test')

    plt.legend()
    plt.title('Loss Function')
    plt.show()


def acc_plot(fit_history):
    plt.figure(figsize=(18, 4))

    plt.plot(fit_history.history['acc'], label='train')
    plt.plot(fit_history.history['val_acc'], label='test')

    plt.legend()
    plt.title('Accuracy')
    plt.show()


def main():

    model = createModel()
    train_set, test_set = prepareDataset()
    training(model, train_set, test_set)
    #model.load_weights("saved_model.h5")
    #result = evaluate(
    #    "D:\\Repos\\KerasFlowersClassification\\dataset\\flowers\\test\\rose\\16525204061_9b47be3726_m.jpg", model)
    #print(result)

# Create Convolution Neural Network


def createModel():

    model = Sequential()

    model.add(Conv2D(32, (5, 5), padding='same', input_shape=(256, 256, 3)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(96, (5, 5)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(GlobalAveragePooling2D())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(5, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='nadam', metrics=['accuracy'])
    return model

def createApplicationModel():
    base_model = keras.applications.InceptionV3(False,input_shape=(256,256,3))
    x = base_model.output

    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)

    y = Dense(5, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=y)
    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
 
    return model

def prepareDataset():
    train_datagen = ImageDataGenerator(
        rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    return (train_datagen.flow_from_directory(training_dir), train_datagen.flow_from_directory(test_dir))


def training(model, train_set, test_set):
    history = model.fit_generator(
        train_set, steps_per_epoch=100, epochs=100, validation_data=test_set, validation_steps=100)

    model.save("saved_model.h5", True, True)

    onnx = onnxmltools.convert_keras(model)
    onnxmltools.save_model(onnx, "converted.onnx")

    with open("history.json","w+") as file:
       file.write(str(history.history))

    acc_plot(history)
    loss_plot(history)


def evaluate(image_dir, model):
    test_img = image.load_img(image_dir, target_size=(256, 256))
    test_img = image.img_to_array(test_img)
    test_img = np.expand_dims(test_img, axis=0)
    result = model.predict(test_img)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str,
                        default=None,
                        help="Input directory where where training dataset and meta data are saved",
                        required=False
                        )
    parser.add_argument("--output_dir", type=str,
                        default=None,
                        help="Input directory where where logs and models are saved",
                        required=False
                        )

    args, unknown = parser.parse_known_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    log_dir = output_dir

    main()
