'''VGG16 model for Keras.
# Reference:
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
'''

import warnings

import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, activations
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras.layers.normalization import BatchNormalization
from keras import backend as K

def VGG16(ROOTPATH):
    # Returns
    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        INPUT_SHAPE = (3, 224, 224)
    else:
        INPUT_SHAPE = (224, 224, 3)

    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=INPUT_SHAPE))
    model.add(Convolution2D(64, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
        
    model.add(Flatten())
    model.add(Dense(4096, activation='relu', trainable=True))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', trainable=True))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax', trainable=True))
        


    weights_path = ROOTPATH+"vgg16_weights_init.h5"
    print "LOAD: " +weights_path
    model.load_weights(weights_path)
    model.pop()  # remove softmax layer
    model.pop()  # remove dropout

    return model

def extract_features_generator(network, generator, size):
    '''Extract VGG features from a generator'''
    
    print("Extracting features :")
    
    features = network.predict_generator(generator, val_samples=size)
    
    return features

def extract_features(network, x):
    '''Extract VGG features from a generator'''
    
    print("Extracting features :")
    
    features = network.predict(x, batch_size=64)
    
    return features

def extract_XY_generator(network, generator, size):
    '''Extract VGG features and data targets from a generator'''
    
    i=0
    X=[]
    Y=[]
    for x,y in generator:
        X.extend(network.predict_on_batch(x))
        Y.extend(y)
        i+=len(y)
        if i>=size:
            break
        
    return np.asarray(X), np.asarray(Y)

