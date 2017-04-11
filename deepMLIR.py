"""
Deep Gllim model in python

__author__ = R.Juge & S.Lathuiliere
"""

import time
import sys
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

from keras.layers.normalization import BatchNormalization

from VGG16_sequential import VGG16, extract_XY_generator
from gllim import GLLIM
from data_generator import load_data_generator
from prob_generator import resp_generators
from test import run_eval
from PCA_init import add_pca_layer
import pickle

ROOTPATH = str(sys.argv[1])

FEATURES_SIZE = 512  # size of the PCA
HIGH_DIM = FEATURES_SIZE
LOW_DIM = 3  # Dimension of the target to modify according to the task

GLLIM_K = 2  # number ov linear regression
MAX_ITER_EM = 100  # maximum iterations for the EM of the gllim layer

WIDTH = 224  # input size
PB_FLAG = 'pose'  #to modify according to the task

BATCH_SIZE = 128

ITER = 12  # number of iterations alternating between M-network steps and gllim layer
NB_EPOCH = 3    # number of epochs to train the model.


LEARNING_RATE = 1e-01

print LEARNING_RATE
JOB = str(sys.argv[4])


Valperc=0.80  # validation_split: float (0. < x < 1).
# Fraction of the data to use as held-out validation data.
               

class DeepGllim:
    ''' Class of deep gllim model'''

    def __init__(self, k, PCA=None):
        
        self.k = k
        self.PCA = PCA
        self.gllim = GLLIM(self.k, HIGH_DIM, LOW_DIM)
        self.network = VGG16(ROOTPATH)
        
    def fit(self, train_txt,test_txt, learning_rate=0.1, it=2):
        '''Trains the model for a fixed number of epochs and iterations.
           # Arguments
                X_train: input data, as a Numpy array or list of Numpy arrays
                    (if the model has multiple inputs).
                Y_train : labels, as a Numpy array.
                batch_size: integer. Number of samples per gradient update.
                learning_rate: float, learning rate
                it: integer, number of iterations of the algorithm
                f: text file for responsability trick

            '''
        start_time_training = time.time()

        print "Training Deep Gllim"

        
        (generator_training, n_train), (generator_val,n_val),(generator_test, n_test) = load_data_generator(ROOTPATH, train_txt, test_txt,validation=Valperc)
        print "n_train size:", n_train
        
        features_training, target_training = extract_XY_generator(self.network, generator_training, n_train)
        print "features size:", features_training.shape
        print "target size:", target_training.shape

        
        add_pca_layer(self.network, features_training, self.PCA)
        self.network.add(BatchNormalization())
        
        for i in range(it):

            # Extract the features used to train the gllim layer
            features_training, target_training = extract_XY_generator(self.network, generator_training, n_train)
            self.gllim.fit(target_training, features_training, MAX_ITER_EM, (i == 0), None)


            # introduced now to evaluate the test set every iteration!
            # self.gllim.evaluate((gen_test, N_test), WIDTH)
                  
            
            # inverse the gllim layer to perform forward evaluation
            self.gllim.inversion()
            print "VALIDATION SET:"
            self.evaluate((generator_val, n_val), WIDTH)
            print "TEST SET:"
            self.evaluate((generator_test, n_test), WIDTH)
            
            # perform the M-network step
            self.fine_tune(16, learning_rate, train_txt)


        # finish by a gllim update
        features_training, target_training = extract_XY_generator(self.network, generator_training, n_train)
        self.gllim.fit(target_training, features_training, MAX_ITER_EM, False, None)
        self.gllim.inversion()

        
        print "--- %s seconds for training Deep Gllim---" % (time.time() - start_time_training)

    def fine_tune(self, layer_nb, learning_rate, data_file):
        '''Fine tune the network according to our custom loss function'''
        
        (generator, N_TRAIN), (generator_val, N_VAL) = resp_generators(ROOTPATH, data_file,
                                                                       self.gllim,
                                                                       batch_size=BATCH_SIZE,validation=Valperc)

        # train only some layers
        for layer in self.network.layers[:layer_nb]:
            layer.trainable = False
        for layer in self.network.layers[layer_nb:]:
            layer.trainable = True
        self.network.layers[-1].trainable = True

        # compile the model
        sgd = SGD(lr=learning_rate,
                  momentum=0.9,
                  decay=1e-06,
                  nesterov=True)

        self.network.compile(optimizer=sgd,
                             loss='mse')

        self.network.summary()


        checkpointer = ModelCheckpoint(filepath=ROOTPATH+"Deep_Gllim_"+PB_FLAG+JOB+"_K"+str(GLLIM_K)+"_weights.hdf5",
                                       monitor='val_loss',
                                       verbose=1,
                                       save_weights_only=True,
                                       save_best_only=True,
                                       mode='min')



        # train the model on the new data for a few epochs
        self.network.fit_generator(generator,
                                   samples_per_epoch=N_TRAIN,
                                   nb_epoch=NB_EPOCH,
                                   verbose=1,
                                   callbacks=[checkpointer],
                                   validation_data=generator_val,
                                   nb_val_samples=N_VAL)

        self.network.load_weights(ROOTPATH+"Deep_Gllim_"+PB_FLAG+JOB+"_K"+str(GLLIM_K)+"_weights.hdf5")

        return self.network

    def predict(self, (generator, n_predict)):
        '''Generates output predictions for the input samples,
           processing the samples in a batched way.
        # Arguments
            generator: input a generator object.
            batch_size: integer.
        # Returns
            A Numpy array of predictions.
        '''
        
        features_test, _ = extract_XY_generator(self.network, generator, n_predict)
        gllim_predict = self.gllim.predict_high_low(features_test)

        return gllim_predict
    
    def evaluate(self, (generator, n_eval), l=WIDTH, pbFlag=PB_FLAG,printError=False):
        '''Computes the loss on some input data, batch by batch.

        # Arguments
            generator: input a generator object.
            batch_size: integer. Number of samples per gradient update.

        # Returns
            Scalar test loss (if the model has no metrics)
            or list of scalars (if the model computes other metrics).
            The attribute `model.metrics_names` will give you
            the display labels for the scalar outputs.
        '''
        
        features_test, target_test = extract_XY_generator(self.network, generator, n_eval)

        gllim_predict = self.gllim.predict_high_low(features_test)
        run_eval(gllim_predict, target_test, l, pbFlag,printError=False)



        
if __name__ == '__main__':

    deep_gllim = DeepGllim(k=GLLIM_K, PCA=FEATURES_SIZE)

    train_txt = sys.argv[2]
    test_txt = sys.argv[3]


    deep_gllim.fit(train_txt,test_txt,
                   learning_rate=LEARNING_RATE,
                   it=ITER)
   
    (generator_training, n_train), (generator_val,n_val),(gen_test, N_test) = load_data_generator(ROOTPATH, train_txt, test_txt,validation=Valperc)

    
    deep_gllim.evaluate((gen_test, N_test), WIDTH,pbFlag=PB_FLAG,printError=True)

