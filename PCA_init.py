import numpy as np
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, activations
from sklearn.decomposition import PCA
import time

def add_pca_layer(network, X_train, size):
    '''add a keras layer of PCA'''
    
    print "IN PCA_init"
    
    start_time_pca = time.time()
    
    pca = PCA(n_components=size)

    pca.fit(X_train)

    network.add(Dense(size, weights=[pca.components_.T,np.zeros(size)], activation=activations.get('linear'), trainable=True))
    network.compile(optimizer='sgd', loss='mse')
    
    print "PCA_init OUT"

    print("--- %s seconds for PCA initialisation---" % (time.time() - start_time_pca))
