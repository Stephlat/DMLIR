import numpy as np
import cPickle as pickle
from sklearn.metrics import mean_absolute_error, explained_variance_score

def run_eval(Y_pred, Y_true, l, pbFlag,idOar="",printError=False):
    print "Evaluating"

 
    # mean absolute error
    MAE = mean_absolute_error(Y_true, Y_pred, multioutput='raw_values')
    evs = explained_variance_score(Y_true, Y_pred, multioutput='raw_values')

    # Head pose estimation: pitch, yaw, roll
    print('Mean absolute error:', MAE,np.sum(MAE)/MAE.shape[0])
    print('Explained variances score:', evs)
    print np.sum(MAE)/MAE.shape[0]
