"""
Gllim model in python

__author__ = R.Juge & S.Lathuiliere

The equation numbers refer to _High-Dimensional Regression with Gaussian Mixtures and Partially-Latent Response Variables_A. Deleforge 2015
"""

import numpy as np
from numpy.linalg import inv
from scipy.misc import logsumexp
from sklearn.mixture import GMM
import matplotlib.pyplot as plt
from itertools import compress
import time
import pickle
from log_gauss_densities import loggausspdf,chol_loggausspdf
from data_generator import get_image_for_vgg

class GLLIM:
    ''' Gaussian Locally-Linear Mapping'''
    
    def __init__(self,K_in,D_in,L_in):

        self.K = K_in
        self.D = D_in
        self.L = L_in

        #inverse regression (used for training)
        self.AkList=[np.random.randn(self.D, self.L) for k in range(self.K)]
        self.bkList=[np.random.randn(self.D,1) for k in range(self.K)]
        self.ckList=[np.random.randn(self.L,1) for k in range(self.K)]
        self.GammakList=[np.ones((self.L, self.L)) for k in range(self.K)]
        self.pikList=[1.0/self.K for k in range(self.K)]
        self.SigmakSquareList=[np.ones((self.D,1)) for k in range(self.K)]
        self.rnk = np.empty((1,1)) # need to be allocated
        
        #direct regression (used for testing)
        self.AkListS=[np.empty((self.L, self.D)) for k in range(self.K)]
        self.bkListS=[np.empty((self.L,1)) for k in range(self.K)]
        self.ckListS=[np.empty((self.D,1)) for k in range(self.K)]
        self.GammakListS=[np.empty((self.D, self.D))for k in range(self.K)]
        self.SigmakListS=[np.empty((self.L, self.L)) for k in range(self.K)]
        
    def fit(self, X, Y, maxIter, init, gmm_init=None):
        '''fit the Gllim
           # Arguments
            X: low dimension targets as a Numpy array
            Y: high dimension features as a Numpy array
            maxIter: maximum number of EM algorithm iterations
            init: boolean, compute GMM initialisation
            gmm_init: give a GMM as init
        '''

        N = X.shape[0]
        LL=np.ndarray((100,1))
        it = 0
        converged = False
        
        if init==True:
            
            print "Start initialization"
            deltaLL = float('inf')
            print "X : ", X.shape
            print "Y : ", Y.shape

            # we initialize the model by running a GMM on the target only
            datas_matrix = X

            # uncomment the following line if you want to initialize the model on the complete data
            # datas_matrix = np.concatenate((X, Y), axis=1) #complete data matrix

            print "datas matrix shape:", datas_matrix.shape
            
            print "Initialization of posterior with GMM"
            start_time_EMinit = time.time()
            
            gmm = GMM(n_components=self.K, covariance_type='diag', random_state=None, tol=0.001, min_covar=0.001, n_iter=100, n_init=3, params='wmc', init_params='wmc', verbose=1)
            
            if gmm_init==None:
                gmm.fit(datas_matrix)
            else:
                gmm = pickle.load(open(gmm_init, 'r'))
                gmm.fit(datas_matrix)
            
            self.rnk = gmm.predict_proba(datas_matrix)
            rkList = [np.sum(self.rnk[:,k]) for k in range(self.K)]            
            
            print("--- %s seconds for EM initialization---" % (time.time() - start_time_EMinit))
            
        print "Training with EM"
        start_time_EM = time.time()
        logrnk = np.ndarray((N,self.K))
        rkList = [np.sum(self.rnk[:,k]) for k in range(self.K)]
        
        while (converged==False) and (it<maxIter):

            it += 1 

            print "Iteration nb "+str(it)
            
            #  M-GMM-step:
            print "M-GMM"
            
            self.pikList=[rk/N for rk in rkList] # (28)

            self.ckList=[np.sum(self.rnk[:,k]*X.T,axis=1)/rk for k,rk in enumerate(rkList)]

            self.GammakList=[np.dot((np.sqrt(self.rnk[:,k]).reshape((1,N)))*(X.T-ck.reshape((self.L,1))),((np.sqrt(self.rnk[:,k]).reshape((1,N)))*(X.T-ck.reshape((self.L,1)))).T)/rk for k,ck,rk in zip(range(self.K),self.ckList,rkList)]  # (30)

            # M-mapping-step
            print"M-mapping"
            xk_bar = [np.sum(self.rnk[:,k]*X.T,axis=1)/rk for k,rk in enumerate(rkList)]# (35)
            
            yk_bar = [np.sum(self.rnk[:,k]*Y.T,axis=1)/rk for k,rk in enumerate(rkList)]  # (36)

            XXt_stark=np.zeros((self.L,self.L))
            YXt_stark=np.zeros((self.D,self.L))

            for k,rk,xk,yk in zip(range(self.K),rkList,xk_bar,yk_bar):
                
                X_stark=(np.sqrt(self.rnk[:,k]))*(X-xk).T  # (33)
                Y_stark=(np.sqrt(self.rnk[:,k]))*(Y-yk).T  # (34)
                XXt_stark=np.dot(X_stark,X_stark.T)
                YXt_stark=np.dot(Y_stark,X_stark.T)                
                self.AkList[k]=np.dot(YXt_stark,inv(XXt_stark))
            
            self.bkList=[np.sum(self.rnk[:,k].T*(Y-(Ak.dot(X.T)).T).T,axis=1)/rk for k,Ak,rk in zip(range(self.K),self.AkList ,rkList)]  # (37)
            
            diffSigmakList = [np.sqrt(self.rnk[:,k]).T*(Y-(Ak.dot(X.T)).T-bk.reshape((1,self.D))).T for k,Ak,bk in zip(range(self.K),self.AkList,self.bkList)]
           
            sigma2 = [np.sum((diffSigma**2),axis=1)/rk for rk,diffSigma in zip(rkList,diffSigmakList)]
            
            # isotropic sigma
            self.SigmakSquareList = [(np.sum(sig2)/self.D) for sig2 in sigma2] 

            # numerical stability
            self.SigmakSquareList = [sig + 1e-08 for sig in self.SigmakSquareList] 
            
            # equal constraints
            self.SigmakSquareList = [self.SigmakSquareList[k]*rk for k,rk in enumerate(rkList)]
            self.SigmakSquareList = [(sum(self.SigmakSquareList)/N)]*self.K 
         
            
            #  E-Z step:
            print "E-Z"
            for (k, Ak, bk, ck, pik, gammak, sigmakSquare) in zip(range(self.K), self.AkList, self.bkList, self.ckList, self.pikList, self.GammakList, self.SigmakSquareList):
                
                y_mean = np.dot(Ak,X.T) + bk.reshape((self.D,1))
                logrnk[:,k] = np.log(pik) + chol_loggausspdf(X.T, ck.reshape((self.L,1)), gammak) +  loggausspdf(Y, y_mean.T, sigmakSquare)

            lognormrnk = logsumexp(logrnk,axis=1,keepdims=True)
            logrnk -= lognormrnk
            self.rnk = np.exp(logrnk)
            LL[it,0] = np.sum(lognormrnk) # EVERY EM Iteration THIS MUST INCREASE
            print "Log-likelihood = " + str(LL[it,0]) + " at iteration nb :" + str(it)
            rkList=[np.sum(self.rnk[:,k]) for k in range(self.K)]
            
            # Remove empty clusters
            ec = [True]*self.K
            cpt = 0
            for k in range(self.K):
                if (np.sum(self.rnk[:,k])==0) or (np.isinf(np.sum(self.rnk[:,k]))):
                    cpt +=1
                    ec[k] = False
                    print "class ",k," has been removed"
            self.K -= cpt
            rkList = list(compress(rkList, ec))
            self.AkList = list(compress(self.AkList, ec))
            self.bkList = list(compress(self.bkList, ec))
            self.ckList = list(compress(self.ckList, ec))
            self.SigmakSquareList = list(compress(self.SigmakSquareList, ec))
            self.pikList = list(compress(self.pikList, ec))
            self.GammakList = list(compress(self.GammakList, ec))
            
            if (it>=3):
                deltaLL_total = np.amax(LL[1:it,0])-np.amin(LL[1:it,0])
                deltaLL = LL[it,0]-LL[it-1,0]
                converged = bool(deltaLL <= 0.001*deltaLL_total)
                        
        print "Final log-likelihood : " + str(LL[it,0])

        print(" Converged in %s iterations" % (it))

        print("--- %s seconds for EM ---" % (time.time() - start_time_EM))

        # plt.plot(LL[1:it,0])
        # plt.show()
        
        return LL[1:it,0]

    def inversion(self):
        ''' Bayesian inversion of the parameters'''
        
        # Inversion step
        print "Proceeding to the inversion"
        start_time_inversion = time.time()
        
        self.ckListS[:]=[Ak.dot(ck)+bk for Ak,bk,ck in zip(self.AkList,self.bkList,self.ckList)]  # (9)
        
        self.GammakListS[:]=[sig*np.eye(self.D)+Ak.dot(gam).dot(Ak.T) for sig,gam,Ak in zip(self.SigmakSquareList,self.GammakList,self.AkList)]  # (10)
        
        self.SigmakListS[:]=[inv(inv(gam)+(Ak.T).dot(1/sig*Ak)) for sig,gam,Ak in zip(self.SigmakSquareList,self.GammakList,self.AkList)]  # (14)
        
        self.AkListS[:]=[sigS.dot(1/sig*Ak.T) for Ak,sigS,sig in zip(self.AkList,self.SigmakListS,self.SigmakSquareList)]  # (12)
        
        self.bkListS[:]=[sigS.dot(inv(gam).dot(ck)-(1/sig*Ak.T).dot(bk)) for Ak,bk,ck,sig,sigS,gam in zip(self.AkList,self.bkList,self.ckList,self.SigmakSquareList,self.SigmakListS,self.GammakList)]  # (13)
        
        print("--- %s seconds for inversion ---" % (time.time() - start_time_inversion))
        
    def predict_high_low(self,Y):
        '''Forward prediction'''
    
        N = Y.shape[0]

        proj = np.empty((self.L, N, self.K))
        logalpha = np.zeros((N,self.K))

        for (k,pik,AkS,bkS,ckS,gamkS) in zip(range(self.K),self.pikList,self.AkListS,self.bkListS,self.ckListS,self.GammakListS):
            
            proj[:,:,k] = AkS.dot(Y.T)+np.expand_dims(bkS,axis=1)
            logalpha[:,k] = np.log(pik) + chol_loggausspdf(Y.T, ckS.reshape((self.D,1)), gamkS)

        density = logsumexp(logalpha,axis=1,keepdims=True)
        logalpha -= density
        alpha = np.exp(logalpha)

        Xpred = np.sum(alpha.reshape((1,N,self.K))*proj,axis=2) # (16)

        return Xpred.T
    
    def predict_low_high(self,X):
        '''Backward prediction'''

        N = X.shape[0]

        proj = np.empty((self.D, N, self.K))
        logalpha = np.zeros((N,self.K))

        for (k,pik,Ak,bk,ck,gamk) in zip(range(self.K),self.pikList,self.AkList,self.bkList,self.ckList,self.GammakList):
            
            proj[:,:,k] = Ak.dot(X.T) + np.expand_dims(bk,axis=1)
            logalpha[:,k] = np.log(pik) + chol_loggausspdf(X.T, ck.reshape((self.L,1)), gamk)

        density = logsumexp(logalpha,axis=1,keepdims=True)
        logalpha -= density
        alpha = np.exp(logalpha)

        Ypred = np.sum(alpha.reshape((1,N,self.K))*proj,axis=2) # (15)

        return Ypred.T


    def get_rnk_batch(self, X, Y):
        '''Get rnk for a batch by considering only target responsabilities'''
        
        logrnk = np.ndarray((len(X),self.K))
        #phiX=network.predict_on_batch(np.asarray(X))
        Y=np.asarray(Y)
        
        for (k, ck, pik, gammak) in zip(range(self.K),self.ckList, self.pikList, self.GammakList):
            # Warning here we removed the term depending on the high dimensional space. We notice that it helps to converge
            logrnk[:,k] = np.log(pik) + chol_loggausspdf(Y.T, ck.reshape((self.L,1)), gammak) 
        lognormrnk = logsumexp(logrnk,axis=1,keepdims=True)
        logrnk -= lognormrnk

        return np.exp(logrnk)
    
    def get_rnk(self, path, X, Y, batch_size):  # Return the cumulative probability used to sample the rnk
        '''Get rnk of all data'''
        
        rnk=np.empty((len(X),self.K))
        nbatches=len(X)/batch_size
        
        for i in range(nbatches):  # for each batch

            X_out=[]
            Y_out=[]
            for x,y in zip(X[i*batch_size:(i+1)*batch_size],Y[i*batch_size:(i+1)*batch_size]):  # for each image
                im = get_image_for_vgg(path+x)
                X_out.append(im)
                Y_out.append(y)
            rnk[i*batch_size:(i+1)*batch_size,:] = self.get_rnk_batch(X_out,Y_out)

        return rnk
        
