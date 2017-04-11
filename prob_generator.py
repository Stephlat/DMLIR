import numpy as np
from data_generator import get_image_for_vgg

OUT_SIZE = 224

def get_random_target(path, X, Y, gllim, Krand):
    '''return (X,Y) for a sample  of cluster Krand'''
    
    X_out = np.empty((len(X),3, OUT_SIZE, OUT_SIZE), dtype=np.float32)
    Y_out = []

    for i,k,x,y in zip(range(len(X)),Krand,X,Y):
        
        im =get_image_for_vgg(path+x)
        X_out[i]=im
        Y_out.append(np.dot(gllim.AkList[k],y)+gllim.bkList[k])

    return (np.asarray(X_out),np.asarray(Y_out))

def resp_generator_list(path, gllim, X_list, Y_list, rnk, batch_size):
    '''Compute the generators from data lists with the distribution trick. Instead of weighting the loss by rnk we train the mse where we select the transformation k we application by drawing it with a propability rnk. In theory it must be assymptotically equivalent and in practice it is strctly the same as all the rnk are either 0 or 1 because of the high dimension of the features'''
    
    
    # Krand = np.argmax(rnk,axis=1)
    n_samples = len(X_list)
    cumul = np.cumsum(rnk, axis=1)
    Rho = np.random.uniform(low=0.0, high=1.0, size=n_samples)
    Krand = [min(np.argmax(cumul[i]>rho), gllim.K-1)for i,rho in enumerate(Rho)]
    
    c = zip(X_list, Y_list, Krand) # shuffle lists
    np.random.shuffle(c)

    X_list = np.asarray([e[0] for e in c])
    Y_list = np.asarray([e[1] for e in c])
    Krand = np.asarray([e[2] for e in c])
    # cumul=np.cumsum(getRnkLi(X_list,Y_list,gllim,network,batch_size), axis=1)
    # Rho = np.random.uniform(low=0.0, high=1.0, size=n_samples)
    # Krand = [min(np.argmax(cumul[i]>rho)-1,gllim.K-1)for i,rho in enumerate(Rho)]

    # we select randomly the cluster
    nbatches=n_samples/batch_size
    
    i=0
    while True:

        X,Y = get_random_target(path, X_list[i*batch_size:(i+1)*batch_size], Y_list[i*batch_size:(i+1)*batch_size],
                                gllim, Krand[i*batch_size:(i+1)*batch_size])
        
        yield (X, Y)
        i=i+1
        if i>=nbatches:  # we shuffle the data when the end of the dataset is reached
            i=0
            c = zip(X_list, Y_list,Krand)
            np.random.shuffle(c)
            X_list = np.asarray([e[0] for e in c])
            Y_list = np.asarray([e[1] for e in c])
            Krand = np.asarray([e[2] for e in c])

def get_list(path, f):
    ''' get lists of data from file '''
    
    imagesLi = open(path+f, 'r').readlines()
    X=[]
    Y=[]

    for currentline in imagesLi:
        i=currentline.split()
        X.append(i[0])
        Y.append(map(lambda x: float(x),i[1:]))
    
    return X, Y

def resp_generators(path, file_txt, gllim, batch_size,validation=0.8):
    '''Create generators from the data test file whose distribution acts like our weighted MSE'''
    
    X_train_List,Y_train_List = get_list(path, file_txt)
    
    totSize = len(X_train_List)
    trainingSize = int(validation*totSize)

    rnk = gllim.get_rnk(path, X_train_List, Y_train_List, batch_size)
    
    genTrain = resp_generator_list(path, gllim, X_train_List[0:trainingSize], Y_train_List[0:trainingSize],
                                   rnk[0:trainingSize], batch_size)
    
    genVal = resp_generator_list(path, gllim, X_train_List[trainingSize:], Y_train_List[trainingSize:],
                                 rnk[trainingSize:], batch_size)
    
    return (genTrain, trainingSize), (genVal, totSize-trainingSize)
