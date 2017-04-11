''' Create generators from dataset '''

import numpy as np
import cv2
import random

HIGH_DIM = 512
GLLIM_K = 1


BATCH_SIZE = 128

# Mode for the validation set for our mixture model
rnEqui=1
rnHard=2
rnTra=3

def load_data_generator_List(rootpath, imIn, file_test, validation=1.0,subsampling=1.0,processingTarget=None,transform=[],outSize=(224,224),batch_size=BATCH_SIZE,shuffle=False):
    ''' create generators from data'''

    
    def generator(rootpath, images):
        
        N=len(images)
        nbatches=N/batch_size+1
        if N%batch_size==0:
            nbatches-=1
        if shuffle:
            random.shuffle(images)

        i=0
        while 1:
            X, Y = get_xy_from_file(rootpath, images[i*batch_size:(i+1)*batch_size],processingTarget=processingTarget,transform=transform,outSize=outSize)
            yield(X, Y)
            i=i+1
            if i>=nbatches:  # we shuffle the data when the end of the dataset is reached
                i=0
                random.shuffle(images)

    imTest = open(rootpath+file_test, 'r').readlines()
    gen_test = generator(rootpath, imTest)
    test_size=len(imTest)





    # we subsample the data if needed
    if subsampling!=1.0:
        im=imIn[0:int(subsampling*len(imIn))][:]
    else:
        im=imIn[:]
        
    if validation!=1.0:  # if we use a validation set
        Ntot=len(im)
        training_size = int(validation*len(im))
        val_size = Ntot-training_size
        
        gen_train = generator(rootpath, im[:training_size])
        gen_val = generator(rootpath, im[training_size:])

        return (gen_train,training_size),(gen_val,val_size), (gen_test,test_size)
    else:  # without validation set
        gen_train = generator(rootpath, im)
        training_size = len(im)
     
        return (gen_train,training_size), (gen_test,test_size)

def load_data_generator(rootpath, file_train, file_test, validation=1.0,subsampling=1.0,processingTarget=None,transform=[],outSize=(224,224),batch_size=BATCH_SIZE,shuffle=False):
    im = open(rootpath+file_train, 'r').readlines()
    return load_data_generator_List(rootpath, im[:], file_test, validation,subsampling,processingTarget=processingTarget,transform=transform,outSize=outSize,batch_size=batch_size,shuffle=shuffle)

def load_data_generator_List_simple(rootpath, imIn,transform=[],outSize=(224,224),batch_size=BATCH_SIZE,processingTarget=None,sample_weights=None):
    ''' create generators from data'''

    
    def generator(rootpath, images, batch_size=BATCH_SIZE):
        
        N=len(images)
        nbatches=N/batch_size+1
        if N%batch_size==0:
            nbatches-=1
        i=0
        if sample_weights is not None:
            rn= sample_weights[:]
        while 1:

            X, Y = get_xy_from_file(rootpath, images[i*batch_size:(i+1)*batch_size],processingTarget=processingTarget,transform=transform,outSize=outSize)
            if sample_weights is None:
                yield(X, Y)
            else:
                yield(X, Y,rn[i*batch_size:(i+1)*batch_size])
            i=i+1
            if i>=nbatches:  # we shuffle the data when the end of the dataset is reached
                i=0
                if sample_weights is None:
                    random.shuffle(images)
                else:
                    c = zip(images,rn)
                    np.random.shuffle(c)
                    images = np.asarray([e[0] for e in c])
                    rn = np.asarray([e[1] for e in c])

                    

    gen = generator(rootpath, imIn[:])
    size=len(imIn)

    return (gen,size)

def load_data_generator_simple(rootpath, fileName, transform=[],outSize=(224,224),batch_size=BATCH_SIZE,processingTarget=None):
    im = open(rootpath+fileName, 'r').readlines()
    return load_data_generator_List_simple(rootpath, im[:],transform=transform,outSize=outSize,processingTarget=processingTarget)

def load_data_generator_Uniform_List(rootpath, imIn, file_test,rni, valMode=rnEqui,validation=0.8,subsampling=1.0,outSize=(224,224),processingTarget=None,batch_size=BATCH_SIZE):
    ''' create generators from data for a mixure of gaussian + uniform'''

    
    def generator(rootpath, images,rniIn, batch_size=BATCH_SIZE):
        # im = get_sublist(im,rniIn)
        c = zip(images,rniIn)
        np.random.shuffle(c)
        im = np.asarray([e[0] for e in c])
        rni = np.asarray([e[1] for e in c])


        N=len(im)

        print "Size of the Selected Data: " + str(N)
        nbatches=N/batch_size
        if N%batch_size==0:
            nbatches-=1
        
        i=0
        while 1:
            X, Y = get_xy_from_file(rootpath, im[i*batch_size:(i+1)*batch_size],processingTarget=processingTarget,transform=None,outSize=outSize)

            yield([X,rni[i*batch_size:(i+1)*batch_size]], Y*rni[i*batch_size:(i+1)*batch_size])
            i=i+1
            if i>=nbatches:  # we shuffle the data when the end of the dataset is reached
                i=0
                c = zip(images,rniIn)
                np.random.shuffle(c)
                im = np.asarray([e[0] for e in c])
                rni = np.asarray([e[1] for e in c])


                # im = get_sublist(images,rniIn)
                # N=len(im)
                # nbatches=N/batch_size
                # if N%batch_size==0:
                #     nbatches-=1


    # we subsample the data if needed
    imCp=imIn[:]
    if subsampling!=1.0:
        imCp=imCp[0:int(subsampling*len(imCp))]
    rniCp=rni[:]
    Ntot=len(imCp)
    training_size = int(validation*len(imCp))
    val_size = Ntot-training_size

    gen_train = generator(rootpath, imCp[:training_size], rniCp[:training_size])
    
    # gen_val = generator(rootpath, imCp[training_size:], np.ones(len(imCp[training_size:]),dtype=np.float))
    if valMode==rnHard:
        LOW_DIM=rni.shape[1]
        rniVal=rniCp[training_size:,:]
        rnOut=np.ones(rniVal.shape)
        for i in range(LOW_DIM):
            nbOutTraining=(len([True for rn in rniVal[:,i] if rn < 0.5]))  # We count the number of outliers in the training set
            indexes=np.argsort(rniVal)  # We get the indexes of the srted rn for the validation set
            nbOutVal=int(float(nbOutTraining)/training_size*val_size)
            for idx in range(nbOutVal):
                rnOut[indexes[idx],i]=0
        gen_val = generator(rootpath, imCp[training_size:], rnOut)
    elif valMode==rnEqui:
        gen_val = generator(rootpath, imCp[training_size:], np.ones(rniCp[training_size:].shape))
    else:  # rn from training mixture
        gen_val = generator(rootpath, imCp[training_size:], rniCp[training_size:])

        
    return (gen_train,training_size),(gen_val,val_size)

def load_data_generator_Uniform(rootpath, file_train, file_test,rni, valMode=rnEqui, validation=0.8,subsampling=1.0,outSize=(224,224),processingTarget=None,batch_size=BATCH_SIZE):
    imFile = open(rootpath+file_train, 'r').readlines()
    return load_data_generator_Uniform_List(rootpath, imFile[:], file_test,rni, valMode, validation,subsampling,outSize,processingTarget=processingTarget)



def load_data_generator_Uniform_List_Simple(rootpath, imIn, rni,outSize=(224,224),processingTarget=None,batch_size=BATCH_SIZE):
    ''' create generators from data for a mixure of gaussian + uniform'''

    
    def generator(rootpath, images,rniIn, batch_size=BATCH_SIZE):
        # im = get_sublist(im,rniIn)
        c = zip(images,rniIn)
        np.random.shuffle(c)
        im = np.asarray([e[0] for e in c])
        rni = np.asarray([e[1] for e in c])

        N=len(im)
        print "Size of the Selected Data: " + str(N)
        nbatches=N/batch_size
        if N%batch_size==0:
            nbatches-=1
        
        i=0
        while 1:
            X, Y = get_xy_from_file(rootpath, im[i*batch_size:(i+1)*batch_size],processingTarget=processingTarget,transform=None,outSize=outSize)
            
            yield([X,rni[i*batch_size:(i+1)*batch_size]], Y*rni[i*batch_size:(i+1)*batch_size])
            i=i+1
            if i>=nbatches:
                i=0
                c = zip(images,rniIn)
                np.random.shuffle(c)
                im = np.asarray([e[0] for e in c])
                rni = np.asarray([e[1] for e in c])

    gen_train = generator(rootpath, imIn[:], rni[:])

        
    return (gen_train,len(imIn))

    

def load_data_generator_Uniform_Simple(rootpath, file_train, file_test,rni, subsampling=1.0,outSize=(224,224),processingTarget=None,batch_size=BATCH_SIZE):
    imFile = open(rootpath+file_train, 'r').readlines()
    return load_data_generator_Uniform_List_Simple(rootpath, imFile[:], rni, subsampling,outSize,processingTarget=processingTarget)

def load_data_generator_noise(rootpath, file_train, validation=0.8,subsampling=1.0):


    def generator(rootpath, images, batch_size=BATCH_SIZE):
        N=len(images)
        nbatches=N/batch_size
        if N%batch_size==0:
            nbatches-=1
        
        i=0
        while 1:
            sol=[x.strip().split(" ") for x in images[i*batch_size:(i+1)*batch_size]]

            yield([[x[0],map(lambda y:int(y),x[1:])] if len(x)>1 else [x[0],[]] for x in sol])
            i=i+1
            if i>=nbatches:
                i=0
                
    imFile = open(rootpath+file_train, 'r').readlines()

    im=imFile[0:int(subsampling*len(imFile))]
    Ntot=len(im)
    training_size = int(validation*Ntot)
    gen_train = generator(rootpath, im[:training_size])
    gen_val = generator(rootpath, im[training_size:])
    return gen_train, gen_val


def applyTransform(x,transform):
    for t in transform:
        x=t(x)
    return x

    
def get_xy_from_file(rootpath, images, processingTarget=None,transform=[],outSize=(224,224),batch_size=BATCH_SIZE):
    '''Extract data arrays from text file'''
    
    X = np.zeros((len(images),3, outSize[0], outSize[1]), dtype=np.float32)
    Y=[]

    
    for i,image in enumerate(images):
        currentline=image.strip().split(" ")
        
        imFile=currentline[0]

        X[i]=get_image_for_vgg(rootpath+imFile,transform,outSize)
            
        Y.append(np.asarray(map(lambda x: float(x),currentline[1:])))


    if processingTarget:
        Y=processingTarget(Y)

    Y=np.squeeze(np.asarray(Y)).reshape((X.shape[0],len(Y[0])))
    return (X,Y)

def get_image_for_vgg(imName,transform=[],outSize=(224,224),batch_size=BATCH_SIZE):
    '''Preprocess images as VGG inputs'''
    im = (cv2.resize(cv2.imread(imName), (outSize[1],outSize[0]))).astype(np.float32)


    # we substract the mean value of imagenet
    if outSize==(224,224):
        im[:,:,0] -= 103.939
        im[:,:,1] -= 116.779
        im[:,:,2] -= 123.68
    im = im.transpose(2,0,1)
    
   
    if transform:
        im=applyTransform(im,transform)

    im = np.expand_dims(im, axis=0)
    
    return im
