Deep Mixture of Linear Inverse Regressions

## Introduction.

This is a Keras implementation of the work :
Deep Mixture of Linear Inverse Regressions Applied to Head-Pose Estimation, Stéphane Lathuilière, Rémi Juge, Pablo Mesejo, Rafael Muñoz Salinas, Radu Horaud, CVPR 2017

For more details [Project](https://team.inria.fr/perception/research/dmlir/) or [pdf](https://hal.inria.fr/hal-01504847/document)

Tested with keras 1.1.0 with theano backend and python 2.7.12
Requieres the installation of scikit-learn.

------------------
## How to run:

trainingAnnotations.txt must contain the list of the training images followed by the targets:
```
img_name_1.jpg y1 y2 y3
img_name_2.jpg y1 y2 y3 
...
```

testAnnotations.txt must contain the list of the test images with the same format

Download the [VGG16 weights](https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view)

Run the following command:
```shell
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,exception_verbosity='high' python  $rootpathData deepMLIR.py trainingAnnotations.txt testAnnotations.txt $JOB_ID
```
where JOB_ID is a job id used to save the network weights. You can give any number. $rootpathData is the path to your dataset folder. The file vgg16_weights.h5 must be moved in the $rootpathData folder.

------------------


## Support

For any question, please contact [Stéphane Lathuilière](https://team.inria.fr/perception/team-members/stephane-lathuiliere/).