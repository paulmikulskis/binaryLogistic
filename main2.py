
from six.moves import cPickle as pickle
from  PIL import Image
from random import randint
from network import Network
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#open file from CIFAR10
f = open('/home/paul/python/project_1_ind/binaryLogistic/cifar-10-batches-py/data_batch_1', 'rb')
tupled_data= pickle.load(f, encoding='latin1')
f.close()
#seperate the image and label data
img = tupled_data["data"]
labels = tupled_data['labels']
#rezip images and labels into list of tuples [(image, label)]
pairs = zip(img, labels)
#filter out only pictures with label cat (label = 3)
tupledTwoCats = list(filter(lambda x: x[1] == 3, pairs))
#actual image data of all cats, reshaped into MxNx3
cats_img = [np.transpose(np.reshape(t[0],(3,32,32)), (1,2,0)) for t in tupledTwoCats]
#prints the number of cats we have
print('cats:',len(cats_img))

#supplementary section of code for opening up the label-string list
f = open('/home/paul/python/project_1_ind/binaryLogistic/cifar-10-batches-py/batches.meta', 'rb')
tupled_data= pickle.load(f, encoding='latin1')
f.close()
label_names = tupled_data['label_names']
#prints all the labels in the dataset as a whole, if you wish to see all the labels available in the dataset
#list(map(lambda x: print(x), label_names))


#collect nxMxMx3 set of images of cat vs non-cat, and their corresponding labels
#in a 1xn matrix, with 1 for cat and 0 for non-cat image
#'truck' images have label number 9
tupledTwoTruck = list(filter(lambda x: x[1] == 9, zip(img, labels)))
trucks_img = [np.transpose(np.reshape(t[0],(3,32,32)), (1,2,0)) for t in tupledTwoTruck]
print('trucks:', len(trucks_img))


#grabs a single truck image from the dataset and initializes the array X which is a
#[ (32x32x3) x 600] array containing examples, and Y which is a [1 x 600] array containing the labels
single_img = np.array(trucks_img[randint(0,len(trucks_img))])
X = np.zeros((single_img.shape[0]*single_img.shape[1]*3,600))
Y = np.zeros((1,600))
for i in range(1,600):
    #here we choose a random integer 600 times.  If 1 is chosen, then stick a 'cat' image
    #in the array of examples, and if 0 is chosen, then stick in a 'truck' image
    d = randint(0,1)
    if d == 1:
        X[:,i] = np.reshape(cats_img[i], (single_img.shape[0]*single_img.shape[1]*3))
        Y[0,i] = 1
    else:
        X[:,i] = np.reshape(trucks_img[i], (single_img.shape[0]*single_img.shape[1]*3))
        Y[0,i] = 0

#normalize
X = X/255

#train on the network:
network = Network(X.shape[0],0.0005)
network.train(X,Y, 2000)


#test the network:
#first we must grab the training sets of images:
f = open('/home/paul/python/project_1_ind/binaryLogistic/cifar-10-batches-py/data_batch_2', 'rb')
tupled_data= pickle.load(f, encoding='latin1')
f.close()
#seperate the image and label data
img2 = tupled_data["data"]
labels2 = tupled_data['labels']
#filter out only pictures with label cat (label = 3)
tupledTwoCats2 = list(filter(lambda x: x[1] == 3, zip(img2, labels2)))
#actual image data of all cats, reshaped into MxNx3
cats_img2 = [np.transpose(np.reshape(t[0],(3,32,32)), (1,2,0)) for t in tupledTwoCats2]

#just like above, procure the data in a random fashion
tupledTwoTruck2 = list(filter(lambda x: x[1] == 9, zip(img2, labels2)))
trucks_img2 = [np.transpose(np.reshape(t[0],(3,32,32)), (1,2,0)) for t in tupledTwoTruck2]
for i in range(0,600):
    d = randint(0,1)
    if d == 1:
        X[:,i] = np.reshape(cats_img2[i], (single_img.shape[0]*single_img.shape[1]*3))
        Y[0,i] = 1
    else:
        X[:,i] = np.reshape(trucks_img2[i], (single_img.shape[0]*single_img.shape[1]*3))
        Y[0,i] = 0

#normalize
X = X/255
accuracy = network.test(X,Y)
print('accuracy of the network = ', accuracy)
