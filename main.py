
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
#prints all the labels in the dataset as a whole
#list(map(lambda x: print(x), label_names))


#collect nxMxMx3 set of images of cat vs non-cat, and their corresponding labels
#in a 1xn matrix, with 1 for cat and 0 for non-cat image
tupledTwoTruck = list(filter(lambda x: x[1] == 9, zip(img, labels)))
trucks_img = [np.transpose(np.reshape(t[0],(3,32,32)), (1,2,0)) for t in tupledTwoTruck]
print('trucks:', len(trucks_img))


#displays a random truck from the dataset
single_img = np.array(trucks_img[randint(0,len(trucks_img))])
plt.imshow(single_img)
#plt.show()

#sets up the array structures for training
X = np.zeros((single_img.shape[0]*single_img.shape[1]*3, 600))
Y = np.zeros((1,600))
for i in range(0,600):
    d = randint(0,1)
    if d == 1:
        X[:,i] = np.reshape(cats_img[i], (single_img.shape[0]*single_img.shape[1]*3))
        Y[0,i] = 1
    else:
        X[:,i] = np.reshape(trucks_img[i], (single_img.shape[0]*single_img.shape[1]*3))
        Y[0,i] = 0
#checking sizes
print('using 600 training examples...')
print('shape of X input:', X.shape)
print('shape of Y input:', Y.shape)
print('number of 1s:', np.where(Y == 1)[0].shape[0])
print('number of 1s:', np.where(Y == 0)[0].shape[0])


#train on the network:
network = Network(single_img.shape[0]*single_img.shape[1]*3,0.2)
network.train(X,Y,100)

#test the network:
#first we must grab the training sets of images:
f = open('/home/paul/python/project_1_ind/binaryLogistic/cifar-10-batches-py/data_batch_2', 'rb')
tupled_data= pickle.load(f, encoding='latin1')
f.close()
#seperate the image and label data
img2 = tupled_data["data"]
labels2 = tupled_data['labels']
#rezip images and labels into list of tuples [(image, label)]
pairs = zip(img2, labels2)
#filter out only pictures with label cat (label = 3)
tupledTwoCats2 = list(filter(lambda x: x[1] == 3, pairs))
#actual image data of all cats, reshaped into MxNx3
cats_img2 = [np.transpose(np.reshape(t[0],(3,32,32)), (1,2,0)) for t in tupledTwoCats2]

#collect nxMxMx3 set of images of cat vs non-cat, and their corresponding labels
#in a 1xn matrix, with 1 for cat and 0 for non-cat image
tupledTwoTruck2 = list(filter(lambda x: x[1] == 9, zip(img2, labels2)))
trucks_img2 = [np.transpose(np.reshape(t[0],(3,32,32)), (1,2,0)) for t in tupledTwoTruck2]


X = np.zeros((single_img.shape[0]*single_img.shape[1]*3, 600))
Y = np.zeros((1,600))
for i in range(0,600):
    d = randint(0,1)
    if d == 1:
        X[:,i] = np.reshape(cats_img2[i], (single_img.shape[0]*single_img.shape[1]*3))
        Y[0,i] = 1
    else:
        X[:,i] = np.reshape(trucks_img2[i], (single_img.shape[0]*single_img.shape[1]*3))
        Y[0,i] = 0

accuracy = network.test(X,Y)
print('accuracy of the network = ', accuracy)
