from six.moves import cPickle as pickle
from  PIL import Image
from random import randint
from network import Network
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py

#This loads the H5 dataset from Andrew Ng's deep learning course dataset of cat vs. non cat images
def load_dataset():
    train_dataset = h5py.File('catvnoncat/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
    test_dataset = h5py.File('catvnoncat/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set label
    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

#load the data
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
n_train = train_set_x_orig.shape[0]
n_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.

#sets up the array structures for training
X = train_set_x
Y = train_set_y

#train on the network:
network = Network(X.shape[0],0.005)
network.train(X,Y, 2000)

#arrange testing data
X = test_set_x
Y = test_set_y

#test the network
accuracy = network.test(X,Y)
print('accuracy of the network = ', accuracy)
