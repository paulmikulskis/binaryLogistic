
from six.moves import cPickle as pickle
from  PIL import Image
from random import randint
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
tupledTwo = list(filter(lambda x: x[1] == 3, pairs))
#actual image data of all cats
cats_img = [t[0] for t in tupledTwo]
#prints the number of cats we have
print(len(cats_img))

#supplementary section of code for opening up the label-string list
f = open('/home/paul/python/project_1_ind/binaryLogistic/cifar-10-batches-py/batches.meta', 'rb')
tupled_data= pickle.load(f, encoding='latin1')
f.close()
label_names = tupled_data['label_names']
#prints all the labels in the dataset as a whole
list(map(lambda x: print(x), label_names))

#displays a random cat from the dataset
single_img = np.array(cats_img[randint(0,len(cats_img))])
#reshapes the image data into something usable
single_img_reshaped = np.transpose(np.reshape(single_img,(3, 32,32)), (1,2,0))
plt.imshow(single_img_reshaped)
plt.show()
