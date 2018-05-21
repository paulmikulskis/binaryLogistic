
# binaryLogistic

## A Simple Binary Logitic Classification Model on Two Datasets


#### Goal
---
The goal of this project was to get myself familiar with binary logistic regression.
To do this, I tried identifying pictures of cats, among non-cat images.  The data would be
classified as 1 if it were a cat, and 0 otherwise.

#### Project Layout
---
This project has three important files:
* main.py
* main2.py
* network.py

Each of the 'main' files works with a different dataset.  The first one works with a pre-procured cat vs. non-cat dataset in the format of an h5 file that I obtained from Andrew Ng's deep learning AI course.  The second one works with the CIFAR-10 dataset, and has some code to procure and sort through the data before running the model.  

The network.py file contains the object representing my small neural net.  This file does the processing of data given from any dataset as long as it is in the correct format.  
#### How To Run
---
To run the model, just run either of the main files with

    python main.py
or

    python main2.py

 And the accuracy of the model's prediction will output to console along with the logistic cost function evaluation every 100 epochs.  

For customizing different hyper-parameters, constants provided to the Network object can be modified.  When instantiating the Network object, the second agrument is alpha, the learning rate of the network:
`network = Network(X.shape[0],0.0005)`, and when the train() method is called, the third parameter is the number of epochs: `network.train(X,Y, 2000)`.

#### Results, Observations and Additional Notes
---
While playing with this project, I noticed some interesting behaviour that I wanted to note:
1) The dataset I used from Andrew Ng's course contained pictures of cats along with other images of various types.  I was able to aget the accuracy up to 70% using a learning rate of 0.005 and 2,000 epochs.  However, if I changed the learning rate to 0.05 and ran 200 epochs, I noticed an accuracy of 80%.  This led me to believe that though I thought more epochs is good, the data might have been getting overfitted at 2,000 epochs.  

2) When using the CIFAR-10 dataset, the main2.py file procures two sets of data, one of cats, and one of trucks, then chooses 600 images from the union of the two sets.  Using a distribution that had two distinct classes of data lef me to be able to boast accuracy of up to 81% using a learning rate of 0.005 and 2,000 epochs.  I reckon this is because in this training exercise, there were exactly two sets of classifications, and there was probably a better classification of what was and what was not a cat.
