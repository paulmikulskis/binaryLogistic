import numpy as np

class Network:

    #initializes this simple neural net with weights and bias to zero, and learning rate to 'a'
    #self.num is the number of training examples we are working with
    def __init__(self, w, a):
        self.weights = np.zeros((w,1))
        self.bias = 0
        self.alpha = a
        self.num = 0


    #Calculates the sigmoid function given this model's weights and biases
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    #this will loop for the specified number of epochs, performing gradient decent to
    #tune the bias and weight terms for this sigmoid neuron
    def train(self,X, Y, e):
        #sets the number of training examples we have
        self.num = X.shape[1]
        for i in range(0,e):
            z = np.dot(self.weights.T,X) + self.bias
            predict = self.sigmoid(z)
            #calculates cost of this model
            cost = -np.sum(Y*np.log(predict) + (1-Y)*np.log(1-predict))/self.num
            #calculates derivatives of the parameters with respect to output
            dz = predict - Y
            dw = (1/self.num) * np.dot(X,dz.T)
            db = (1/self.num) * np.sum(dz)
            #update the biases and weights
            self.weights = self.weights - self.alpha * dw
            self.bias = self.bias - self.alpha * db
            #every 100 epochs, print out the caluclated cost at this given time
            if i%100 == 0:
                print('{} epochs have passed, cost ={}'.format(i,cost))

    #tests the network given dataset X with labels Y
    def test(self, X, Y):
        z = np.dot(self.weights.T, X) + self.bias
        predict = self.sigmoid(z)
        #form a prediction using the sigmoid output, where >0.5 indicates the image is a cat,
        #and <0.5 indicates a non-cat image
        for i in range(predict.shape[1]):
            if predict[0,i] > 0.5:
                predict[0,i] = 1
            else:
                predict[0,i] = 0
        #check accuracy
        correct = 0
        total = 0
        for i in range(0,X.shape[1]):
            total += 1
            if predict[0,i] == Y[0,i]:
                correct += 1
        #return fraction of images correctly identified
        return correct/total
