import csv
import random
import numpy as np

import algorithms as algs
import utilities as utils
 
def splitdataset(dataset, trainsize=300, testsize=100):
    # Now randomly split into train and test    
    randindices = np.random.randint(0,dataset.shape[0],trainsize+testsize)
    numinputs = dataset.shape[1]-1
    offset = 50 # Ignore the first 50 features
    Xtrain = dataset[randindices[0:trainsize],offset:numinputs]
    ytrain = dataset[randindices[0:trainsize],numinputs]
    Xtest = dataset[randindices[trainsize:trainsize+testsize],offset:numinputs]
    ytest = dataset[randindices[trainsize:trainsize+testsize],numinputs]

    # Add a column of ones; done after to avoid modifying entire dataset
    Xtrain = np.hstack((Xtrain, np.ones((Xtrain.shape[0],1))))
    Xtest = np.hstack((Xtest, np.ones((Xtest.shape[0],1))))
                              
    return ((Xtrain,ytrain), (Xtest,ytest))
 
 
def geterror(predictions, ytest):
    # Can change this to other error values
    return utils.l2err_squared(predictions,ytest)/ytest.shape[0]
 
if __name__ == '__main__':
    filename = 'blogData_train_small.csv'
    dataset = utils.loadcsv(filename)   
    trainset, testset = splitdataset(dataset)
    print('Split {0} rows into train={1} and test={2} rows').format(
        len(dataset), trainset[0].shape[0], testset[0].shape[0])
    classalgs = {'Random': algs.Regressor(),
                 'Mean': algs.MeanPredictor(),
                 'FSLinearRegression': algs.FSLinearRegression()
                 }

    # Runs all the algorithms on the data and print out results    
    for learnername, learner in classalgs.iteritems():
        print 'Running learner = ' + learnername
        # Train model
        learner.learn(trainset[0], trainset[1])
        # Test model
        predictions = learner.predict(testset[0])
        #print predictions
        accuracy = geterror(testset[1], predictions)
        print 'Accuracy for ' + learnername + ': ' + str(accuracy)
 
