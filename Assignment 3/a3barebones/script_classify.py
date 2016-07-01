from __future__ import division  # floating point division
import csv
import random
import math
import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold

import classalgorithms as algs
 
def splitdataset(dataset, trainsize=500, testsize=300, testfile=None):
    randindices = np.random.randint(0,dataset.shape[0],trainsize+testsize)
    numinputs = dataset.shape[1]-1
    Xtrain = dataset[randindices[0:trainsize],0:numinputs]
    ytrain = dataset[randindices[0:trainsize],numinputs]
    Xtest = dataset[randindices[trainsize:trainsize+testsize],0:numinputs]
    ytest = dataset[randindices[trainsize:trainsize+testsize],numinputs]

    if testfile is not None:
        testdataset = loadcsv(testfile)
        Xtest = dataset[:,0:numinputs]
        ytest = dataset[:,numinputs]        
        
    # Add a column of ones; done after to avoid modifying entire dataset
    Xtrain = np.hstack((Xtrain, np.ones((Xtrain.shape[0],1))))
    Xtest = np.hstack((Xtest, np.ones((Xtest.shape[0],1))))
                              
    return ((Xtrain,ytrain), (Xtest,ytest))
 
 
def getaccuracy(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    return (correct/float(len(ytest))) * 100.0

def loadsusy():
    dataset = np.genfromtxt('susysubset.csv', delimiter=',')
    trainset, testset = splitdataset(dataset)    
    return trainset,testset

def loadmadelon():
    datasettrain = np.genfromtxt('madelon/madelon_train.data', delimiter=' ')
    trainlab = np.genfromtxt('madelon/madelon_train.labels', delimiter=' ')
    trainlab[trainlab==-1] = 0
    trainsetx = np.hstack((datasettrain, np.ones((datasettrain.shape[0],1))))
    trainset = (trainsetx, trainlab)
    
    datasettest = np.genfromtxt('madelon/madelon_valid.data', delimiter=' ')
    testlab = np.genfromtxt('madelon/madelon_valid.labels', delimiter=' ')
    testlab[testlab==-1] = 0
    testsetx = np.hstack((datasettest, np.ones((datasettest.shape[0],1))))
    testset = (testsetx, testlab)
      
    return trainset,testset

if __name__ == '__main__':
    dataset = np.genfromtxt('susysubset.csv', delimiter=',')

    rand_acc = []
    lin_acc = []
    logis_acc = []
    naive_acc = []

    data = np.array(dataset)
    kf = KFold(data.shape[0], n_folds=10)
    mylist = list(kf)
    mylist = np.array(mylist)

    for i in range(mylist.shape[0]):
        train, test = mylist[i]
        #numinputs = data.shape[1]-1
        #offset = 0
        Xtrain = data[train,0:-1]
        #Xtrain = Xtrain[1,:,:]
        ytrain = data[train,-1]
        #ytrain = ytrain.ravel()
        Xtest = data[test,0:-1]
        ytest = data[test,-1]
        #ytest = ytest.ravel()

#    trainset, testset = splitdataset(dataset)
#    print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), trainset[0].shape[0], testset[0].shape[0])
#    nnparams = {'ni': trainset[0].shape[1], 'nh': 32, 'no': 1}
    #Add comments here for testing purpose.

        classalgs = {'Random': algs.Classifier(),
                    'Linear Regression': algs.LinearRegressionClass(),
                    'Naive Bayes': algs.NaiveBayes({'usecolumnones': False}),
                   'Logistic Regression': algs.LogitReg()
                    }

        for learnername, learner in classalgs.iteritems():
            print 'Running learner = ' + learnername
            # Train model
            learner.learn(Xtrain, ytrain)
            # Test model
            #print ytest
            predictions = learner.predict(Xtest)
            #print
            accuracy = getaccuracy(ytest, predictions)
            if learnername == 'Random' :
                print accuracy
                rand_acc.append(accuracy)
            elif learnername == 'Linear Regression' :
                print accuracy
                lin_acc.append(accuracy)
            elif learnername == 'Naive Bayes' :
                print accuracy
                naive_acc.append(accuracy)
            else :
                logis_acc.append(accuracy)
                print accuracy
                #print 'Accuracy for ' + learnername + ': ' + str(accuracy)
    print rand_acc, np.mean(rand_acc)
    print lin_acc, np.mean(lin_acc)
    print naive_acc, np.mean(naive_acc)
