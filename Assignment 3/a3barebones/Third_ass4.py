#import csv
#import random
#import math
import numpy as np
import scipy
from scipy.stats import ttest_ind

import classalgorithms as algs

import pandas as pd

from sklearn.cross_validation import KFold

 
def getaccuracy(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    return (correct/float(len(ytest))) * 100.0
 

if __name__ == '__main__':
    filename = 'susysubset.csv'
    #filename = 'mutants_normalized.csv'
    data = pd.read_csv(filename)
    rand_acc = []
    lin_acc = []
    logis_acc = []
    naive_acc = []
    
    data = np.array(data)
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
#                    'Naive Bayes': algs.NaiveBayes({'usecolumnones': False})
                    'Logistic Regression': algs.LogitReg()
                    }
    
        for learnername, learner in classalgs.iteritems():
            #print 'Running learner = ' + learnername
            # Train model
            learner.learn(Xtrain, ytrain)
            # Test model
            #print ytest
            predictions = learner.predict(Xtest)
            #print 
            accuracy = getaccuracy(ytest, predictions)
            if learnername == 'Random' :
                rand_acc.append(accuracy)
            elif learnername == 'Linear Regression' :
                lin_acc.append(accuracy)
            elif learnername == 'Naive Bayes' :
                naive_acc.append(accuracy)
            else :
                logis_acc.append(accuracy)
            print 'Accuracy for ' + learnername + ': ' + str(accuracy)
    print rand_acc, np.mean(rand_acc)
    print lin_acc, np.mean(lin_acc)
    #print naive_acc, np.mean(naive_acc)
    
    statistic, pvalue = ttest_ind(rand_acc, lin_acc, equal_var=False)
    print "statistic = %.5f and pvalue = %.10f" % (statistic, pvalue)
    statistic, pvalue = ttest_ind(rand_acc, lin_acc, equal_var=True)
    print "statistic = %.5f and pvalue = %.10f" % (statistic, pvalue)

