import numpy as np
import utilities as utils

class Regressor:
    """
    Generic regression interface; returns random regressor
    Random regressor randomly selects value between max and min in training set.
    """
    
    def __init__( self, params=None ):
        """ Params can contain any useful parameters for the algorithm """
        self.min = 0
        self.max = 1
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.min = np.amin(ytrain)
        self.max = np.amax(ytrain)

    def predict(self, Xtest):
        ytest = np.random.rand(Xtest.shape[0])*(self.max-self.min) + self.min
        return ytest

class MeanPredictor(Regressor):
    """
    Returns the average target value observed; a reasonable baseline
    """
    def __init__( self, params=None ):
        self.mean = None
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.mean = np.mean(ytrain)
        
    def predict(self, Xtest):
        return np.ones((Xtest.shape[0],))*self.mean

class FSLinearRegression(Regressor):
    """
    Linear Regression with feature selection
    """
    def __init__( self, params=None ):
        self.weights = None
        self.features =range(20)# [1,2,3]   
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.features =range(231)
        Xless = Xtrain[:,self.features]
        lamb=0.001
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xless.T,Xless)), Xless.T+lamb*np.i[231]),ytrain)
        
    def predict(self, Xtest):
        Xless = Xtest[:,self.features]        
        ytest = np.dot(Xless, self.weights)       
        return ytest

    
    
