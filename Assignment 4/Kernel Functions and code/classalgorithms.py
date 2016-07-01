from __future__ import division  # floating point division
import numpy as np
import utilities as utils

class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """
    
    def __init__( self, params=None ):
        """ Params can contain any useful parameters for the algorithm """
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        
    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest

class LinearRegressionClass(Classifier):
    """
    Linear Regression with ridge regularization
    Simply solves (X.T X/t + lambda eye)^{-1} X.T y/t
    """
    def __init__( self, params=None ):
        self.weights = None
        if params is not None and 'regwgt' in params:
            self.regwgt = params['regwgt']
        else:
            self.regwgt = 0.01
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Ensure ytrain is {-1,1}
        yt = np.copy(ytrain)
        yt[yt == 0] = -1
        
        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = Xtrain.shape[0]
        self.weights = np.dot(np.dot(np.linalg.inv(np.add(np.dot(Xtrain.T,Xtrain)/numsamples,self.regwgt*np.identity(Xtrain.shape[1]))), Xtrain.T),yt)/numsamples
        
    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1     
        ytest[ytest < 0] = 0    
        return ytest

class NaiveBayes(Classifier):
    """ Gaussian naive Bayes; need to complete the inherited learn and predict functions """
    
    def __init__( self, params=None ):
        """ Params can contain any useful parameters for the algorithm """
        self.usecolumnones = True
        if params is not None:
            self.usecolumnones = params['usecolumnones']
            
    
class LogitReg(Classifier):
    """ Logistic regression; need to complete the inherited learn and predict functions """

    def __init__( self, params=None ):
        # self.weights = np.ones(9)
        self.step_size=2
        
    def learn(self, Xtrain, ytrain):
        #ll=loglikihood(Xtrain,ytrain,beta)
        #step=1
        #eps=tolerance
        self.weights = np.ones(Xtrain.shape[1])
        p = utils.sigmoid(np.dot(Xtrain, self.weights))      #(500*9) * (9*1) = 500*1
     
        
        #for i in range(20):
        #    p = logsig(X * w)
        #    w_old=w/np.sum(abs(w))
        #    P =
        #x = np.multiply(P,(1-p))
        #x = np.dot(Xtrain.T,x)
        #self.weights = self.weights + np.dot(np.dot(np.linalg.inv(np.dot(x,Xtrain),Xtrain.T),(ytrain-p)))
        #    w = w + 0.1*inv(X' * diag(p .* (1 - p)) * X) * X' * (y - p)
        #    eps = sum(abs(w_old - w / sum(abs(w))))
        #    plot_logreg_figure(X0, X1, X, w)
        #    ll = get_log_likelihood(X, y, w)
        #    step = step + 1
        #

        # w = step_size * X.T * (y-p) 
        #print(p)
        for i in range(500):
            p = utils.sigmoid(np.dot(Xtrain, self.weights))  #(500*9) * (9*1) = 500*1
            self.weights = self.weights + self.step_size * np.dot(Xtrain.T,(ytrain-p)) #(500*9) * (500*1) = (9*1)

    def predict(self, Xtest):
        # print self.weights
        ytest = utils.sigmoid(np.dot(Xtest, self.weights))
        ytest[ytest >= 0.5] = 1
        ytest[ytest < 0.5] = 0
        return ytest

class ClassifierLogitReg(Classifier):
    """ Logistic regression; need to complete the inherited learn and predict functions """

    def __init__( self, params=None ):
        self.weights = np.ones(9)
        self.step_size=0.05

    def learn(self, Xtrain, ytrain):
        xvec = np.dot(Xtrain, self.weights)
        p = utils.sigmoid(np.dot(Xtrain, self.weights))  #(500*9) * (9*1) = 500*1
        P = np.diagflat(p)

        for j in range(500):
            for i in range(Xtrain.shape[0]):
                xvec = np.dot(Xtrain[i], self.weights)  #(1*9) * (9*1) = 500*1
                delta = np.divide((2*ytrain[i]-1)*np.sqrt(np.square(xvec)+1)-xvec,np.square(xvec)+1)
                delta = np.dot(Xtrain[i].T,delta)
                first_term = np.divide((2*ytrain[i]-1)*xvec - np.sqrt(np.square(xvec)+1)-xvec,np.power(np.square(xvec)+1,3/2))
                second_term = 2*xvec*np.divide((2*ytrain[i]-1)*np.sqrt(np.square(xvec)+1)-xvec,np.square(np.square(xvec)+1))
                hessian = np.dot(Xtrain[i].T,Xtrain[i])*(first_term-second_term)
                self.weights = self.weights + self.step_size * delta/hessian #(500*9) * (500*1) = (9*1)
                #print self.weights


    def predict(self, Xtest):
        # print self.weights
        ytest = utils.sigmoid(np.dot(Xtest, self.weights))
        ytest[ytest >= 0.5] = 1     
        ytest[ytest < 0.5] = 0    
        return ytest


class NeuralNet(Classifier):
    """ Two-layer neural network; need to complete the inherited learn and predict functions """
    
    def __init__(self, params=None):
        # Number of input, hidden, and output nodes
        # Hard-coding sigmoid transfer for this implementation for simplicity
        self.ni = params['ni']
        self.nh = params['nh']
        self.no = params['no']
        self.transfer = utils.sigmoid
        self.dtransfer = utils.dsigmoid

        # Set step-size
        self.stepsize = 0.01

        # Number of repetitions over the dataset
        self.reps = 5
        
        # Create random {0,1} weights to define features
        self.wi = np.random.randint(2, size=(self.nh, self.ni))
        self.wo = np.random.randint(2, size=(self.no, self.nh))

    def learn(self, Xtrain, ytrain):
        """ Incrementally update neural network using stochastic gradient descent """        
        for reps in range(self.reps):
            for samp in range(Xtrain.shape[0]):
                self.update(Xtrain[samp,:],ytrain[samp])
            
    # Need to implement predict function, since currently inherits the default

    def evaluate(self, inputs,outputs):
        """ Including this function to show how predictions are made """
        if inputs.shape[0] != self.ni:
            raise ValueError('NeuralNet:evaluate -> Wrong number of inputs')
        
        # hidden activations
        ah = np.ones(self.nh)
        ah = self.transfer(np.dot(self.wi,inputs))  

        # output activations
        ao = np.ones(self.no)
        ao = self.transfer(np.dot(self.wo,ah))
        
        return (ah, ao)

    def update(self, inp, out):
        """ This function needs to be implemented """    
        (ah,ao)=self.evaluate(inp,out)
        delta= (-(out/ao)+ ((1-out)/(1-ao)))* ao*(1-ao)
        deriv1=delta*ah.T
        # deriv2=delta*()

    
