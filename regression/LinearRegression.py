import numpy as np
import pandas as pd

class LinearRegression():
    
    def __init__(self, learning_rate):
        '''Initializes the class with a learning rate for the optimization of weights.'''
        self.learning_rate = learning_rate
          
    def fit(self, train_data, train_target):
        '''Input the training data and its respective target values'''
        
        #Convert data to numpy arrays
        X = np.concatenate((np.array([np.zeros(len(train_data))+1]).T, train_data), axis = 1)
        y = np.array(train_target)
        
        #initialize coefficients
        coefficients = np.random.normal(0, 1, X.shape[1])
        self.coefficients = coefficients
        
        #Keep a list of SSE and MSE for each iteration if you desire to analyze
        
        self.SSE_list = []
        self.MSE_list = []
            
        
        #iteratively improve coefficients:
        i = 0
        while i < 1000:
            predict = np.matmul(X, self.coefficients)
            SSE = np.linalg.norm(y-predict)
            MSE = SSE/X.shape[0]
            self.SSE_list.append(SSE)
            self.MSE_list.append(MSE)
            
            #Calculate the gradient and apply stochastic gradient descent
            gradient = -(1/SSE) * np.matmul((y-predict).T, X)+np.random.normal(0, 1, self.coefficients.shape)
            self.coefficients = self.coefficients - (1/(i+1))*self.learning_rate * gradient
            i += 1
        SStotal = np.sum((y - np.mean(y))**2)
        self.intercept = self.coefficients[0]
        self.weights = self.coefficients[1:]
        self.error_analysis = pd.DataFrame({'Sum of Squared Errors': [self.SSE_list[-1]],
                                           'Mean Squared Error': [self.MSE_list[-1]],
                                           'Root Mean Squared Error': [np.sqrt(self.MSE_list[-1])],
                                           'R-squared': [1-self.SSE_list[-1]/SStotal]})
        return self