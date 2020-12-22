import numpy as np
import pandas as pd

def numpy_distance(point, df):
        '''Given a point and a pandas dataframe, numpy_distance computes the euclidean distance '''
        
        try: 
            return np.sqrt(np.sum((np.array(point).astype(float) - np.array(df).astype(float))**2, axis = 1))
        except:
            pass

        #It helps to add some error messages in case something goes wrong
        try:
            if len(point) != len(df.columns):
                return "Error: The dimensions of your point and DataFrame don't match!"
        except:
            pass

        return "User Error: Please review input critera."

class KNNClassifier:
    
    #initialize the hyperparameter k
    def __init__(self, k):
        try:
            if type(k) == int and k >= 1:
                self.k = k
            else:
                raise ValueError('Bad k')
        except ValueError as exp:
            line = "\n---------------------------------------\n"
            print("Value Error:{}Given k = {}. Bad choice my friend!{}k must be a nonzero positive integer.{}\
            ".format(line, k, line, line))
        
             
    #Fit the training data.
    #You should recall that KNN doesn't actually calculate anything to fit. It just creates a copy of the data.
    def fit(self, X_train, y_train):
        '''Makes a copy of training data and the target to train knn'''
        if len(X_train) != len(y_train):
            line = "\n---------------------\n"
            return print("Dimensionality Error:{}Training data and training target dimensions don't match.".format(line))
        
        self.train_data = X_train
        self.train_target = y_train
    
    
    def predict_fast(self, x_test):
        '''Classify unseen data using the k-nearest points in the train data'''
        
        # First, Make a list of distances:
        distances = numpy_distance(x_test, self.train_data)
        if type(distances) == str:
            return "numpy_distance error: {}".format(distances)
        distances_index = distances.argsort()
        
        
        #Now pick the k-closest points:
        k_nearest = [val for i, val in enumerate(list(self.train_target)) if i in distances_index[:self.k]]
        
        #Count the unique values
        counts = np.unique(k_nearest, return_counts=True)
        
        #Find all of the max value classes:
        max_values = counts[0][np.where(counts[1] == max(counts[1]))[0]]
        return np.random.choice(max_values,1)[0]
    
    
    #After fitting the model, we make predictions on unseen test data
    def predict_tie_break(self, x_test):
        '''Classify unseen data using the k-nearest points in the train data'''
        
        # First, Make a list of distances:
        distances = numpy_distance(x_test, self.train_data)
        if type(distances) == str:
            return "numpy_distance error: {}".format(distances)
        distances_index = distances.argsort()
        
        
        #Now pick the k-closest points:
        k_nearest = [val for i, val in enumerate(list(self.train_target)) if i in distances_index[:self.k]]
        
        #Count the unique values
        counts = np.unique(k_nearest, return_counts=True)
        
        #Find all of the max value classes:
        max_values = counts[0][np.where(counts[1] == max(counts[1]))[0]]
        
        if len(max_values) == 1:
            return max_values[0]
        
        #What if we have a tie situation?
        #For this situation, we will iteratively remove a neighbor from consideration until there is a unique max
        new_k = self.k - 1
        while new_k > 0:
            #This is all the same code:
            k_nearest = [val for i, val in enumerate(list(self.train_target)) if i in distances_index[:self.k]]
            counts = np.unique(k_nearest, return_counts=True)
            max_values = counts[0][np.where(counts[1] == max(counts[1]))[0]]
            if len(max_values) == 1:
                return max_values[0] 
    
    #A different tie-breaker
    def predict_imbalanced(self, x_test):
        '''If you are working with imbalanced data and want to give priority to minority class,
        this prediction function always gives any ties to the minority class.'''
        
        # First, Make a list of distances:
        distances = numpy_distance(x_test, self.train_data)
        if type(distances) == str:
            return "numpy_distance error: {}".format(distances)
        distances_index = distances.argsort()
        
        
        #Now pick the k-closest points:
        k_nearest = [val for i, val in enumerate(list(self.train_target)) if i in distances_index[:self.k]]
        
        #Count the unique values
        counts = np.unique(k_nearest, return_counts=True)
        
        #Find all of the max value classes:
        max_values = counts[0][np.where(counts[1] == max(counts[1]))[0]]
        
        if len(max_values) == 1:
            return max_values[0]
        
        #If we have a tie situation, just pick the smallest class
        return max_values[np.array([self.train_target.count(x) for x in max_values]).argmin()]   