import numpy as np
import pandas as pd 

class NaiveBayesClassifier(object):
    '''Naive Bayes algorithm for classifying discrete targets. Be sure to one-hot-encode 
    any categorical variables. The default classification will be '''
    
    def fit(self, train_data, train_target):
        '''Calculates summary statistics for the training data by class.'''
        #Make data np arrays
        X = np.array(train_data)
        y = np.array(train_target)
        self.data = X
        self.target = y
        self.class_counts = np.unique(y, return_counts = True)
        
        #Dictionary to separate data already converted to numpy arrays
        class_dict = dict()
        for class_ in self.class_counts[0]:
            class_dict[class_] = X[y == class_]
            
        #Get summary statistics for every class
        class_summary = dict()
        for class_ in class_dict.keys():
            class_summary[class_] = [[np.mean(class_dict[class_][:, i]), 
                                      np.std(class_dict[class_][:, i])] for i in 
                                     range(class_dict[class_].shape[1])]
            
        
        self.class_statistics = class_summary 
        return self

    #Now take the summary statistics to predict unseen data using gaussian distributions
    def predict(self, test_data):
        '''Now take the summary statistics from the test data to predict unseen data using 
        gaussian distributions.'''
        
        #Function to calculate the gaussian probability 
        def gaussian_probability(x, mean, std):
            z_score = (x-mean)/std
            return np.exp(-0.5*(z_score**2)) * (1/(std*np.sqrt(2*np.pi)))
        
        class_list = []
        probability_list = []
        for key in self.class_statistics.keys():
            class_list.append(key)
            statistics = self.class_statistics[key]
            
            #Add the total probability of the class first
            probability = len(self.target[self.target == key])/len(self.target)
            for i in range(len(statistics)):
                probability_i = gaussian_probability(test_data[i], 
                                                    self.class_statistics[key][i][0],
                                                    self.class_statistics[key][i][1]) 
                probability *= probability_i
            probability_list.append(probability)
        
        self.predict_proba = [[class_list[i], probability_list[i]] for i in range(len(class_list))]
        self.prediction = class_list[[i for i, val in enumerate(probability_list) if 
                                      val == max(probability_list)][0]]
        return self  