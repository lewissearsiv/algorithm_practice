import numpy as np
import pandas as pd

class K_Means_Clustering:
    '''Basic k-means algorithm for cluster analysis. '''
    
    #initialize the hyperparameter k
    def __init__(self, k):
        try:
            if type(k) == int and k >= 1:
                self.k = k
            else:
                raise ValueError('Bad k')
        except ValueError as exp:
            line = "\n-------------------------------------------------\n"
            print("Value Error:{}Given k = {}. Bad choice my friend!{}k must be a nonzero positive integer.{}\
            ".format(line, k, line, line))
    
    #Find the best centroids to create labels
    def fit(self, df, iterations):
        '''Input a scaled np.array with only numerical columns to be assigned labels. For a pandas dataframe,
        just fit StandardScaler and transform the dataframe. Put in the amount of iterations you desire. The
        iterations will stop if the labels do not change.'''
        
        #We will run through this process based on the set ammount of iterations.
        iteration_counter = 0
        while iteration_counter < iterations:
            
            #Initialize centroids
            if iteration_counter == 0:
                #df is scaled so we just want random points normally distibuted around 0
                centroids = np.random.normal(0 , 1, size = (self.k,df.shape[1]))
            #Update centroids
            else:
                centroids = centroids_update


            #Pick the labels
            closest_centroid = []
            for vec in df:
                distances = np.sum((centroids - vec)**2, axis = 1)
                label = np.argmin(distances)
                closest_centroid.append(label)
                
            #Check if any labels have changed
            try:
                if len(np.unique(labels == np.array(closest_centroid))) == 1:
                    self.k_labels = labels
                    line = "\n----------------------------------------------\n"
                    df_count_labels = pd.DataFrame({'Label Counts': 
                                                    np.unique(self.k_labels, return_counts = True)[1]})
                    return print("Convergence Reached. Stopped at iteration {}.{}{}\
                                ".format(iteration_counter, line, df_count_labels))
            except:
                pass
            
            
            labels = np.array(closest_centroid)

            #Now calculate new centroids
            updates = []
            unique_labels = np.unique(labels)
            for some_label in unique_labels:
                some_label_group = df[labels == some_label]
                
                #Find the average
                try: 
                    center = np.sum(some_label_group, axis = 0)/some_label_group.shape[0]
                    
                #This is basically the case where there are no points with this label
                except:
                    center = np.random.normal(0 , 1, size = (5,)) 
                
                updates.append(center)
            centroids_update = np.array(updates)
            iteration_counter += 1

        self.k_labels = labels
        line = "\n-----------------------\n"
        df_count_labels = pd.DataFrame({'Label Counts': np.unique(self.k_labels, return_counts = True)[1]})
        return print("Reached max iterations.{}{}".format(line,df_count_labels))