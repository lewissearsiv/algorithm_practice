import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
class RandomForestClassifier(object):
    '''Our random forest classifier built on sci-kit learn's decision tree classifier'''
    
    
    def __init__(self, num_bags, bag_size_percentage):
        '''Define the hyperparameter for the number of bags and the size of the bags as a percentage
        of the total data you'll fit the model on. About 63% of the train data is recommended.'''
        self.num_bags = num_bags
        self.bag_percentage = bag_size_percentage
    
    def fit(self, train_data, train_target):
        '''We will train various decision trees. Be sure to one-hot-encode any categorical data.
        Inputs should be the same form as pandas dataframes after sklearn's test_train_split.'''
        
        data = np.array(train_data)
        target = np.array(train_target)
        train_class_counts = np.unique(target, return_counts = True)
        self.train_class_counts = train_class_counts
        
        #train num_bags many decision trees using bootstrapping techniques
        bag_size = int(np.floor(self.bag_percentage * len(data)))
        models = []
        i = 0
        while i < self.num_bags:
            random_sample_index = np.random.choice(data.shape[0], bag_size)
            data_bagged = data[random_sample_index]
            target_bagged = target[random_sample_index]
            decision_tree_model = DecisionTreeClassifier(random_state=0)
            decision_tree_model.fit(data_bagged, target_bagged)
            models.append(decision_tree_model)
            i += 1
        
        self.all_models = models
        return self
        
    def predict(self, data_frame):
        '''Use the set of models created by the bagging to predict a class.
        Should be formatted as data frames. '''
        
        data = np.array(data_frame)
        
        #Create an array of all predictions
        model_predictions = np.array([model.predict(data) for model in self.all_models]).T
        model_counts = [np.unique(votes, return_counts = True) for votes in model_predictions]
        self.model_counts = model_counts
        max_votes = [counts[0][counts[1] == max(counts[1])] for counts in model_counts]
        
        #function to sift through vote counts with some bias:
        #We will always break a tie with the minority class
        def vote_cleaner(votes):
            if len(votes) == 1:
                return votes
            else:
                 return self.train_class_counts[0][np.argmin(self.train_class_counts[1][self.train_class_counts[0] == votes])]
        
        #Final predictions
        self.predictions = [int(vote_cleaner(votes)) for votes in max_votes]        
        return self
    
    def score(self, test_data, test_target):
        predictions = np.array(self.predict(test_data).predictions)
        actual = np.array([int(x) for x in list(test_target.values)])
        self.score_predictions = predictions
        
        #First complete accuracy
        correctly_classified = predictions == actual
        correct_counts = np.unique(correctly_classified, return_counts = True)
        self.accuracy = float(correct_counts[1][correct_counts[0] == True]/len(correctly_classified))
        
        #Breakdown by class
        classes = []
        class_counts_actual = np.unique(actual, return_counts = True)
        for val in class_counts_actual[0]:
            classes.append([val, predictions[actual == val]])
        class_breakdown = [[class_[0],np.unique(class_[1], return_counts = True)] for class_ in classes]
        
        #make a confusion matrix in a dataframe
        class_dict = dict()
        class_columns = []
        for i in range(len(class_breakdown)):
            class_ = class_breakdown[i][0]
            class_columns.append(class_)
            class_dict[class_] = class_breakdown[i][1][1]
        confusion_df = pd.DataFrame(class_dict)
        confusion_df.index.name = "Actual"
        columns_ = ["Predicted: {}".format(class_) for class_ in class_columns]
        confusion_df.columns = columns_
        self.confusion_matrix = confusion_df
        
        #accuracy analysis by class
        line = "----------------------------------------------------"
        output = line
        for ind in confusion_df.index:
            row = confusion_df.iloc[ind]
            accuracy = row["Predicted: {}".format(ind)]/sum(row)
            output += "\nClass {} Accuracy: {}\n".format(ind, accuracy) + line
        self.class_accuracy_analysis = output
    
        
        return self