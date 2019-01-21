# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 14:48:40 2018

@author: Carlos Ortega
"""
# Built in Modules
import os
import sys
import time
import json

# Third Party Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy import stats
from pylab import rcParams
from sklearn.utils import check_random_state
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import fbeta_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from scipy.sparse import vstack, csr_matrix

np.set_printoptions(threshold=sys.maxsize)
max_queried = 100

#### AVAILABLE MODELS ####
class BaseModel(object):

    def __init__(self):
        pass

    def fit_predict(self):
        pass

class LogModel(BaseModel):

    model_type = 'Multinominal Logistic Regression' 
    
    def fit_predict(self, X_train, y_train, X_val, X_labelled_test, c_weight):
        print ('Training Multinomial Logistic Regression')
        train_samples = X_train.shape[0]
        self.classifier = LogisticRegression(
            C=50. / train_samples,
            multi_class='multinomial',
            penalty='l1',
            solver='saga',
            tol=0.1,
            class_weight=c_weight,
            )
        self.classifier.fit(X_train, y_train)
        self.test_y_predicted = self.classifier.predict(X_labelled_test)
        self.val_y_predicted = self.classifier.predict(X_val)
        return (X_train, X_val, X_labelled_test, self.val_y_predicted, self.test_y_predicted)

class RfModel(BaseModel):

    model_type = 'Random Forest'
    
    def fit_predict(self, X_train, y_train, X_val, X_labelled_test, c_weight):
        print ('Training Random Forest...')
        self.classifier = RandomForestClassifier(n_estimators=500, class_weight=c_weight)
        self.classifier.fit(X_train, y_train)
        self.test_y_predicted = self.classifier.predict(X_labelled_test)
        self.val_y_predicted = self.classifier.predict(X_val)
        return (X_train, X_val, X_labelled_test, self.val_y_predicted, self.test_y_predicted)

class SvmModel(BaseModel):

    model_type = 'Support Vector Machine with linear Kernel'
    def fit_predict(self, X_train, y_train, X_val, X_labelled_test, c_weight):
        print ('Training SVM...')
        self.classifier = SVC(C=1, kernel='linear', probability=True, class_weight=c_weight)
        self.classifier.fit(X_train, y_train)
        self.test_y_predicted = self.classifier.predict(X_labelled_test)
        self.val_y_predicted = self.classifier.predict(X_val)
        return (X_train, X_val, X_labelled_test, self.val_y_predicted,
                self.test_y_predicted)
 
### TRAINING ####     
class TrainModel:

    def __init__(self, model_object):        
        self.fscores = []
        self.model_object = model_object()        

    def print_model_type(self):
        print (self.model_object.model_type)

    # we train normally and get probabilities for the validation set. i.e., we use the probabilities to select the most uncertain samples

    def train(self, X_train, y_train, X_val, X_labelled_test, c_weight):
        print ('Train set:', X_train.shape)
        print ('Val   set:', X_val.shape)
        print ('Test  set:', X_labelled_test.shape)
        t0 = time.time()
        (X_train, X_val, X_labelled_test, self.val_y_predicted,
         self.test_y_predicted) = \
            self.model_object.fit_predict(X_train, y_train, X_val, X_labelled_test, c_weight)
        self.run_time = time.time() - t0
        return (X_train, X_val, X_labelled_test)  # we return them in case we use PCA, with all the other algorithms, this is not needed.

    # we want fscore only for the test set
    def get_test_fscore(self, i, y_test, beta = 0.5):
        classif_rate = fbeta_score(y_test.ravel(), self.test_y_predicted.ravel(), beta, pos_label='cs', average ='binary')
        self.fscores.append(classif_rate)               
        print('--------------------------------')
        print('Iteration:',i)
        print('--------------------------------')
        print('Example run in %.3f s' % self.run_time,'\n')
        print("F-Score rate for %.3f with beta %.2f" % (classif_rate, beta))    
        print("Classification report for classifier %s:\n%s\n" % (self.model_object.classifier, metrics.classification_report(y_test, self.test_y_predicted)))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, self.test_y_predicted))
        print('--------------------------------')   

#### SAMPLING ACTIVE LEARNING METHODS ####
class BaseSelectionFunction(object):

    def __init__(self):
        pass

    def select(self):
        pass

class RandomSelection(BaseSelectionFunction):
    @staticmethod
    def select(probas_val, initial_labeled_samples):
        random_state = check_random_state(0)
        random_state
        selection = np.random.choice(probas_val.shape[0], initial_labeled_samples, replace=False)
#     print('uniques chosen:',np.unique(selection).shape[0],'<= should be equal to:',initial_labeled_samples)
        return selection

class EntropySelection(BaseSelectionFunction):
    @staticmethod
    def select(probas_val, initial_labeled_samples):
        e = (-probas_val * np.log2(probas_val)).sum(axis=1)
        selection = (np.argsort(e)[::-1])[:initial_labeled_samples]
        return selection   
      
class MarginSamplingSelection(BaseSelectionFunction):
    @staticmethod
    def select(probas_val, initial_labeled_samples):
        rev = np.sort(probas_val, axis=1)[:, ::-1]
        values = rev[:, 0] - rev[:, 1]
        selection = np.argsort(values)[:initial_labeled_samples]
        return selection

# Row Deletion for Sparse Matrix
def del_rows_sparse(mat, indices):
    """
    Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
    """
    if not isinstance(mat, csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    indices = list(indices)
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[indices] = False
    return mat[mask]

# The Wrapper function that will set up the different models and settings the user chose    
class TheAlgorithm(object):

    fscores = []

    def __init__(self, initial_labeled_samples, model_object, selection_function):
        self.initial_labeled_samples = initial_labeled_samples
        self.model_object = model_object
        self.sample_selection_function = selection_function

    def run(self, X_labelled_train, X_labelled_test, y_train, y_test, X_val, X_valref):
        y_train = np.ravel(y_train)
        X_valref = np.array(X_valref.sentence)

        # initialize process by applying base learner to labeled training data set to obtain Classifier

        # (permutation, X_train, y_train) = get_k_random_samples(self.initial_labeled_samples, X_labelled_train, y_train)
        self.queried = self.initial_labeled_samples
        self.samplecount = [self.initial_labeled_samples]

        # permutation, X_train, y_train = get_equally_k_random_samples(self.initial_labeled_samples,classes)

        # assign the test set as part of the train labelled data
        print ()
        print ('Test set:', X_labelled_test.shape)
        # TODO - Implement some Handling Exceptions that prevents to use not fitting matrices

        self.clf_model = TrainModel(self.model_object)
        # TODO - TRAIN MODEL AND APPLY 10 FOLD CROSSVALIDATION FOR EVALUATION, SO NOT FOR PREDICTION AS FACTRANK DID

        (X_labelled_train, X_val, X_labelled_test) = self.clf_model.train(X_labelled_train, y_train, X_val, X_labelled_test, 'balanced')
        active_iteration = 1
        self.clf_model.get_test_fscore(1, y_test)

        while self.queried <= max_queried:

            active_iteration += 1

            # get validation probabilities
            probas_val = self.clf_model.model_object.classifier.predict_proba(X_val)
            print ('val predicted:',
                   self.clf_model.val_y_predicted.shape,
                   self.clf_model.val_y_predicted[0:10])
            print(probas_val[0:10])
            print()
            print ('First ten instances\' probabilities:', probas_val.shape, '\n',
                   np.amax(probas_val, axis=1)[0:10])
            print()

            # select samples using a selection function

            uncertain_samples = self.sample_selection_function.select(probas_val, self.initial_labeled_samples)
            print ('Training Set before Querying: %s where y: %s' % (X_labelled_train.shape, y_train.shape))
            X_labelled_train = vstack([X_labelled_train, X_val[uncertain_samples]])
            print('--------------------------------')
            oracle_sentence = np.array(X_valref[uncertain_samples])
            y_aux = []
            print('START QUERYING')
            print('--------------------------------')
            print()
            for s in range(len(oracle_sentence)):
              print(oracle_sentence[s])
              print()
              print('--INTRODUCE GROUND TRUTH LABELS---')
              y_aux.append(input())
              print('Answered Question %s out of %s' % (s+1,len(oracle_sentence)))
              if (s < len(oracle_sentence)-1):
                print('----------NEXT QUESTION---------')
              
            print()
            print('--------------------------------')
            print('END QUERYING')
            print('--------------------------------')
            # TODO: Write a try catch error if the dimensions of y_aux doesn't correspond to X_val
            y_aux = np.array(y_aux)
            y_train = np.concatenate((y_train, y_aux))
            print ('Training Set after Querying: %s where y: %s' % (X_labelled_train.shape, y_train.shape))
            print()
            self.samplecount.append(X_labelled_train.shape[0])
            labels, counts = np.unique(y_train, return_counts=True)
            print(dict(zip(labels, counts)))
            print('Counts per Label in Traning Set after Querying')
            print()
            print('Unlabelled Pool before Querying: %s' % (X_val.shape,))
            X_val = del_rows_sparse(X_val, uncertain_samples)
            X_valref = np.delete(X_valref, uncertain_samples, 0)
            print('Unlabelled Pool after Querying: %s' % (X_val.shape,))
            print()
            print ('Queried Instances so far: %d' % self.queried)
            self.queried += self.initial_labeled_samples
            print()
            print('New Train Iteration')
            (X_labelled_train, X_val, X_labelled_test) = self.clf_model.train(X_labelled_train, y_train, X_val, X_labelled_test, 'balanced')
            # Get the F-score 
            self.clf_model.get_test_fscore(active_iteration, y_test)

        print ('Final F-scores', self.clf_model.fscores)
        return self.clf_model

# Trigger Function
def experiment(d, models, selection_functions, Ks, repeats, contfrom, X_labelled_train, \
               X_labelled_test, y_train, y_test, X_val, X_valref):
    print ('stopping at:', max_queried)
    models_trained = [None]*len(Ks)*len(selection_functions)*len(models)*repeats
    count = 0
    for model_object in models:
      if model_object.__name__ not in d:
          d[model_object.__name__] = {}
      
      for selection_function in selection_functions:
        if selection_function.__name__ not in d[model_object.__name__]:
            d[model_object.__name__][selection_function.__name__] = {}
        
        for k in Ks:
            d[model_object.__name__][selection_function.__name__][str(k)] = []           
            
            for i in range(0, repeats):
                count+=1
                count2 = count - 1
                if count >= contfrom:
                    print ('Count = %s, using model = %s, selection_function = %s, k = %s, iteration = %s.' % (count, model_object.__name__, selection_function.__name__, k, i))
                    alg = TheAlgorithm(k, 
                                       model_object, 
                                       selection_function
                                       )
                    
                    models_trained[count2] = alg.run(X_labelled_train, X_labelled_test, y_train, y_test, X_val, X_valref)
                    models_trained[count2] = models_trained[count2].model_object.classifier
                    d[model_object.__name__][selection_function.__name__][str(k)].append(alg.clf_model.fscores)
                    if count % 5 == 0:
                        print(json.dumps(d, indent=2, sort_keys=True))
                    print ()
                    print ('---------------------------- FINISHED ---------------------------')
                    print ()
    return d, models_trained