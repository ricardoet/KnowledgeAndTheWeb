# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 14:48:40 2018

@author: Carlos Ortega
"""
import os
import sys
import time
import json
import pickle
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
max_queried = 10
#### AVAILABLE MODELS ####
class BaseModel(object):

    def __init__(self):
        pass

    def fit_predict(self):
        pass

class LogModel(BaseModel):

    model_type = 'Multinominal Logistic Regression' 
    
    def fit_predict(self, X_train, y_train, X_val, X_labelled_test, c_weight):
        print ('training multinomial logistic regression')
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
        print ('training random forest...')
        self.classifier = RandomForestClassifier(n_estimators=500, class_weight=c_weight)
        self.classifier.fit(X_train, y_train)
        self.test_y_predicted = self.classifier.predict(X_labelled_test)
        self.val_y_predicted = self.classifier.predict(X_val)
        return (X_train, X_val, X_labelled_test, self.val_y_predicted, self.test_y_predicted)

class SvmModel(BaseModel):

    model_type = 'Support Vector Machine with linear Kernel'
    def fit_predict(self, X_train, y_train, X_val, X_labelled_test, c_weight):
        print ('training svm...')
        self.classifier = SVC(C=1, kernel='linear', probability=True,
                              class_weight=c_weight)
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
        print ('Train set:', X_train.shape, 'y:', y_train.shape)
        print ('Val   set:', X_val.shape)
        print ('Test  set:', X_labelled_test.shape)
        t0 = time.time()
        (X_train, X_val, X_labelled_test, self.val_y_predicted,
         self.test_y_predicted) = \
            self.model_object.fit_predict(X_train, y_train, X_val, X_labelled_test, c_weight)
        self.run_time = time.time() - t0
        return (X_train, X_val, X_labelled_test)  # we return them in case we use PCA, with all the other algorithms, this is not needed.

    # we want fscore only for the test set

    def get_test_fscore(self, i, y_test, beta = 1):
        classif_rate = fbeta_score(y_test.ravel(), self.test_y_predicted.ravel(), beta, \
                                   average = 'micro')
        self.fscores.append(classif_rate)               
        print('--------------------------------')
        print('Iteration:',i)
        print('--------------------------------')
        print('y-test set:',y_test.shape)
        print('Example run in %.3f s' % self.run_time,'\n')
        print("F-Score rate for %f with beta %f" % (classif_rate, beta))    
        print("Classification report for classifier %s:\n%s\n" % (self.model_object.classifier, metrics.classification_report(y_test, self.test_y_predicted)))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, self.test_y_predicted))
        print('--------------------------------')
        
# Normalizer
class Normalize(object):
    
    def normalize(self, X_train, X_val, X_labelled_test):
        self.scaler = MinMaxScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_val   = self.scaler.transform(X_val)
        X_labelled_test  = self.scaler.transform(X_labelled_test)
        return (X_train, X_val, X_labelled_test) 
    
    def inverse(self, X_train, X_val, X_labelled_test):
        X_train = self.scaler.inverse_transform(X_train)
        X_val   = self.scaler.inverse_transform(X_val)
        X_labelled_test  = self.scaler.inverse_transform(X_labelled_test)
        return (X_train, X_val, X_labelled_test)

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

# Retriever of the sampling technique
def get_k_random_samples(initial_labeled_samples, X_labelled_train,y_train):
    random_state = check_random_state(0)
    random_state
    permutation = np.random.choice(len(X_labelled_train),
                                   initial_labeled_samples,
                                   replace=False)
    print ()
    print ('initial random chosen samples', permutation.shape),
#            permutation)
    X_train = X_labelled_train[permutation]
    y_train = y_train[permutation]
    X_train = X_train.reshape((X_train.shape[0], -1))
    bin_count = np.bincount(y_train.astype('int64'))
    unique = np.unique(y_train.astype('int64'))
    print (
        'initial train set:',
        X_train.shape,
        y_train.shape,
        'unique(labels):',
        bin_count,
        unique,
        )
    return (permutation, X_train, y_train)

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

        # initialize process by applying base learner to labeled training data set to obtain Classifier

        # (permutation, X_train, y_train) = get_k_random_samples(self.initial_labeled_samples, X_labelled_train, y_train)
        self.queried = self.initial_labeled_samples
        self.samplecount = [self.initial_labeled_samples]

        # permutation, X_train, y_train = get_equally_k_random_samples(self.initial_labeled_samples,classes)

        # assign the test set as part of the train labelled data

        print ('test set:', X_labelled_test.shape, y_test.shape)
        print ()

        # normalize data

        #normalizer = Normalize()
        #X_labelled_train, X_val, X_labelled_test = normalizer.normalize(X_labelled_train, X_val, X_labelled_test)   
        
        self.clf_model = TrainModel(self.model_object)
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
            print ('First ten instances\' probabilities:', probas_val.shape, '\n',
                   np.amax(probas_val, axis=1)[0:10])

            # select samples using a selection function

            uncertain_samples = self.sample_selection_function.select(probas_val, self.initial_labeled_samples)

            # normalization needs to be inversed and recalculated based on the new train and test set.
 
            #  X_labelled_train, X_val, X_labelled_test = normalizer.inverse(X_labelled_train, X_val, X_labelled_test)   

            print ('trainset before', X_labelled_train.shape, y_train.shape)
            X_labelled_train = vstack([X_labelled_train, X_val[uncertain_samples]])
            print('these id are asked by machine, please check them')
            print('--------------------------------')
            oracle_sentence = np.array(X_valref.sentence[uncertain_samples])
            y_aux = []
            print('START QUERYING')
            print()
            for s in range(len(oracle_sentence)):
              print(oracle_sentence[s])
              print('--------------------------------')
              print('please introduce the ground truth labels')
              y_aux.append(input())
            print()
            print('END QUERYING')
            y_aux = np.array(y_aux)
            # TODO: Write a try catch error if the dimensions of y_aux doesn't correspond to X_val
            y_train = np.concatenate((y_train, y_aux))
            print ('trainset after', X_labelled_train.shape, y_train.shape)
            self.samplecount.append(X_labelled_train.shape[0])
            
            labels, counts = np.unique(y_train, return_counts=True)
            print()
            print(
                'updated train set:',
                X_labelled_train.shape,
                y_train.shape)
            print(dict(zip(labels, counts)))
            print ('val set:', X_val.shape)
            X_val = del_rows_sparse(X_val, uncertain_samples)
            print ('val set before next iteration:', X_val.shape)
            print ()

            # normalize again after creating the 'new' train/test sets
            #normalizer = Normalize()
            #X_train, X_val, X_labelled_test = normalizer.normalize(X_train, X_val, X_labelled_test)               

            self.queried += self.initial_labeled_samples
            print(X_labelled_train.shape, y_train.shape, X_val.shape, X_labelled_test.shape)
            (X_labelled_train, X_val, X_labelled_test) = self.clf_model.train(X_labelled_train, y_train, X_val, X_labelled_test, 'balanced')
            self.clf_model.get_test_fscore(active_iteration, y_test)

        print ('final active learning fscores',
               self.clf_model.fscores)

# Trigger Function
def experiment(d, models, selection_functions, Ks, repeats, contfrom, X_labelled_train, \
               X_labelled_test, y_train, y_test, X_val, X_valref):
    print ('stopping at:', max_queried)
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
                if count >= contfrom:
                    print ('Count = %s, using model = %s, selection_function = %s, k = %s, iteration = %s.' % (count, model_object.__name__, selection_function.__name__, k, i))
                    alg = TheAlgorithm(k, 
                                       model_object, 
                                       selection_function
                                       )
                    alg.run(X_labelled_train, X_labelled_test, y_train, y_test, X_val, X_valref)
                    d[model_object.__name__][selection_function.__name__][str(k)].append(alg.clf_model.fscores)
                    if count % 5 == 0:
                        print(json.dumps(d, indent=2, sort_keys=True))
                    print ()
                    print ('---------------------------- FINISHED ---------------------------')
                    print ()
    return d