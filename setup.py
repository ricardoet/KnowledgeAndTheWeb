# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 03:28:06 2018
* Code from https://github.com/orico/ActiveLearningFrameworkTutorial/blob/master/Active_Learning_Tutorial.ipynb
1 ca
2 cs
3 na
4 us

@author: Carlos Ortega
"""
###############################
#### LIBRARIES & DIRECTORY ####
###############################

# Install and Import requirements, Also set the directory
import os
os.chdir("C:/Users/orteg/Dropbox/ArchOrg/1Almacen/EMDS/Computacion/3 Cursos/KUL Knowledge and the Web/Project")
import pickle

# from pylab import rcParams
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    
# Third-party modules
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.externals import joblib
from sklearn.utils import check_random_state
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stopwords = stopwords.words('spanish')

# Local modules
from helper import FeatureSelector, ArrayCaster
from active_learner import BaseModel, LogModel, RfModel, SvmModel, TrainModel, \
BaseSelectionFunction, EntropySelection, get_k_random_samples, \
TheAlgorithm, experiment

#### LOADING DATASET & DATA SPLITTING ####

filename_xval = "x_unlabelled.xlsx"
filename_xltrain = "x_labelled_train.xlsx"
filename_xltest = "x_labelled_test.xlsx"

filename_ytest = "y_labelled_test.xlsx"
filename_yltrain = "y_labelled_train.xlsx"

x_unlabelled = pd.read_excel(filename_xval)
x_labelled_train = pd.read_excel(filename_xltrain)
x_labelled_test = pd.read_excel(filename_xltest)

y_labelled_train = pd.read_excel(filename_yltrain)
y_labelled_test = pd.read_excel(filename_ytest)

# The following code of Active Learning requires Numpy Arrays instead of Pandas Dataframes
xarray_labelled_train = np.array(x_labelled_train).ravel()
xarray_labelled_test = np.array(x_labelled_test).ravel()
xarray_unlabelled = np.array(x_unlabelled).ravel()

# Transforming number labels to string ones
conditions = [
    y_labelled_test['label'] == 1,
    y_labelled_test['label'] == 2,
    y_labelled_test['label'] == 3,
    y_labelled_test['label'] == 4
    ]
conditions2 = [
    y_labelled_train['label'] == 1,
    y_labelled_train['label'] == 2,
    y_labelled_train['label'] == 3,
    y_labelled_train['label'] == 4
    ]
choices = ['ca', 'cs', 'na','us']
np.select(conditions, choices)

y_test = np.array(np.select(conditions, choices))
y_train = np.array(np.select(conditions2, choices))

############################
####  FEATURE PIPELINE   ###
############################

# UNION OF ALL FEATURES EXTRACTED FROM SENTENCE
features = FeatureUnion([
    ("words", Pipeline([("cont", FeatureSelector(key='sentence')),
                        ('vect', CountVectorizer(ngram_range=(1,1),
                                                 max_df=0.5,
                                                 min_df=1,
                                                 stop_words= stopwords)),
                        ('tfidf', TfidfTransformer(use_idf=True,
                                                   sublinear_tf=False))])),
    ("lempos", Pipeline([("cont", FeatureSelector(key='LEMMA_POS')),
                         ('vect', CountVectorizer(ngram_range=(1,2),
                                                  max_df=0.4,
                                                  lowercase=False,
                                                  min_df=2)),
                         ('tfidf', TfidfTransformer(use_idf=True,
                                                    sublinear_tf=False))])),
    ("possen", Pipeline([("cont", FeatureSelector(key='POS_SENT')),
                         ('vect', CountVectorizer(ngram_range=(1,2),
                                                  max_df=0.6,
                                                  lowercase=False,
                                                  min_df=3)),
                         ('tfidf', TfidfTransformer(use_idf=True,
                                                    sublinear_tf=False))])),
    ("length", Pipeline([("cont", FeatureSelector(key='LENGTH')),
                         ('caster', ArrayCaster())])),
  ])

features.fit(pd.concat([x_unlabelled,x_labelled_train],ignore_index=True))
xl_train_trans = features.transform(x_labelled_train)
xu_train_trans = features.transform(x_unlabelled)
xl_test_trans = features.transform(x_labelled_test)

#### EXPERIMENT SETTINGS ###
repeats = 1
models = [SvmModel]
selection_functions = [EntropySelection]
Ks = [5] 
d = {}
stopped_at = -1

d, model_trained = experiment(d, models, selection_functions, Ks, repeats, stopped_at+1, \
               xl_train_trans, xl_test_trans, y_train, y_test, xu_train_trans, x_unlabelled)
print (d)
results = json.loads(json.dumps(d, indent=2, sort_keys=True))
print(results)

# Save model
filename_trained_model = 'trained_model.sav'
pickle.dump(model_trained, open(filename_trained_model, 'wb'))


# load the model from disk
loaded_model = pickle.load(open(filename_trained_model, 'rb'))


# alg1 = TheAlgorithm(5,SvmModel,EntropySelection)
# alg1.run(xl_train_trans, xl_test_trans, y_train, y_test, xu_train_trans, x_unlabelled)
# TODO - TRAIN MODEL AND APPLY 10 FOLD CROSSVALIDATION FOR EVALUATION
#feature_pipe_SVM.fit(x_labelled_train,y_labelled_train)
#predicted = cross_val_predict(feature_pipe_SVM,
#                              x_labelled_train,
#                              y_labelled_train,
#                              cv=10)
