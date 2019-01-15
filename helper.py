# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 12:05:10 2018
@author: FactRank authors (Rafael Hautekiet) with modifications by Carlos Ortega


"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from extractor import Extractor

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        """
        if key is part of data then preprocessing is already done
        ex. sentence of POS-tags or sentence of lemma_POS-tags
        """
        if self.key in data:
            return data[self.key]
        else:
            return list(map(lambda s: Extractor(s)[self.key], data.iloc[:,0]))

class ArrayCaster(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, data):
        return np.transpose(np.matrix(data))