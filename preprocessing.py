# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 12:25:54 2018

@author: orteg
"""
from spacy.lang.es import Spanish
import pandas as pd
import os
os.chdir("C:/Users/orteg/Dropbox/ArchOrg/1Almacen/EMDS/Computacion/3 Cursos/KUL Knowledge and the Web/Project")
import itertools
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer 
from nltk.stem import PorterStemmer

nlp_spacy = Spanish() 
#tokenizer = nltk.data.load("tokenizers/punkt/spanish.pickle")

# Using the newer with construct to close the file automatically.
data_csv = pd.read_csv("df_speeches.csv", encoding="ISO-8859-1")
data_excel = pd.read_excel("df_speeches.xlsx")

# A bit more preprocessing to get out ": " from the Speech
data = data_excel.loc[:,'speech']

# TODO-NN: eliminate the double colon

# Tokenize sentences
sentences = [sent_tokenize(i) for i in data.dropna().tolist()]

# Flat a nested Python List & remove whitespaces (leading and trailing)
flat_sentences = [item for items in sentences for item in items]
flat_sentences = [item.strip() for item in flat_sentences]

# Filter 
words_list = [i.split() for i in flat_sentences]
words_filtered = [[word for word in words_ if word.lower() not in stopwords.words('spanish')] for words_ in words_list]

# Greater or equal to 7
index0 = [x >= 7 for x in [len(y) for y in words_filtered]]
long_sentences = list(itertools.compress(flat_sentences, index0))

# Create Dataset of long sentences
df_sent = pd.DataFrame(data= {'sentence': long_sentences, 'sentence_id': range(1,len(long_sentences)+1)})
df_sent.to_excel('df_sentences.xlsx',sheet_name='sheet1', index = False)

# Training and Test Split
x_labelled, x_unlabelled = train_test_split(df_sent, test_size=0.80, random_state = 123)
x_labelled_train, x_labelled_test = train_test_split(x_labelled, test_size=0.25, random_state = 123) 

# Create Excel files
x_unlabelled.to_excel('x_unlabelled.xlsx',sheet_name='sheet1', index = False)
x_labelled_train.to_excel('x_labelled_train.xlsx',sheet_name='sheet1', index=False)
x_labelled_test.to_excel('x_labelled_test.xlsx',sheet_name='sheet1', index=False)