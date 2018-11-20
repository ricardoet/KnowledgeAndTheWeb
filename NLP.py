# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 12:25:54 2018

@author: orteg
"""
from spacy.lang.es import Spanish
import pandas as pd
import os
os.chdir("C:/Users/orteg/Dropbox/ArchOrg/1Almacen/EMDS/Computacion & Data Sc/3 Cursos/KUL Knowledge and the Web/Project")
import itertools
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer

nlp_spacy = Spanish() 
#tokenizer = nltk.data.load("tokenizers/punkt/spanish.pickle")

# Using the newer with construct to close the file automatically.
data = pd.read_csv("df.csv", encoding="ISO-8859-1")

# A bit more preprocessing to get out ": " from the Speech

speech = [i[2:]for i in data.Speech]

# Tokenize sentences
sentences = [sent_tokenize(i) for i in speech]

# Flat a nested Python List
flat_sentences = [item for items in sentences for item in items]

# Filter 
len_vector = [len(i.split()) for i in flat_sentences]

index1 = [x > 4 for x in len_vector]
long_sentences = list(itertools.compress(flat_sentences, index1))

# Create Dataset of long sentences
df_sent = pd.DataFrame(data= {'sentences': long_sentences[0:len(long_sentences)]})
df_sent.to_excel('df_sentences.xlsx',sheet_name='sheet1', index=False)

# Training and Test Split
train_x = df_sent[0:200]
test_x = df_sent[201:]

# Instantiate the vectorizer
word_vectorizer = TfidfVectorizer(stop_words=stopwords.words("spanish"),
                                  strip_accents = "unicode",
                                  ngram_range=(1, 3))

word_vectorizer.fit(train_x)
train_x_word_features = word_vectorizer.transform(train_x)

# Lemmanizing (INCOMPLETE - we need Pattern)





#######################
# Make a set of unique tokens in the entire scene: unique_tokens
unique_tokens = set(word_tokenize(acusacion))

# Print the unique tokens result
print(unique_tokens)

word_lengths = [len(w) for w in word_tokenize(acusacion, language='spanish')]
plt.hist(word_lengths)
