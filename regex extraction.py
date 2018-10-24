
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import os
os.chdir("C:/Users/orteg/Desktop")
import xlwt
import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from matplotlib import pyplot as plt
tokenizer = nltk.data.load("tokenizers/punkt/spanish.pickle")

# Using the newer with construct to close the file automatically.
filename = "C:\\Users\orteg\desktop\Transcripcion 16 Oct.txt"
with open(filename, encoding="utf8") as f:
    data = f.read()
    
pattern = "(?:El presidente diputado|La presidenta diputada|La secretaria diputada|El diputado|La diputada|La secretaria) (?:[A-Z][a-z]*?\s)?.*?(?=:|\s[(])"
names = re.findall(pattern,data)
re.split(pattern,data)[3]

speeches = re.split('|'.join(map(re.escape, names)), data)

df_names = {'names': names}
df_names = pd.DataFrame(data=df_names)
df_names.to_excel('df_names.xlsx',sheet_name='sheet1', index=False)

df_speeches = {'speeches': speeches[1:len(speeches)]}
df_speeches = pd.DataFrame(data=df_speeches)
df_speeches.to_excel('df_speeches.xlsx',sheet_name='sheet1', index=False)

#####################
lle_train_df.rename(columns = {0:'lle1',1:'lle2'}, inplace =True)
lle_train_df.to_excel('lle_tr.xlsx',sheet_name='sheet1', index=False)

# Split scene_one into sentences: sentences
sentences = tokenizer.tokenize(acusacion)

# Use word_tokenize to tokenize the fourth sentence: tokenized_sent
tokenized_sent = word_tokenize(sentences[3])

# Make a set of unique tokens in the entire scene: unique_tokens
unique_tokens = set(word_tokenize(acusacion))

# Print the unique tokens result
print(unique_tokens)

word_lengths = [len(w) for w in word_tokenize(acusacion, language='spanish')]
plt.hist(word_lengths)

