# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 03:00:09 2018

@author: orteg
"""
import os
os.chdir("C:/Users/orteg/Dropbox/ArchOrg/1Almacen/EMDS/Computacion/3 Cursos/KUL Knowledge and the Web/Project")
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create simulation for binary classification with 10 variables and 10k observations
X, y = make_blobs(n_samples=10000, centers=2, n_features=10, random_state=0)

y = np.array(y, dtype=X.dtype)

# This dataframe is just to check visually how it looks the structure of the Gaussian blobs
df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))

# Make scatter plot of 1975 data
colors = {0:'red', 1:'blue'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
plt.show()

# Building complete simulation dataframe that will be exported to the chosen os.dir 
Xy = np.hstack((X, y[:, None]))

cols = ['x'+str(i) for i in range(10)]  # crea
cols.append('label')

df2 = pd.DataFrame(Xy, columns = cols)
df2.to_csv('df_bin_simu.csv', index=False)
