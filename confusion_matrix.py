import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import numpy as np

array_final = [[0, 2, 3, 7], 
        [1,7,5,11], 
        [0,1,12,32], 
        [0,6,11,289]]

array_inicial = [[0, 1, 5, 6], 
        [2, 4, 5, 13], 
        [0, 0, 14, 31], 
        [0, 8, 16, 282]]

array_inicial = [[8,28],
                 [7,344]]

array_final = [[13,23],
                 [7,344]]

array = [[0,0],
                 [0,0]]

for row in range(len(array_final)):
    for num in range(len(array_final)):
        array[row][num] = array_final[row][num] - array_inicial[row][num]

for row in range(len(array)):
    for num in range(len(array[row])):
        if sum(array[row]) > 0:
            array[row][num] = array[row][num] / sum(array_final[row])
        

df_cm = pd.DataFrame(array, index = [i for i in ["CS", "US"]],
                  columns = [i for i in ["CS", "US"]])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)