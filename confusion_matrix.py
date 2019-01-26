import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import numpy as np

def print_confusion(array, names):
        df_cm = pd.DataFrame(array, index = [i for i in names], 
                             columns = [i for i in names])
    
        plt.figure(figsize = (10,7))
        sn.heatmap(df_cm, annot=True, cmap="YlGnBu")

def confusion_total(array, names):
    print_confusion(array, names)
    
def confusion_relative(array, names):
        for row in range(len(array)):
            total = sum(array[row])
            for num in range(len(array[row])):
                array[row][num] = array[row][num] / total
                
        print_confusion(array, names)


array_final = [[0, 2, 3, 7], 
               [1,7,5,11], 
               [0,1,12,32], 
               [0,6,11,289]]

array_final = [[1,1],
              [1,1]]

for row in range(len(array_final)):
    for num in range(len(array_final)):
        array[row][num] = array_final[row][num] - array_inicial[row][num]

for row in range(len(array)):
    for num in range(len(array[row])):
        if sum(array[row]) > 0:
            array[row][num] = array[row][num] / sum(array_final[row])
        

confusion_relative(array_final, ["CS", "US"])