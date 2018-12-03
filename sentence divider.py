# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 17:34:56 2018

@author: ricar
"""

file = open("input_oct16.txt", "r")
mat_lines = file.readlines()

for line in mat_lines:
    line = line.split(':')[-1]
    line = line.split('.')[:-1]
    print (line)
