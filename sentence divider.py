# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 17:34:56 2018

@author: ricar
"""
class labeled_sentence:
    def __init__(self, sentence, label):
        self.sentence = sentence
        self.label = label

file = open("input_oct16.txt", "r")
mat_lines = file.readlines()

list_labeled_sentence = []

for line in mat_lines:
    line = line.split(':')[-1]
    line = line.split('.')[:-1]
    
    for separate_line in line:1
        if separate_line.count(' ') > 60:
            print (separate_line)
            label = input("What is the label for this? 1-4")
            ##HACER CHEQUEOS
            pair = labeled_sentence(separate_line, label)
            list_labeled_sentence.append(pair)

#list_labeled_sentence es una lista de objetos 'labeled_sentence'