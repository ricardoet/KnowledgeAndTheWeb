# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 23:00:34 2018

@author: orteg
"""
import numpy as np
import pandas as pd
import urllib3
try:
     from BeautifulSoup import BeautifulSoup
except ImportError:
     from bs4 import BeautifulSoup
http = urllib3.PoolManager()
url = 'http://cronica.diputados.gob.mx/Estenografia/64/2018/oct/20181016.html'
response = http.request('GET', url)
parse_html = BeautifulSoup(response.data)
array_par = parse_html.body.find_all('p')

#
names  = []
speech  = []
for par in array_par:
     par_bold = par.find('b')
     if par_bold : 
          names.append(par_bold.text)
          speech.append(par.text.replace(par_bold.text,''))
          
#  Create dataframes
df_names = {'names': names}
df_names = pd.DataFrame(data=df_names)
df_names.to_excel('df_names.xlsx',sheet_name='sheet1', index=False)

df_speeches = {'speech': speech}
df_speeches = pd.DataFrame(data=df_speeches)
df_speeches.to_excel('df_speeches.xlsx',sheet_name='sheet1', index=False)
df_speeches.to_csv('df_speeches.csv')

# TODO: Eliminate the extra symbol that appears at the end of each session