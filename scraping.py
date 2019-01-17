import numpy as np
import pandas as pd
import urllib3
try:
     from BeautifulSoup import BeautifulSoup
except ImportError:
     from bs4 import BeautifulSoup
http = urllib3.PoolManager()
names  = []
speech  = []
with open('urls.txt') as file:
    array_url = file.readlines()
    array_url = [x.strip() for x in array_url]

#array_url = ['http://cronica.diputados.gob.mx/Estenografia/64/2018/ago/20180829.html', 'http://cronica.diputados.gob.mx/Estenografia/64/2018/sep/20180906.html']
for url in array_url:
    response = http.request('GET', url)
    parse_html = BeautifulSoup(response.data, 'lxml')
    array_par = parse_html.body.find_all('p')
    
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