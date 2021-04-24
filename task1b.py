import pandas as pd
import csv

import nltk
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
import re


Lemm = WordNetLemmatizer()
stopWords = set(stopwords.words('english'))

task1ba_csv=open("abt_blocks.csv",'w')
writer1ba = csv.writer(task1ba_csv)
writer1ba.writerow(['block_key','product_id'])

task1bb_csv=open("buy_blocks.csv",'w')
writer1bb = csv.writer(task1bb_csv)
writer1bb.writerow(['block_key','product_id'])

buy_s=pd.read_csv('buy.csv',encoding = 'ISO-8859-1')
abt_s=pd.read_csv('abt.csv',encoding = 'ISO-8859-1')

buy_s=buy_s.loc[:,['idBuy','name','description','price','manufacturer']]
abt_s=abt_s.loc[:,['idABT','name','description','price']]

buy_s[['price']] = buy_s[['price']].replace('[\$,]','',regex=True).astype(float)
abt_s[['price']] = abt_s[['price']].replace('[\$,]','',regex=True).astype(float)

common_block = ['black','green','white','blue','yellow','red','purple','sony','panasonic','canon','digital','camera','lcd','system']

for a_idx, a_item in abt_s.iterrows():
    name = a_item['name']

    wordList = nltk.word_tokenize(name)
    wordList = [w.lower() for w in wordList]
    wordList = [word for word in wordList if word.isalpha()]

    filteredList = [w for w in wordList if not w in stopWords]
    
    for word in filteredList:
        stemWord = Lemm.lemmatize(word)
        if stemWord not in common_block:
            writer1ba.writerow([stemWord,a_item['idABT']])
            
for b_idx, b_item in buy_s.iterrows():
    name = b_item['name']

    wordList = nltk.word_tokenize(name)
    wordList = [w.lower() for w in wordList]
    wordList = [word for word in wordList if word.isalpha()]

    filteredList = [w for w in wordList if not w in stopWords]
    
    for word in filteredList:
        stemWord = Lemm.lemmatize(word)
        if stemWord not in common_block:
            writer1bb.writerow([stemWord,b_item['idBuy']])