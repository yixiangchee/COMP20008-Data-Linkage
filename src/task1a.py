import pandas as pd
import textdistance
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import csv
import nltk
import re

buy_s=pd.read_csv('buy_small.csv',encoding = 'ISO-8859-1')
abt_s=pd.read_csv('abt_small.csv',encoding = 'ISO-8859-1')

buy_s=buy_s.loc[:,['idBuy','name','description','price']]
abt_s=abt_s.loc[:,['idABT','name','description','price']]

buy_s[['price']] = buy_s[['price']].replace('[\$,]','',regex=True).astype(float)
abt_s[['price']] = abt_s[['price']].replace('[\$,]','',regex=True).astype(float)


param = []
a_match = []

for b_idx, b_item in buy_s.iterrows():
    
    scores = []
    
    for a_idx, a_item in abt_s.iterrows():
            
        similarity=0
        name_d = 0
        price_n = 0.5
        model_n = 0.2
        brand_n = 0
        a_name = (a_item['name'].replace("-",'').replace("/",'').replace("\'",'')).lower()
        a_name = nltk.word_tokenize(a_name)
        b_name = (b_item['name'].replace("-",'').replace("/",'').replace("\'",'')).lower()
        b_name = nltk.word_tokenize(b_name)                            
        name_d = textdistance.cosine.normalized_similarity(a_name, b_name)
        
        if pd.notna(a_item['price']) and pd.notna(b_item['price']):
            diff = abs(a_item['price'] - b_item['price'])
            if diff < 10:
                price_n = 1
            elif diff > 50:
                price_n = 0
        
        mod_a = a_name[-1]
        mod_b = b_name[-1]
        
        brand_a = a_name[0]
        brand_b = b_name[0]
        
        if ((mod_a == mod_b and (not mod_a.isalpha()) and (not mod_b.isalpha())) 
            or (not mod_a.isalpha() and mod_a in b_name) or (not mod_b.isalpha() and mod_b in a_name)) :
            model_n = 1
        
        if brand_a == brand_b:
            brand_n = 1
            
        similarity = 0.5*name_d + 0.2*model_n + 0.2*brand_n + 0.1*price_n
        
        scores.append([a_item['idABT'], similarity, a_item['name']])
        
        
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    if scores[0][1] > 0.55:
        if scores[0][0] not in a_match:
            param.append([scores[0][0], b_item['idBuy'], scores[0][2], b_item['name'], scores[0][1]])
        elif scores[0][0] in a_match and scores[1][1] > 0.7:
            param.append([scores[1][0], b_item['idBuy'], scores[1][2], b_item['name'], scores[1][1]])
            a_match.append(scores[1][0])
    
    if scores[0][1] > 0.7:
        a_match.append(scores[0][0])
    


task1a_csv=open("task1a.csv",'w')
writer1a = csv.writer(task1a_csv)
writer1a.writerow(['idABT','idBuy'])

for row in param:
    writer1a.writerow([row[0], row[1]])