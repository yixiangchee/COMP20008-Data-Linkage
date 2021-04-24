import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv

from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

from sklearn.tree import DecisionTreeClassifier

##load in the data
life=pd.read_csv('life.csv',encoding = 'ISO-8859-1',na_values='..')
world=pd.read_csv('world.csv',encoding = 'ISO-8859-1',na_values='..')

result = pd.merge(life, world, on=['Country Code'])
result = result.sort_values(by='Country Code', ascending = True)

##get just the features
data=result[['Access to electricity (% of population) [EG.ELC.ACCS.ZS]',
 'Adjusted net national income per capita (current US$) [NY.ADJ.NNTY.PC.CD]',
 'Age dependency ratio (% of working-age population) [SP.POP.DPND]',
 'Cause of death, by communicable diseases and maternal, prenatal and nutrition conditions (% of total) [SH.DTH.COMM.ZS]',
 'Current health expenditure per capita (current US$) [SH.XPD.CHEX.PC.CD]',
 'Fertility rate, total (births per woman) [SP.DYN.TFRT.IN]',
 'Fixed broadband subscriptions (per 100 people) [IT.NET.BBND.P2]',
 'Fixed telephone subscriptions (per 100 people) [IT.MLT.MAIN.P2]',
 'GDP per capita (constant 2010 US$) [NY.GDP.PCAP.KD]',
 'GNI per capita, Atlas method (current US$) [NY.GNP.PCAP.CD]',
 'Individuals using the Internet (% of population) [IT.NET.USER.ZS]',
 'Lifetime risk of maternal death (%) [SH.MMR.RISK.ZS]',
 'People using at least basic drinking water services (% of population) [SH.H2O.BASW.ZS]',
 'People using at least basic drinking water services, rural (% of rural population) [SH.H2O.BASW.RU.ZS]',
 'People using at least basic drinking water services, urban (% of urban population) [SH.H2O.BASW.UR.ZS]',
 'People using at least basic sanitation services, urban (% of urban population) [SH.STA.BASS.UR.ZS]',
 'Prevalence of anemia among children (% of children under 5) [SH.ANM.CHLD.ZS]',
 'Secure Internet servers (per 1 million people) [IT.NET.SECR.P6]',
 'Self-employed, female (% of female employment) (modeled ILO estimate) [SL.EMP.SELF.FE.ZS]',
 'Wage and salaried workers, female (% of female employment) (modeled ILO estimate) [SL.EMP.WORK.FE.ZS]']].astype(float)

##get just the class labels
classlabel=result['Life expectancy at birth (years)']

X_train, X_test, y_train, y_test = train_test_split(data, classlabel, train_size=0.7, test_size=0.3, random_state=200)
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_train.median())

med = X_train.median()

#normalise the data to have 0 mean and unit variance using the library functions.  This will help for later
#computation of distances between instances
scaler = preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)


writerdf=pd.DataFrame({'feature': data.columns, 'median':med,'mean': scaler.mean_,'variance': scaler.var_})
writerdf=writerdf.round(decimals=3)
writerdf.to_csv(r'task2a.csv', index = False)

dt = DecisionTreeClassifier(random_state=200, max_depth=3)
dt.fit(X_train, y_train)
y_pred=dt.predict(X_test)
print(f"Accuracy of decision tree: {accuracy_score(y_test, y_pred):.3f}")

knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred=knn.predict(X_test)
print(f"Accuracy of k-nn (k=3): {accuracy_score(y_test, y_pred):.3f}")

knn = neighbors.KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
y_pred=knn.predict(X_test)
print(f"Accuracy of k-nn (k=7): {accuracy_score(y_test, y_pred):.3f}")