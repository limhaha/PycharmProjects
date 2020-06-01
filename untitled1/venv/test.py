import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd
import math
from sklearn import preprocessing
from math import log
from pandas import Series,DataFrame

df=pd.read_excel('knn_data.xlsx')
t = DataFrame(df)
t.head()

height, weight = input("input height & weight").split()
height = int(height)
weight = int(weight)

print(type(height))

df={'height':[height],
     'weight':[weight],
    'size':['']}
new = DataFrame(df)
print(new)

t['dist']=0.0
length = len(t['height'])
for i in range(length):
    t['dist'][i] = np.sqrt(np.power(t['height'][i]-new['height'],2)
                           +np.power(t['weight'][i]-new['weight'],2))
t.head()

t = t.sort_values(['dist'],ascending=[True])
t.head()
t = t.reset_index(drop=True)
t.head()
knn7 = t
knn7 = knn7.iloc[0:7]
print(knn7)

num = {'size':['M','L'],
       'number':[0,0]}
num = DataFrame(num)

for i in range(7):
    if(knn7['size'][i]==num['size'][0]):
        num['number'][0] = num['number'][0] +1
    else:
        num['number'][1] = num['number'][1] +1

print(num)

if(num['number'][0]>num['number'][1]):
    new['size'] = num['size'][0]
else:
      new['size'] = num['size'][1]

print(new)