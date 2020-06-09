import pandas as pd
import numpy as np
import io
import warnings
from sklearn import linear_model

warnings.filterwarnings(action='ignore')
from matplotlib import pyplot as plt
from pandas import Series, DataFrame

df = pd.read_csv('knn_data.csv')
df['longitude'] = pd.to_numeric(df['longitude'])
df['latitude'] = pd.to_numeric(df['latitude'])
df.loc[:, 'Predict'] = np.array([0] * len(df))

# Normarlizatoin data
longtitude = df['longitude']
latitude = df['latitude']

df1 = (longtitude - longtitude.mean()) / longtitude.std()
df2 = (latitude - latitude.mean()) / latitude.std()

df.at[:, 'longitude'] = round(df['longitude'], 1)
df.at[:, 'latitude'] = round(df['latitude'], 1)

# Separate 5 dataset
dataset_1 = df.iloc[0:15]
dataset_1.reset_index(drop=True)

dataset_2 = df.iloc[15:30]
dataset_2.reset_index(drop=True)

dataset_3 = df.iloc[30:45]
dataset_3.reset_index(drop=True)

dataset_4 = df.iloc[45:60]
dataset_4.reset_index(drop=True)

dataset_5 = df.iloc[60:75]
dataset_5.reset_index(drop=True)

array = np.arange(15 * 75).reshape(15, 75)

# Test 1
for i in range(15):
    index = 0
    min = 100000000

    for j in range(15, 75):  # j= 15~74
        uclean = (((df['longitude'][j] - dataset_1['longitude'][i]) ** 2) +
                  ((df['latitude'][j] - dataset_1['latitude'][i]) ** 2)) ** 1 / 2
        array[i][j] = uclean

        if array[i][j] < min:  # Find min value
            min = array[i][j]
            index = j  # Later, find nearest lang by index

    dataset_1['Predict'][i] = df['lang'][index]  # i=0~14

print(dataset_1)
print('')

# Test 2
for i in range(15):
    index = 0
    min = 100000000

    for j in range(75):  # j= 0~74
        if (j < 15) or (j > 30):
            uclean = (((df['longitude'][j] - dataset_2['longitude'][i + 15]) ** 2) + (
                        (df['latitude'][j] - dataset_2['latitude'][i + 15]) ** 2)) ** 1 / 2
            array[i][j] = uclean

            if array[i][j] < min:  # Find min value
                min = array[i][j]
                index = j  # Later, find nearest lang by index

    dataset_2['Predict'][i + 15] = df['lang'][index]  # i=0~14

print(dataset_2)
print('')

# Test 3
for i in range(15):
    index = 0
    min = 100000000

    for j in range(75):  # j= 0~74
        if (j < 30) or (j > 45):
            uclean = (((df['longitude'][j] - dataset_3['longitude'][i + 30]) ** 2) + (
                        (df['latitude'][j] - dataset_3['latitude'][i + 30]) ** 2)) ** 1 / 2
            array[i][j] = uclean

            if array[i][j] < min:  # Find min value
                min = array[i][j]
                index = j  # Later, find nearest lang by index

    dataset_3['Predict'][i + 30] = df['lang'][index]  # i=0~14

print(dataset_3)
print('')

# Test 4
for i in range(15):
    index = 0
    min = 100000000

    for j in range(75):  # j= 0~74
        if (j < 45) or (j > 60):
            uclean = (((df['longitude'][j] - dataset_4['longitude'][i + 45]) ** 2) + (
                        (df['latitude'][j] - dataset_4['latitude'][i + 45]) ** 2)) ** 1 / 2
            array[i][j] = uclean

            if array[i][j] < min:  # Find min value
                min = array[i][j]
                index = j  # Later, find nearest lang by index

    dataset_4['Predict'][i + 45] = df['lang'][index]  # i=0~14

print(dataset_4)
print('')

# Test 5
for i in range(15):
    index = 0
    min = 100000000

    for j in range(75):  # j= 0~74
        if j < 60:
            uclean = (((df['longitude'][j] - dataset_5['longitude'][i + 60]) ** 2) + (
                        (df['latitude'][j] - dataset_5['latitude'][i + 60]) ** 2)) ** 1 / 2
            array[i][j] = uclean

            if array[i][j] < min:  # Find min value
                min = array[i][j]
                index = j  # Later, find nearest lang by index

    dataset_5['Predict'][i + 60] = df['lang'][index]  # i=0~14

print(dataset_5)
