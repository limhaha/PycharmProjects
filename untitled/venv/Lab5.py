#Bagging
import pandas as pd
import numpy as np
import warnings
warnings.filterswarnings(action='ignore')

for j in range(len(compare)):
    if (compare [labels][j] == 0): compare['labels'][j] = 2
    elif (compare['labels'][j] == 1) : compare['labels'][j] = 0
    else: compare['labels'][j] = 1

    iris = pd.read_csv('Iris.csv', encoding='utf-8')
    labels = Iris ['Species']
    iris.head()

    sample = []
    sample = pd.read_csv('Iris_bagging_dataset (1).csv', encoding='utf-8')
    smaples.append(sample)
    sample = pd.read_csv('Iris_bagging_dataset (2).csv', encoding='utf-8')
    smaples.append(sample)
    sample = pd.read_csv('Iris_bagging_dataset (3).csv', encoding='utf-8')
    smaples.append(sample)
    sample = pd.read_csv('Iris_bagging_dataset (4).csv', encoding='utf-8')
    smaples.append(sample)
    sample = pd.read_csv('Iris_bagging_dataset (5).csv', encoding='utf-8')
    smaples.append(sample)
    sample = pd.read_csv('Iris_bagging_dataset (6).csv', encoding='utf-8')
    smaples.append(sample)
    sample = pd.read_csv('Iris_bagging_dataset (7).csv', encoding='utf-8')
    smaples.append(sample)
    sample = pd.read_csv('Iris_bagging_dataset (8).csv', encoding='utf-8')
    smaples.append(sample)
    sample = pd.read_csv('Iris_bagging_dataset (9).csv', encoding='utf-8')
    smaples.append(sample)
    sample = pd.read_csv('Iris_bagging_dataset (10).csv', encoding='utf-8')
    smaples.append(sample)

