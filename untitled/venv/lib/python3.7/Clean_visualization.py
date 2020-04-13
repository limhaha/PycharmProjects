import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
dataset=pd.read_csv('/Users/halim/Downloads/clean.csv')
fig = sns.distplot(dataset['GRE Score'])
plt.title('GRE Scores')
plt.show()
fig = sns.distplot(dataset['TOEFL Score'])
plt.title('TOEFL Score')
plt.show()
fig = sns.distplot(dataset['University Rating'])
plt.title('University Rating')
plt.show()
fig = sns.distplot(dataset['SOP'])
plt.title('SOP')
plt.show()
fig = sns.distplot(dataset['LOR '])
plt.title('LOR ')
plt.show()
fig = sns.distplot(dataset['Research'])
plt.title('Research')
plt.show()
fig = sns.distplot(dataset['Chance of Admit '])
plt.title('Chance of Admit ')
plt.show()

y = np.array([dataset["A"].sum(), dataset["B"].sum(),dataset["C"].sum(),dataset["D"].sum()])
x = ["A", "B", "C", "D"]
plt.bar(x,y)
plt.title("CGPA")
plt.show()
