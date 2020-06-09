import pandas as pd
import numpy as np
import io
import warnings
from sklearn import linear_model

warnings.filterwarnings(action='ignore')
from matplotlib import pyplot as plt

df = pd.read_csv('linear_regression_data.csv')

df_training = df.head(24)
df_test = df.tail(6)

distance = df_training.loc[:, 'Distance']
time = df_training.loc[:, 'Delivery Time']

test_distance = df_test.loc[:, 'Distance']

reg = linear_model.LinearRegression()
reg.fit(X=pd.DataFrame(distance), y=time)
prediction = reg.predict(X=pd.DataFrame(test_distance))

plt.plot(test_distance, prediction, color='black')
plt.scatter(df_training.loc[:, 'Distance'], df_training.loc[:, 'Delivery Time'])
plt.scatter(df_test.loc[:, 'Distance'], df_test.loc[:, 'Delivery Time'])
plt.show()
