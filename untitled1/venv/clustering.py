import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load the train and test datasets to create two DataFrames
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)
test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)

# For the train set
train.isna().head()

# For the test set
test.isna().head()

# Fill missing values with mean column values in the train set
train.fillna(train.mean(), inplace=True)
# Fill missing values with mean column values in the test set
test.fillna(test.mean(), inplace=True)

# print(train.isna().sum())
#
# print(test.isna().sum())

train = train.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
test = test.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)

labelEncoder = LabelEncoder()
labelEncoder.fit(train['Sex'])
labelEncoder.fit(test['Sex'])
train['Sex'] = labelEncoder.transform(train['Sex'])
test['Sex'] = labelEncoder.transform(test['Sex'])

X = np.array(train.drop(['Survived'], 1).astype(float))
y = np.array(train['Survived'])

kmeans = KMeans(n_clusters=2)  # cluster the passenger records into Survived or Not survived

kmeans.fit(X)
# KMeans(algorithm='auto', copy_x=True, init='k-means++',
#        max_iter=300, n_clusters=2, n_init=10, n_jobs=1,
#        precompute_distances='auto', random_state=None,
#        tol=0.0001, verbose=0)

# Let's see the percentage of passenger records that were clustered correctly.

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print('<Maximum number if iterations = 300>')
print(correct / len(X))
print('')


kmeans = KMeans(n_clusters=2, max_iter=600, algorithm='auto')
kmeans.fit(X)

# KMeans(algorithm='auto', copy_x=True, init='k-means++',
#        max_iter=600, n_clusters=2, n_init=10, n_jobs=1,
#        precompute_distances='auto', random_state=None,
#        tol=0.0001, verbose=0)

correct = 0

for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print('<Maximum number if iterations = 600>')
print(correct / len(X))
print('')




kmeans = KMeans(n_clusters=3, max_iter=600, algorithm='auto')
kmeans.fit(X)

# KMeans(algorithm='auto', copy_x=True, init='k-means++',
#        max_iter=600, n_clusters=3, n_init=10, n_jobs=1,
#        precompute_distances='auto', random_state=None,
#        tol=0.0001, verbose=0)

correct = 0

for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print('<Maximum number if iterations = 600 & 3 cluster>')
print(correct / len(X))
print('')


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
kmeans.fit(X_scaled)

# KMeans(algorithm='auto', copy_x=True, init='k-means++',
#        max_iter=600, n_clusters=2, n_init=10, n_jobs=1,
#        precompute_distances='auto', random_state=None, tol=0.0001, verbose=0)

correct = 0

for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print('<Maximum number if iterations = 600 & MinMaxScaler>')
print(correct / len(X))
