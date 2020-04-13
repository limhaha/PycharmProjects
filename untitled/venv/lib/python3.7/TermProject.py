import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
import os
"""
df = pd.read_csv("/Users/halim/Documents/자료/2019-1/데이터과학/텀프/graduate-admissions/Admission_Predict.csv",sep = ",")

serialNo = df["Serial No."].values

df.drop(["Serial No."],axis=1,inplace = True)

df=df.rename(columns = {'Chance of Admit ':'Chance of Admit'})

y = df["Chance of Admit"].values
x = df.drop(["Chance of Admit"],axis=1)

# separating train (80%) and test (%20) sets
from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.20,random_state = 42)

# normalization
from sklearn.preprocessing import MinMaxScaler
scalerX = MinMaxScaler(feature_range=(0, 1))
x_train[x_train.columns] = scalerX.fit_transform(x_train[x_train.columns])
x_test[x_test.columns] = scalerX.transform(x_test[x_test.columns])

#LinearRegression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
y_head_lr = lr.predict(x_test)

print("real value of y_test[1]: " + str(y_test[1]) + " -> the predict: " + str(lr.predict(x_test.iloc[[1],:])))
print("real value of y_test[2]: " + str(y_test[2]) + " -> the predict: " + str(lr.predict(x_test.iloc[[2],:])))

from sklearn.metrics import r2_score
print("r_square score: ", r2_score(y_test,y_head_lr))

print()

y_head_lr_train = lr.predict(x_train)
print("r_square score (train dataset): ", r2_score(y_train,y_head_lr_train))

#DecisionTree
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state = 42)
dtr.fit(x_train,y_train)
y_head_dtr = dtr.predict(x_test)

from sklearn.metrics import r2_score
print("r_square score: ", r2_score(y_test,y_head_dtr))
print("real value of y_test[1]: " + str(y_test[1]) + " -> the predict: " + str(dtr.predict(x_test.iloc[[1],:])))
print("real value of y_test[2]: " + str(y_test[2]) + " -> the predict: " + str(dtr.predict(x_test.iloc[[2],:])))

y_head_dtr_train = dtr.predict(x_train)
print("r_square score (train dataset): ", r2_score(y_train,y_head_dtr_train))

y = np.array([r2_score(y_test,y_head_lr)])
x = ["LinearRegression", "DecisionTreeReg."]
plt.bar(x,y)
plt.title("Linear Regression")
plt.xlabel("Regressor")
plt.ylabel("r2_score")
plt.show()

print("real value of y_test[5]: " + str(y_test[5]) + " -> the predict: " + str(lr.predict(x_test.iloc[[5],:])))
print("real value of y_test[5]: " + str(y_test[5]) + " -> the predict: " + str(dtr.predict(x_test.iloc[[5],:])))

print()

print("real value of y_test[50]: " + str(y_test[50]) + " -> the predict: " + str(lr.predict(x_test.iloc[[50],:])))
print("real value of y_test[50]: " + str(y_test[50]) + " -> the predict: " + str(dtr.predict(x_test.iloc[[50],:])))

red = plt.scatter(np.arange(0,80,5),y_head_lr[0:80:5],color = "red")
blue = plt.scatter(np.arange(0,80,5),y_head_dtr[0:80:5],color = "blue")
black = plt.scatter(np.arange(0,80,5),y_test[0:80:5],color = "black")

plt.title("Linear Regression")
plt.xlabel("Index of Candidate")
plt.ylabel("Chance of Admit")
plt.legend((red,blue,black),('LR', 'DTR', 'REAL'))
plt.show()

df["Chance of Admit"].plot(kind = 'hist',bins = 200,figsize = (6,6))
plt.title("Chance of Admit")
plt.xlabel("Chance of Admit")
plt.ylabel("Frequency")
plt.show()


"""
"""
import numpy as np
import csv
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import linear_model

result_all = []
result_gre = []
result_toefl = []

with open("/Users/halim/Documents/자료/2019-1/데이터과학/텀프/graduate-admissions/Admission_Predict_dirty.csv") as csvfile:
    reader = csv.reader(csvfile)  # change contents to floats
    for row in reader:  # each row is a list
        result_all.append(row)

result_all_arr = np.asarray(result_all)
result_gre = result_all_arr[1:, 1]
result_toefl = result_all_arr[1:, 2]

error_gre = []
error_toefl = []
error_all = []

print(result_gre)
print(result_toefl)

# GRE error data에 0 넣기
for i in range(0, result_gre.size):
    if result_gre[i] == "":
        error_gre.append(i)
        result_gre[i] = '0'
    elif int(result_gre[i]) < 0:
        error_gre.append(i)
        result_gre[i] = '0'
    elif int(result_gre[i]) > 350:
        error_gre.append(i)
        result_gre[i] = '0'

# TOEFL error data에 0 넣기
for i in range(0, result_toefl.size):
    if result_toefl[i] == "":
        error_toefl.append(i)
        result_toefl[i] = '0'
    elif int(result_toefl[i]) < 0:
        error_toefl.append(i)
        result_toefl[i] = '0'
    elif int(result_toefl[i]) > 150:
        error_toefl.append(i)
        result_toefl[i] = '0'

error_all = error_toefl + error_gre

# astype= int
b_result_gre = result_gre.astype(int)
b_result_toefl = result_toefl.astype(int)

clean_result_gre = np.delete(result_gre, error_all)
clean_result_toefl = np.delete(result_toefl, error_all)

clean_result_gre = clean_result_gre.astype(int)
clean_result_toefl = clean_result_toefl.astype(int)

# Linear regression
A = np.vstack([clean_result_toefl, np.ones(len(clean_result_toefl))]).T
m, c = np.linalg.lstsq(A, clean_result_gre, rcond=None)[0]

# m 기울기 , c 절
plt.plot(clean_result_toefl, clean_result_gre, 'o', label='Original data', markersize=5)
plt.plot(clean_result_toefl, m * clean_result_toefl + c, 'r', label='Fitted line')
plt.legend()
plt.show()

bad_gre = []
bad_toefl = []

for i in range(0, b_result_gre.size):
    if b_result_gre[i] == 0:
        b_result_gre[i] = m * b_result_gre[i] + c
        if b_result_gre[i] < 350:
            clean_result_gre.append(b_result_gre[i])
            clean_result_toefl.append(b_result_toefl[i])

for i in range(0, b_result_toefl.size):
    if b_result_toefl[i] == 0:
        b_result_toefl[i] = m * b_result_toefl[i] + c
        if b_result_toefl[i] > 0 and b_result_toefl[i] < 150:
            clean_result_gre.append(b_result_gre[i])
            clean_result_toefl.append(b_result_toefl[i])

print(clean_result_gre)
print(clean_result_toefl)

plt.scatter(clean_result_toefl, clean_result_gre, color="g")
plt.scatter(celan_gre, clean_toefl, color='r')
plt.show()


"""

# reading the dataset
df = pd.read_csv("/Users/halim/Documents/자료/2019-1/데이터과학/텀프/graduate-admissions/Admission_Predict.csv",sep = ",")



df=df.rename(columns = {'Chance of Admit ':'Chance of Admit'})
y = df["Chance of Admit"].values
x = df.drop(["Chance of Admit"],axis=1)

# separating train (80%) and test (20%) sets
from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.20, random_state=42)


# normalization
from sklearn.preprocessing import MinMaxScaler
scalerX = MinMaxScaler(feature_range=(0, 1))
x_train[x_train.columns] = scalerX.fit_transform(x_train[x_train.columns])
x_test[x_test.columns] = scalerX.transform(x_test[x_test.columns])

y_train_01 = [1 if each > 0.9
              else 0 for each in y_train]
y_test_01  = [1 if each > 0.9
              else 0 for each in y_test]

y_train_02 = [2 if each > 0.7
              else 0 for each in y_train]
y_test_02  = [2 if each > 0.7
              else 0 for each in y_test]

y_train_03 = [2 if each > 0.5
              else 0 for each in y_train]
y_test_03  = [2 if each > 0.5
              else 0 for each in y_test]

y_train_04 = [2 if each > 0.3
              else 0 for each in y_train]
y_test_04  = [2 if each > 0.3
              else 0 for each in y_test]


# list to array
y_train_01 = np.array(y_train_01)
y_test_01 = np.array(y_test_01)

#의사 결정트리 선언
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()


#훈련 ( 모든리프 노드 사용)
dtc.fit(x_train,y_train_01)

#의사결정 트리 선언 (트리 깊이 제한)
dtc = DecisionTreeClassifier(max_depth=3, random_state=0)

#훈련 (가지치기 : 리프노드 깊이 제한)
dtc.fit(x_train, y_train_01)

print("score: ", dtc.score(x_test,y_test_01))
print("real value of y_test_01[1]: " + str(y_test_01[1]) + " -> the predict: " + str(dtc.predict(x_test.iloc[[1],:])))
print("real value of y_test_01[2]: " + str(y_test_01[2]) + " -> the predict: " + str(dtc.predict(x_test.iloc[[2],:])))


print()



# confusion matrix
from sklearn.metrics import confusion_matrix
cm_dtc = confusion_matrix(y_test_01,dtc.predict(x_test))
# print("y_test_01 == 1 :" + str(len(y_test_01[y_test_01==1]))) # 29


# cm visualization
import seaborn as sns
import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm_dtc,annot = True)
plt.title("Decision Tree Test for Test Dataset")
plt.xlabel("predicted y values")
plt.ylabel("real y values")
plt.show()

from sklearn.metrics import precision_score, recall_score
print("precision_score: ", precision_score(y_test_01,dtc.predict(x_test)))
print("recall_score: ", recall_score(y_test_01,dtc.predict(x_test)))

from sklearn.metrics import f1_score
print("f1_score: ",f1_score(y_test_01,dtc.predict(x_test)))



print()
print()

"""
cm_dtc_train = confusion_matrix(y_train_01,dtc.predict(x_train))
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm_dtc_train,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("predicted y values")
plt.ylabel("real y values")
plt.title("Decision Tree Test for Train Dataset")
plt.show()







from sklearn.neighbors import KNeighborsClassifier

# finding k value
scores = []
for each in range(1, 50):
    knn_n = KNeighborsClassifier(n_neighbors=each)
    knn_n.fit(x_train, y_train_01)
    scores.append(knn_n.score(x_test, y_test_01))

plt.plot(range(1, 50), scores)
plt.xlabel("k")
plt.ylabel("accuracy")
plt.show()

knn = KNeighborsClassifier(n_neighbors=3)  # k가 3인 KNN 분류기 (클래스 생성)
knn.fit(x_train, y_train_01)
print("score of 3 :", knn.score(x_test, y_test_01))
print("real value of y_test_01[1]: " + str(y_test_01[1]) + " -> the predict: " + str(knn.predict(x_test.iloc[[1], :])))
print("real value of y_test_01[2]: " + str(y_test_01[2]) + " -> the predict: " + str(knn.predict(x_test.iloc[[2], :])))

# confusion matrix
from sklearn.metrics import confusion_matrix

cm_knn = confusion_matrix(y_test_01, knn.predict(x_test))
# print("y_test_01 == 1 :" + str(len(y_test_01[y_test_01==1]))) # 29

# cm visualization
import seaborn as sns
import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(cm_knn, annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax=ax)
plt.title("K neighbor Test for Test Dataset")
plt.xlabel("predicted y values")
plt.ylabel("real y values")
plt.show()

from sklearn.metrics import precision_score, recall_score

print("precision_score: ", precision_score(y_test_01, knn.predict(x_test)))
print("recall_score: ", recall_score(y_test_01, knn.predict(x_test)))

from sklearn.metrics import f1_score

print("f1_score: ", f1_score(y_test_01, knn.predict(x_test)))

cm_knn_train = confusion_matrix(y_train_01,knn.predict(x_train))
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm_knn_train,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("predicted y values")
plt.ylabel("real y values")
plt.title("K neighbor Test for Train Dataset")
plt.show()

"""