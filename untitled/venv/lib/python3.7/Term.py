import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings(action='ignore')
df = pd.read_csv('/Users/halim/Downloads/clean.csv')
df = df.drop(['Unnamed: 0'], axis=1)
x = df.drop(['Chance of Admit '], axis=1)
y = df['Chance of Admit ']
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, shuffle=True)
xtrain, xtrain1, ytrain, ytrain1 = train_test_split(xtrain, ytrain, test_size=50, shuffle=True)
xtrain, xtrain2, ytrain, ytrain2 = train_test_split(xtrain, ytrain, test_size=50, shuffle=True)
xtrain, xtrain3, ytrain, ytrain3 = train_test_split(xtrain, ytrain, test_size=50, shuffle=True)
xtrain, xtrain4, ytrain, ytrain4 = train_test_split(xtrain, ytrain, test_size=50, shuffle=True)
xtrain, xtrain5, ytrain, ytrain5 = train_test_split(xtrain, ytrain, test_size=50, shuffle=True)
xtrain, xtrain6, ytrain, ytrain6 = train_test_split(xtrain, ytrain, test_size=50, shuffle=True)
xtrain, xtrain7, ytrain, ytrain7 = train_test_split(xtrain, ytrain, test_size=50, shuffle=True)

tree = DecisionTreeClassifier(criterion='entropy')
tree.fit(xtrain, ytrain)
tree1 = DecisionTreeClassifier(criterion='entropy')
tree1.fit(xtrain1, ytrain1)
tree2 = DecisionTreeClassifier(criterion='entropy')
tree2.fit(xtrain2, ytrain2)
tree3 = DecisionTreeClassifier(criterion='entropy')
tree3.fit(xtrain3, ytrain3)
tree4 = DecisionTreeClassifier(criterion='entropy')
tree4.fit(xtrain4, ytrain4)
tree5 = DecisionTreeClassifier(criterion='entropy')
tree5.fit(xtrain5, ytrain5)
tree6 = DecisionTreeClassifier(criterion='entropy')
tree6.fit(xtrain6, ytrain6)
tree7 = DecisionTreeClassifier(criterion='entropy')
tree7.fit(xtrain7, ytrain7)
result = []
result.append(tree.predict(xtest))
result.append(tree1.predict(xtest))
result.append(tree2.predict(xtest))
result.append(tree3.predict(xtest))
result.append(tree4.predict(xtest))
result.append(tree5.predict(xtest))
result.append(tree6.predict(xtest))
result.append(tree7.predict(xtest))

print(result)
final = []
for i in range(100):
    num1 = 0
    num2 = 0
    num3 = 0
    num4 = 0
    for j in range(8):
        if result[j][i] == 1:
            num1 = num1 + 1
        elif result[j][i] == 2:
            num2 = num2 + 1
        elif result[j][i] == 3:
            num3 = num3 + 1
        else:
            num4 = num4 + 1
    if num1 > num2 and num1 > num3 and num1 > num4:
        final.append(1)
    elif num2 > num1 and num2 > num3 and num2 > num4:
        final.append(2)
    elif num3 > num1 and num3 > num2 and num3 > num4:
        final.append(3)
    else:
        final.append(4)
print(final)
print('--------------')
print(ytest)
matrix = confusion_matrix(ytest, final)
print(matrix)
print("Precision Score: ", precision_score(ytest, final, average='weighted'))
print("Recall Score: ", recall_score(ytest, final, average='weighted'))
print("f1 Score: ", f1_score(ytest, final, average='weighted'))

import seaborn as sns
import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(matrix,annot = True)
plt.title("Confusion Matrix")
plt.show()









