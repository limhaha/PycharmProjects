import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd
from sklearn import preprocessing
from math import log
from pprint import pprint
import io
import math
import warnings

warnings.filterwarnings(action='ignore')
from pandas import Series, DataFrame

feature_cols = Series(['level', 'lang', 'tweets', 'phd', 'interview'])
df = pd.read_csv('decision_tree_data.csv')

# Split training/test set
df_training = df.head(27)
df_test = df.tail(3)

global marketing, root_entropy, total
marketing = DataFrame(df_training)
total = len(marketing['interview'])

DecisionTree = {'depth': ['root', 'child1', 'child2', 'child3', 'child4'],
                'attribute': ['', '', '', '', '']}
DecisionTree = DataFrame(DecisionTree)


def compute_root_entropy(criterion):
    # Count the number of each value
    label = marketing[criterion]

    counts = label.value_counts()

    # Compute the probability per counts
    total = len(label)

    probs = counts / total

    # compute root entropy
    root_entropy = 0

    for i in range(len(probs)):
        if probs[i] != 0:
            root_entropy = root_entropy + (probs[i] * math.log(probs[i], 2))

        root_entropy = abs(root_entropy)
        return root_entropy


def compute_child_entropy(criterion, vals):
    # Find unique value
    feature_length = len(feature_cols)
    info_gain = np.zeros((feature_length, 1))
    val_length = len(vals)

    for index in range(feature_length):
        unique_value = marketing[feature_cols[index]].unique()
        unique_length = len(unique_value)

        prob = np.zeros((unique_length, val_length + 1))  # last col: sub total
        for i in range(val_length):
            for j in range(len(marketing[criterion])):
                if marketing[criterion][j] == vals[i]:
                    if marketing[feature_cols[index]][j] == unique_value[0]:
                        prob[0][i] = prob[0][i] + 1
                    elif marketing[feature_cols[index]] == unique_value[1]:
                        prob[1][i] = prob[1][i] + 1
                    else:
                        prob[2][i] = prob[2][i] + 1

        entropy = np.zeros((unique_length, 1))
        for j in range(unique_length):  # row
            for i in range(val_length):  # col
                # Compute subtotal
                prob[j][val_length] = prob[j][val_length] + prob[j][i]

            for i in range(val_length):
                # Compute probabilty
                prob[j][i] = prob[j][i] / prob[j][val_length]
                if prob[j][i] != 0:
                    entropy[j] = entropy[j] + (prob[j][i] * math.log(prob[j][i], 2))

        # abs value
        for j in range(unique_length):
            entropy[j] = abs(entropy[j])

        # Get the information gain
        for j in range(unique_length):
            info_gain[index] = info_gain[index] + (prob[j][val_length] * entropy[j])

        info_gain[index] = root_entropy - (info_gain[index] / total)

        # Show all information gain values per each feature
        info_gain = DataFrame(info_gain)
        info_gain['feature'] = feature_cols
        info_gain.rename(columns={info_gain.columns[0]: 'information gain'}, inplace=True)

        info_gain = info_gain.sort_values(by='information gain', ascending=False)
        info_gain = info_gain.reset_index(drop=True)

        return info_gain


# # Make decision tree
# def decisionTree(data, original_data, features, target_attribute_name, parent_node_class=None):
#     # When target attribute has only one data, return that target attribute name
#     if len(np.unique(data[target_attribute_name])) <= 1:
#         return np.unique(data[target_attribute_name])[0]
#
#     # When there are no data, return target attribute name that has max value in original data
#     elif len(data) == 0:
#         return np.unique(original_data[target_attribute_name]) \
#             [np.argmax(np.unique(original_data[target_attribute_name], return_counts=True)[1])]
#
#     # When there no features, return parent node
#     elif len(features) == 0:
#         return parent_node_class
#
#     # Make tree
#     else:
#         # Define parent node's data(True or False)
#         parent_node_class = np.unique(data[target_attribute_name]) \
#             [np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
#
#         # Select attribute that standard of deviding data
#         item_values = [info_gain(data, feature, target_attribute_name) for feature in features]
#         best_feature_index = np.argmax(item_values)
#         best_feature = features[best_feature_index]
#
#         # Make tree structure
#         tree = {best_feature: {}}
#
#         # Remove feature has max information gain
#         features = [i for i in features if i != best_feature]
#
#         # Create branch
#         for value in np.unique(data[best_feature]):
#             # Data split and drop missing data
#             sub_data = data.where(data[best_feature] == value).dropna()
#
#             # Recursive structure
#             subtree = decisionTree(sub_data, data, features, target_attribute_name, parent_node_class)
#             tree[best_feature][value] = subtree
#
#         return tree


# Get root entropy about interview
root_entropy = compute_root_entropy('interview')
print('<ROOT ENTROPY>')
print(root_entropy)
print('')

info_gain = compute_child_entropy('interview', ['FALSE', 'TRUE'])
print(info_gain)


# pprint(decisionTree(df_training, df_training, ['level', 'lang', 'tweets', 'phd'], 'interview'))