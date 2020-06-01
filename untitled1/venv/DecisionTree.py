import numpy as np
import pandas as pd
import math
from math import log
from pandas import Series

data = pd.read_excel('/Users/halim/sample_data.xlsx')
data2 = data[['District', 'House', 'Income', 'Customer', 'Outcome']]
# print(data2)

data_length = len(data)
# print(data_length)

selection = list(set(data['Outcome']))
# print(selection)

result = []
for i in selection:
    outcome_i = len(data[data['Outcome'] == i]) / data_length
    result.append(-outcome_i * math.log(outcome_i, 2))

total_entropy = sum(result)
print('<Root Entropy>')
print(total_entropy)

global root_entropy, total


def get_entropy(dataset, attribute, y):
    result = []
    dataset_len = len(dataset)
    selection = list(set(dataset[attribute]))
    selection2 = list(set(dataset[y]))

    # classify (attribute)
    for i in selection:
        data = dataset[dataset[attribute] == i]
        data_len = len(data)

        # classify (Outcome)
        for j in selection2:
            data2 = data[data[y] == j]
            data2_len = len(data2)
            result.append((data_len / dataset_len) *
                          (-data2_len / dataset_len * math.log(data2_len / data_len, 2)))
        print(result)
        entropy = sum(result)

        return entropy


# result1 = get_entropy(data2, 'District', 'Outcome')
#
# print(result1)
# print(total_entropy - result1)

total = len(data['Outcome'])
data.head()

DecisionTree = {'depth': ['root', 'child1', 'child2', 'child3'],
                'attribute': ['', '', '', '']}
DecisionTree = pd.DataFrame(DecisionTree)

# print(DecisionTree)


def compute_root_entropy(criterion):
    # number of values
    label = data[criterion]
    counts = {}
    counts = label.value_counts()

    total = len(label)
    probs = {}
    probs = counts / total

    root_entropy = 0
    for i in range(len(probs)):
        if (probs[i] != 0):
            root_entropy = root_entropy + (probs[i] * log(probs[i], 2))
        root_entropy = abs(root_entropy)
        return root_entropy


# result2 = compute_root_entropy('District',)
#
# print(result2)

feature_cols = pd.Series(['District', 'House type', 'Income', 'Previous Customer'])


def compute_child_entropy(criterion, vals):
    feature_length = len(feature_cols)
    information_gain = np.zeros((feature_length, 1))
    val_length = len(vals)

    for index in range(feature_length):
        uniq_val = data[feature_cols[index]].unique()
        uniq_len = len(uniq_val)

        prob = np.zeros((uniq_len, val_length + 1))
        for i in range(val_length):
            for j in range(len(data[criterion])):
                if (data[criterion][j] == vals[i]):
                    if (data[feature_cols[index]][j] == uniq_val[0]):
                        prob[0][i] = prob[0][i] + 1
                    elif (data[feature_cols[index]][j] == uniq_val[1]):
                        prob[1][i] = prob[1][i] + 1
                    else:
                        prob[2][i] = prob[2][i] + 1

        entropy = np.zeros((uniq_len, 1))

        for i in range(uniq_len):
            for j in range(val_length):
                prob[i][val_length] = prob[i][val_length] + prob[i][j]

            for j in range(val_length):
                prob[i][j] = prob[i][j] / prob[i][val_length]

                if (prob[i][j] != 0):
                    entropy[i] = entropy[j] + (prob[i][j] * math.log(prob[i][j], 2))

        for i in range(uniq_len):
            entropy[i] = abs(entropy[i])

        for i in range(uniq_len):
            information_gain[index] = information_gain[index] + (prob[i][val_length] * entropy[i])

        information_gain[index] = root_entropy - (information_gain[index] / total)

    information_gain = pd.DataFrame(information_gain)
    information_gain['feature'] = feature_cols
    information_gain.rename(columns={information_gain.colums[0]: 'information gain'}, inplace=False)
    information_gain = information_gain.reset_index(drop=Ture)

    return information_gain


# root_entropy = compute_root_entropy('Outcome')
# print(root_entropy)
#
# info_gain = compute_child_entropy('Outcome', ['Respond', 'Nothing'])
# print(info_gain)


print('')

print('<District Entropy - Root Entropy>')
final_dis = get_entropy(data2, 'District', 'Outcome')
print(total_entropy - final_dis)

print('')

print('<House Entropy - Root Entropy>')
final_house = get_entropy(data2, 'House', 'Outcome')
print(total_entropy - final_house)

print('')

print('<Income Entropy - Root Entropy>')
final_income = get_entropy(data2, 'Income', 'Outcome')
print(total_entropy - final_income)

print('')

print('<Customer Entropy - Root Entropy>')
final_customer = get_entropy(data2, 'Customer', 'Outcome')
print(total_entropy - final_customer)