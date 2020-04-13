import numpy as np
import pandas as pd
from pprint import pprint

data = pd.DataFrame({"District": ["Suburban","Suburban","Rural","Urban","Rural","Rural","Suburban","Urban","Rural","Urban"],
                     "House_Type": ["Detached","Semi-detached","Semi-detached","Detached","Detached","Semi-detached","Detached","Detached","Detached","Semi-detached"],
                     "Income": ["High","High","Low","Low","Low","Low","Low","High","Low","Low"],
                     "Previous_Customer": ["No","Yes","No","Yes","Yes","Yes","No","Yes","No","No"],
                     "Outcome": ["Nothing", "Respond","Respond","Nothing","Nothing","Respond","Nothing","Respond","Respond","Nothing"]},
                    columns=["District", "House_Type", "Income", "Previous_Customer", "Outcome"])
# descriptive features
features = data[["District", "House_Type", "Income", "Previous_Customer"]]
# target feature
target = data["Outcome"]


def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = -np.sum([(counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy


def InfoGain(data, split_attribute_name, target_name):
    total_entropy = entropy(data[target_name])

    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    Weighted_Entropy = np.sum([(counts[i] / np.sum(counts)) *
                               entropy(data.where(data[split_attribute_name] == vals[i]).dropna()[target_name])
                               for i in range(len(vals))])

    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain


def ID3(data, originaldata, features, target_attribute_name, parent_node_class=None):



    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]

    elif len(data) == 0:
        return np.unique(originaldata[target_attribute_name]) \
            [np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])]

    elif len(features) == 0:
        return parent_node_class


    else:

        parent_node_class = np.unique(data[target_attribute_name]) \
            [np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]

        # choose attribute
        item_values = [InfoGain(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        # make tree
        tree = {best_feature: {}}

        features = [i for i in features if i != best_feature]


        for value in np.unique(data[best_feature]):
            # 데이터 분할
            sub_data = data.where(data[best_feature] == value).dropna()

            subtree = ID3(sub_data, data, features, target_attribute_name, parent_node_class)
            tree[best_feature][value] = subtree

        return (tree)



print('InfoGain( District ) = ', round(InfoGain(data, "District", "Outcome"), 5), '\n')
print('InfoGain( House Type ) = ', round(InfoGain(data, "House_Type", "Outcome"), 5), '\n')
print('InfoGain( Income ) = ', round(InfoGain(data, "Income", "Outcome"), 5), '\n')
print('InfoGain( Previous Customer ) = ', round(InfoGain(data, "Previous_Customer", "Outcome"), 5))

print()

# unique value
print('numpy.unique: ', np.unique(data["Outcome"], return_counts = True)[1])
#  max value
print('numpy.max: ', np.max(np.unique(data["Outcome"], return_counts = True)[1]))
# maxindex
print('numpy.argmax: ', np.argmax(np.unique(data["Outcome"], return_counts = True)[1]))

print()

tree = ID3(data, data, ["District", "House_Type", "Income", "Previous_Customer"], "Outcome")

pprint(tree)
