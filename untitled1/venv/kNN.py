import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')

df = {'height': [158, 158, 158, 160, 160, 163, 163, 160, 163, 165, 165, 165, 168, 168, 168,
                 170, 170, 170],
      'weight': [58, 59, 63, 59, 60, 60, 61, 64, 64, 61, 62, 65, 62, 63, 66,
                 63, 64, 68],
      'size': ['M', 'M', 'M', 'M', 'M', 'M', 'M', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L',
               'L', 'L', 'L']}

Tee = pd.DataFrame(df)
Tee.head()
Tee_original = Tee

height, weight = input("Input height and weight : ").split()

height = int(height)
weight = int(weight)
type(height)

df = {'height': [height],
      'weight': [weight],
      'size': ['']}

new_customer = pd.DataFrame(df)

Tee['Euclidean distance'] = 0.0

length = len(Tee['height'])

for i in range(length):
    Tee['Euclidean distance'][i] = np.sqrt(np.power(Tee['height'][i] - new_customer['height'], 2)
                                           + np.power(Tee['weight'][i] - new_customer['weight'], 2))
#RANK
print('')
print('<Top 5 Ranked Records>')
print(Tee.head())

Tee = Tee.sort_values(['Euclidean distance'], ascending=[True])

# print(Tee.head())

Tee = Tee.reset_index(drop=True)
Tee.head()

KNN = Tee
Tee = KNN.iloc[0:5].copy()
# print(KNN)

majority = {'size': ['M', 'L'],
            'number': [0, 0]}
majority = pd.DataFrame(majority)

for i in range(5):
    if KNN['size'][i] == majority['size'][0]:
        majority['number'][0] = majority['number'][0] + 1
    else:
        majority['number'][1] = majority['number'][1] + 1

# print(majority)

if majority['number'][0] > majority['number'][1]:
    new_customer['size'] = majority['size'][0]
else:
    new_customer['size'] = majority['size'][1]

print('')
print('<New Customer DATA>')
print(new_customer)

df_M = Tee_original[Tee_original['size'] == 'M']
df_L = Tee_original[Tee_original['size'] == 'L']

plt.figure()

plt.scatter(df_M['weight'], df_M['height'], marker='s')
plt.scatter(df_L['weight'], df_L['height'], marker='^')
plt.scatter(new_customer['weight'], new_customer['height'], marker='*')

plt.show()
