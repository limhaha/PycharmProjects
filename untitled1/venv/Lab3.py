import pandas as pd
import numpy as np

df = pd.DataFrame({'column_a': [3, '?', 2, 5],
                   'column_b': ['*', 4, 5, 6],
                   'column_c': ['+', 3, 2, '&'],
                   'column_d': [5, '?', 7, '!']})

print('\n<Original Data Frame>\n')
print(df)
print('\n')


df.replace({'?': np.nan, '*': np.nan, '+': np.nan, '&': np.nan, '!': np.nan}, inplace=True)
print('<Replace non-numeric value with NaN>\n')
print(df)
print('\n')


print('<isna with any function>\n')
print(df.isna().any())
print('\n')


print('<isna with sum function>\n')
print(df.isna().sum())
print('\n')


print('<dropna with how any function>\n')
print(df.dropna(how='any'))
print('\n')


print('<dropna with how all function>\n')
print(df.dropna(how='all'))
print('\n')


print('<dropna with thresh 1 function>\n')
print(df.dropna(thresh=1))
print('\n')


print('<dropna with thresh 2 function>\n')
print(df.dropna(thresh=2))
print('\n')


print('<fillna with 100 function>\n')
print(df.fillna(100))
print('\n')


print('<fillna with mean function>\n')
print(df.fillna(df.mean()))
print('\n')


print('<fillna with median function>\n')
print(df.fillna(df.median()))
print('\n')


print('<fillna with ffill function>\n')
print(df.fillna(method='ffill'))
print('\n')


print('<fillna with bfill function>\n')
print(df.fillna(method='bfill'))
print('\n')
