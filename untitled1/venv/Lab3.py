import pandas as pd
import numpy as np

df = pd.DataFrame({'column_a':[3, '?', 2, 5],
                   'column_b':['*', 4, 5, 6],
                   'column_c':['+', 3, 2, '&'],
                   'column_d':[5, '?', 7, '!']})

print(df)