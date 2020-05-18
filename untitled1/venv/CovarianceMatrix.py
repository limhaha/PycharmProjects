import pandas as pd

data = {'1':[30, 200, 10, 4],
        '2':[40, 300, 20, 4],
        '3':[50, 800, 20, 1],
        '4':[60, 600, 20, 2],
        '5':[40, 300, 20, 5]
        }

df = pd.DataFrame(data, columns=['1', '2', '3', '4', '5'])

covMatrix = pd.DataFrame.cov(df)
print(round(covMatrix, 1))
