import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn import linear_model

df = pd.read_csv('/Users/halim/Downloads/bmi_data_lab3.csv')

print(df)  #Print dataset statistical data, feature names & data types


# bit_mask0 = df['BMI'] == 0
# df_eweak = df[bit_mask0]
#
# bit_mask1 = df['BMI'] == 1
# df_weak = df[bit_mask1]
#
# bit_mask2 = df['BMI'] == 2
# df_normal = df[bit_mask2]
#
# bit_mask3 = df['BMI'] == 3
# df_overweight = df[bit_mask3]
#
# bit_mask4 = df['BMI'] == 4
# df_obesity = df[bit_mask4]
#
#
#
# plt.hist(df_eweak['Height (Inches)'], bins=10)
# plt.title('extremely weak')
# plt.xlabel('Height (Inches)')
# plt.ylabel('Number of student')
# plt.show()
#
# plt.hist(df_weak['Height (Inches)'], bins=10)
# plt.title('weak')
# plt.xlabel('Height (Inches)')
# plt.ylabel('Number of student')
# plt.show()
#
# plt.hist(df_normal['Height (Inches)'], bins=10)
# plt.title('normal')
# plt.xlabel('Height (Inches)')
# plt.ylabel('Number of student')
# plt.show()
#
# plt.hist(df_overweight['Height (Inches)'], bins=10)
# plt.title('overweight')
# plt.xlabel('Height (Inches)')
# plt.ylabel('Number of student')
# plt.show()
#
# plt.hist(df_obesity['Height (Inches)'], bins=10)
# plt.title('obesity')
# plt.xlabel('Height (Inches)')
# plt.ylabel('Number of student')
# plt.show()



# ht_wt = {'height' : df['Height (Inches)'], 'weight' : df['Weight (Pounds)']}
# ht_wt_df = pd.DataFrame(ht_wt)
#
# standard_scaler = preprocessing.StandardScaler()
# standard_scaler.fit(ht_wt_df)
# standard_scaler.transform(ht_wt_df)
#
# fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))
#
# ax1.set_title('Before Scaling')
# sns.kdeplot(ht_wt_df)
#
# ax2.set_title('After MinMax Scaler')
# sns.kdeplot(ht_wt_df)
#
# plt.show()

nan_df = pd.read_csv('/Users/halim/Downloads/bmi_data_lab3_clean.csv')

print("\nNumber of NAN for each coulumn")
print('Sex : ', nan_df['Sex'].isnull().sum())
print('Age : ', nan_df['Age'].isnull().sum())
print('Height : ', nan_df['Height (Inches)'].isnull().sum())
print('Weight : ', nan_df['Weight (Pounds)'].isnull().sum())
print('BMI : ', nan_df['BMI'].isnull().sum())


print('\nNumber of rows with NAN')
row_nan = nan_df.isnull().sum(1)
print(row_nan.sum(axis=0))

print("\nExtract all rows without NAN")
print(nan_df.dropna(axis=0))


print('<fillna with mean function>\n')
print(nan_df.fillna(df.mean()))
print('\n')


print('<fillna with median function>\n')
print(nan_df.fillna(df.median()))
print('\n')


print('<fillna with ffill function>\n')
print(nan_df.fillna(method='ffill'))
print('\n')


print('<fillna with bfill function>\n')
print(nan_df.fillna(method='bfill'))
print('\n')


clean_nan_df = nan_df.dropna(axis=0)
clean_nan_df.head()
c_height = clean_nan_df['Height (Inches)'].values
c_weight = clean_nan_df['Weight (Pounds)'].values

reg = linear_model.LinearRegression()
reg.fit(c_height[:, np.newaxis], c_weight)

px = np.array([c_height.min()-1, c_height.max()+1])
py = reg.predict(px[:, np.newaxis])
plt.scatter(c_height, c_weight)
plt.plot(px, py, color='black')
plt.show()


# nan_df.head()
#
# print('--------------------------------')
# print(nan_df[(~nan_df.notnull()).any(axis=1)])
#
# df2 = nan_df[(~nan_df.notnull()).any(axis=1)]
#
# for i in range (18):
#     if
#
