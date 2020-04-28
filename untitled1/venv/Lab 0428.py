import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

df = pd.read_csv('/Users/halim/Downloads/bmi_data_lab3.csv')



# extremelyweak_wt = []
# weak_wt = []
# normal_wt = []
# overweight_wt = []
# obesity_wt = []
#
# extremelyweak_ht = []
# weak_ht = []
# normal_ht = []
# overweight_ht = []
# obesity_ht = []

bit_mask0 = df['BMI'] == 0
df_eweak = df[bit_mask0]

bit_mask1 = df['BMI'] == 1
df_weak = df[bit_mask1]

bit_mask2 = df['BMI'] == 2
df_normal = df[bit_mask2]

bit_mask3 = df['BMI'] == 3
df_overweight = df[bit_mask3]

bit_mask4 = df['BMI'] == 4
df_obesity = df[bit_mask4]


print(df_normal['Height (Inches)'])


# plt.hist(df_eweak['Height (Inches)'], bins=10)
# plt.title('extremely weak')
# plt.xlabel('Height (Inches)')
# plt.ylabel('Number of student')
# plt.show()
#
# plt.hist(df_weak['Height (Inches)'], bins=10)
# plt.title('extremely weak')
# plt.xlabel('Height (Inches)')
# plt.ylabel('Number of student')
# plt.show()
#
# plt.hist(df_normal['Height (Inches)'], bins=10)
# plt.title('extremely weak')
# plt.xlabel('Height (Inches)')
# plt.ylabel('Number of student')
# plt.show()
#
# plt.hist(df_overweight['Height (Inches)'], bins=10)
# plt.title('extremely weak')
# plt.xlabel('Height (Inches)')
# plt.ylabel('Number of student')
# plt.show()
#
# plt.hist(df_obesity['Height (Inches)'], bins=10)
# plt.title('extremely weak')
# plt.xlabel('Height (Inches)')
# plt.ylabel('Number of student')
# plt.show()

ht_wt = {'height' : df['Height (Inches)'], 'weight' : df['Weight (Pounds)']}
ht_wt_df = pd.DataFrame(ht_wt)

standard_scaler = preprocessing.StandardScaler()
standard_scaler.fit(ht_wt_df)
standard_scaler.transform(ht_wt_df)