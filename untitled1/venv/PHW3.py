import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.linear_model as lm
import sklearn.preprocessing as pr

df = pd.read_excel('bmi_data_phw3.xlsx')

#Print dataset statistical data, feature names & data types

# print(df)
#
# print(df.info())

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

# Plot height & weight histograms (bins=10) for each BMI value
# Using matplotlib.pyplot.subplots
fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
axs = [ax0, ax1, ax2]

axs[0].hist(df_eweak['Height (Inches)'], bins=10)
axs[0].set_title('bmi = 0')
axs[1].hist(df_weak['Height (Inches)'], bins=10)
axs[1].set_title('bmi = 1')
axs[2].hist(df_normal['Height (Inches)'], bins=10)
axs[2].set_title('bmi = 2')
plt.show()

fig, (ax3, ax4) = plt.subplots(1, 2)
axs = [ax3, ax4]
axs[0].hist(df_overweight['Height (Inches)'], bins=10)
axs[0].set_title('bmi = 3')
axs[1].hist(df_obesity['Height (Inches)'], bins=10)
axs[1].set_title('bmi = 4')
plt.show()



# Plot scaling results for height and weight
df = pd.DataFrame({
    'x1': df['Height (Inches)'], 'x2': df['Weight (Pounds)'], 'x3': df['BMI']
})

# StandardScaler
scaler = pr.StandardScaler()
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df,
                         columns=['x1', 'x2', 'x3'])
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))
ax1.set_title('Before Scaling')
sns.kdeplot(df['x1'], ax=ax1)
sns.kdeplot(df['x2'], ax=ax1)
sns.kdeplot(df['x3'], ax=ax1)
ax2.set_title('After Standard Scaler')
sns.kdeplot(scaled_df['x1'], ax=ax2)
sns.kdeplot(scaled_df['x2'], ax=ax2)
sns.kdeplot(scaled_df['x3'], ax=ax2)
plt.show()


# MinMaxScaler
scaler = pr.MinMaxScaler()
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df,
                         columns=['x1', 'x2', 'x3'])
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))
ax1.set_title('Before Scaling')
sns.kdeplot(df['x1'], ax=ax1)
sns.kdeplot(df['x2'], ax=ax1)
sns.kdeplot(df['x3'], ax=ax1)
ax2.set_title('After MinMax Scaler')
sns.kdeplot(scaled_df['x1'], ax=ax2)
sns.kdeplot(scaled_df['x2'], ax=ax2)
sns.kdeplot(scaled_df['x3'], ax=ax2)
plt.show()


# RobustScaler
scaler = pr.RobustScaler()
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df,
                         columns=['x1', 'x2', 'x3'])
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))
ax1.set_title('Before Scaling')
sns.kdeplot(df['x1'], ax=ax1)
sns.kdeplot(df['x2'], ax=ax1)
sns.kdeplot(df['x3'], ax=ax1)
ax2.set_title('After Robust Scaler')
sns.kdeplot(scaled_df['x1'], ax=ax2)
sns.kdeplot(scaled_df['x2'], ax=ax2)
sns.kdeplot(scaled_df['x3'], ax=ax2)
plt.show()




D = pd.read_excel('bmi_data_phw3.xlsx')

# Read the Excel dataset file
height = np.array(D['Height (Inches)'])
weight = np.array(D['Weight (Pounds)'])


# compute the linear regression equation E for the input dataset D
E = lm.LinearRegression()
E.fit(height[:, np.newaxis], weight)

# For (height h, weight w) of each record, compute e=w-w’, where w’ is obtained for h using E
e = weight - E.predict(weight[:, np.newaxis])

z = pr.scale(e)

# draw a histogram
plt.hist(z, bins=10, width=0.4)
plt.xlabel('Ze')
plt.ylabel('frequency')
plt.show()


