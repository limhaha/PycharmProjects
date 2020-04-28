import numpy as np
import matplotlib.pyplot as plt

wt = np.random.randint(40, 90, size=100)
ht = np.random.randint(140, 200, size=100)

BMI = wt / ((ht / 100) * (ht / 100))

print(BMI)

for i in range(10):
    print(round(BMI[i], 1))


underweight_wt = []
healthy_wt = []
overweight_wt = []
obese_wt = []

underweight_ht = []
healthy_ht = []
overweight_ht = []
obese_ht = []


for i in range(100):
    if BMI[i] < 18.5:
        underweight_wt.append(wt[i])
        underweight_ht.append(ht[i]/100)

    elif 18.5 <= BMI[i] < 25:
        healthy_wt.append(wt[i])
        healthy_ht.append(ht[i]/100)

    elif 25 <= BMI[i] < 30:
        overweight_wt.append(wt[i])
        overweight_ht.append(ht[i]/100)

    else:
        obese_wt.append(wt[i])
        obese_ht.append(ht[i]/100)

# # Boxplot
# plotData_wt = [underweight_wt, healthy_wt, overweight_wt, obese_wt]
# plt.xlabel('BMI status')
# plt.ylabel('Weight (kg)')
# plt.boxplot(plotData_wt, labels=["underweight", "healthy", "overweight", "obese"])
# plt.show()
#
# plotData_wt = [underweight_ht, healthy_ht, overweight_ht, obese_ht]
# plt.xlabel('BMI status')
# plt.ylabel('Height (m)')
# plt.boxplot(plotData_wt, labels=["underweight", "healthy", "overweight", "obese"])
# plt.show()
#
#
# # Bar chart
# dist = ["underweight", "healthy", "overweight", "obese"]
# bmi = [len(underweight_ht), len(healthy_ht), len(overweight_ht), len(obese_ht)]
# plt.bar(dist, bmi)
# plt.xlabel('BMI status')
# plt.ylabel('Number of student')
# plt.show()


# Histogram
plt.hist(underweight_ht)
plt.title('underweight')
plt.xlabel('BMI status')
plt.ylabel('Number of student')
plt.show()

plt.hist(healthy_ht)
plt.title('healthy')
plt.xlabel('BMI status')
plt.ylabel('Number of student')
plt.show()

plt.hist(overweight_ht)
plt.title('overweight')
plt.xlabel('BMI status')
plt.ylabel('Number of student')
plt.show()

plt.hist(obese_ht)
plt.title('obese')
plt.xlabel('BMI status')
plt.ylabel('Number of student')
plt.show()


# # Pie chart
# plt.pie(bmi, labels=dist, autopct='%1.2f%%')
# plt.show()
#
#
# # Scatter plot
# plt.scatter(wt, ht/100, color='r')
# plt.xlabel('Weight (kg)')
# plt.ylabel('Height (m)')
# plt.show()