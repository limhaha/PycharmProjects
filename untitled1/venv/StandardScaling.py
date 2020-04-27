import numpy as np

x = np.array([20, 15, 26, 32, 18, 28, 35, 14, 26, 22, 17])

mean = round(np.mean(x), 1)
print('Mean : ', mean)
print('')

standard = round(np.std(x), 1)
print('Standard deviation : ', standard)
print('')

score = ((x - mean) / standard).round(2)
print('Standard scores : ', score)
print('')

count = 0

print('Received F student scores')
for i in range(11):
    if (score[i] < -1):
        print(x[i])



