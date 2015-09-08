#encoding: utf-8

import numpy as np
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

auth = np.genfromtxt('../src/CodeIQ_auth.txt',delimiter=' ')

# training
train_X = np.array([[x[0],x[1]] for x in auth])
# label in training
labels = [int(x[2]) for x in auth]

# test
test_X = np.genfromtxt('../src/CodeIQ_mycoins.txt',delimiter=' ')

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)


# 教師データから正解を抽出
correct = np.array([[x[0], x[1]] for x in auth if x[2] == 1]).T
# 同じくニセモノを抽出
wrong   = np.array([[x[0], x[1]] for x in auth if x[2] == 0]).T

# これらを散布図にプロッティングする
ax1.scatter(correct[0], correct[1], color='g')
ax1.scatter(wrong[0],   wrong[1],   color='r')
ax2.scatter(train_X.T[0], train_X.T[1], color='b')
ax2.scatter(test_X.T[0],  test_X.T[1],  color='r')

plt.legend(loc='best')
plt.savefig("image.png")
