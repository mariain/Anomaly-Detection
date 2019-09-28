#!/usr/bin/env python
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import estimategaussian as eg
import multivariategaussian as mvg
import visualizefit as vf
import selectthreshold as st

print('Visualizing example dataset for outlier detection.\n');

mat = scipy.io.loadmat('ex8data1.mat')
X = mat["X"]
Xval = mat["Xval"]
yval = mat["yval"].flatten()

#  Visualize the example dataset
plt.plot(X[:, 0], X[:, 1], 'bx', markersize=10, markeredgewidth=1)
plt.axis([0,30,0,30])
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.show(block=False)
input('Program paused. Press enter to continue.')

print('Visualizing Gaussian fit.\n')

mu, sigma2 = eg.estimategaussian(X)
p = mvg.multivariategaussian(X, mu, sigma2)

plt.close()
vf.visualizefit(X, mu, sigma2)
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.show(block=False)
input('Program paused. Press enter to continue.')

pval = mvg.multivariategaussian(Xval, mu, sigma2)

epsilon, F1 = st.selectthreshold(yval, pval)
print('Best epsilon found using cross-validation: {:e}'.format(epsilon))
print('Best F1 on Cross Validation Set:  {:f}'.format(F1))
print('   (you should see a value epsilon of about 8.99e-05)\n')
print('   (you should see a Best F1 value of  0.875000)\n\n');

# Find the outliers in the training set and plot the
outliers = p < epsilon

# interactive graphs
plt.ion()

#  Draw a red circle around those outliers

# plt.scatter(X[outliers, 0], X[outliers, 1], s=325, facecolors='none', edgecolors='r')
plt.plot(X[outliers, 0], X[outliers, 1], 'ro', linewidth=2, markersize=18, fillstyle='none', markeredgewidth=1)
plt.show(block=False)

input('Program paused. Press enter to continue.')

mat = scipy.io.loadmat('ex8data2.mat')
X = mat["X"]
Xval = mat["Xval"]
yval = mat["yval"].flatten()

#  Apply the same steps to the larger dataset
mu, sigma2 = eg.estimategaussian(X)

#  Training set
p = mvg.multivariategaussian(X, mu, sigma2)

#  Cross-validation set
pval = mvg.multivariategaussian(Xval, mu, sigma2)

#  Find the best threshold
epsilon, F1 = st.selectthreshold(yval, pval)

print('Best epsilon found using cross-validation: {:e}'.format(epsilon))
print('Best F1 on Cross Validation Set:  {:f}'.format(F1))
print('   (you should see a value epsilon of about 1.38e-18)\n')
print('   (you should see a value epsilon of about 1.38e-18)\n');
print('   (you should see a Best F1 value of 0.615385)\n');
print('# Outliers found: {:d}'.format(np.sum((p < epsilon).astype(int))))