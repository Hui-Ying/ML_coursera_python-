#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pylab
import numpy as np
import matplotlib.pyplot as plt
iterations = 1500;
alpha = 0.01;

data = np.loadtxt(fname = "ex1data1.txt", delimiter="," )
X = data[:,0]
y = data[:,1]
m = len(y) # number of training examples
X = np.column_stack((np.ones((m,1)), X))
theta = np.zeros((2,1))

# Performs gradient descent to learn theta
# Updateing theta by taking number of iterations  
# gradient steps with learning rate alpha

for i in range(iterations): 
    
    h = np.mat(X)*np.mat(theta)
    h = h.transpose()

    J1 = np.sum((np.transpose(X[:,0]))*np.array(h-y))
    J2 = np.sum((np.transpose(X[:,1]))*np.array(h-y))

    theta[0,0] -= alpha*J1/m
    theta[1,0] -= alpha*J2/m
    
# print theta to screen
print ('Theta found by gradient descent:\n',theta[0,0], theta[1,0])

# Plot the linear fit
X2 = data[:,1]
plt.plot(X,y,'rx', markersize=10)
plt.plot(X2,y,'-r')
pylab.xlim(4, 24)
