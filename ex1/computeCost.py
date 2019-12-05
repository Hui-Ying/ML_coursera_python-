#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
import array as arr

data = np.loadtxt(fname = "ex1data1.txt", delimiter="," )
X = data[:,0]

y = data[:,1]
m = len(y)  # number of training examples
X = np.column_stack((np.ones((m,1)), X))

theta = np.zeros((2,1)) # initialize fitting parameters

# You need to return the following variables correctly 
J = 0

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta
#               You should set J to the cost.
#print computeCost(X, y, theta)
#theta = np.zeros((2,1))
h = np.mat(X)*np.mat(theta)
h = h.transpose()

J = (1/(2*m))*np.sum(np.array(h-y)**2)
print ('With theta = [0 ; 0]\nCost computed =', J)
print ('Expected cost value (approx) 32.07\n')
print(h)
input("Program paused. Press enter to continue.\n")

# Further testing of the cost function
theta = np.array([[-1],[2]])
h = np.mat(X)*np.mat(theta)
h = h.transpose()
J = (1/(2*m))*np.sum(np.array(h-y)**2)
print('\nWith theta = [-1 ; 2]\nCost computed =', J)
print ('Expected cost value (approx) 54.24\n')
