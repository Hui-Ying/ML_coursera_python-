#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import plotData as pd
import matplotlib.pyplot as plt


# ==================== Part 1: Basic Function ====================
# Complete warmUpExercise.m
print ("Running warmUpExercise ... \n")
print ('5x5 Identity Matrix: \n')
import warmUpExercise
input("Program paused. Press enter to continue.\n")

# ======================= Part 2: Plotting =======================
print('Plotting Data... \n')
data = np.loadtxt(fname = "ex1data1.txt", delimiter="," )

X = data[:,0]
y = data[:,1]

m = len(y)  # number of training examples
#Plot Data
# Note: You have to complete the code in plotData.py

input("Program paused. Press enter to continue.\n")

# =================== Part 3: Cost and Gradient descent ===================

X = np.column_stack((np.ones((m,1)), X)) # Add a column of ones to x

# Some gradient descent settings
iterations = 1500;
alpha = 0.01;
print ('\nTesting the cost function ...\n')

# compute and display initial cost
import computeCost
plt.show()
input("Program paused. Press enter to continue.\n")
print ('\nRunning Gradient Descent ...\n')

import gradientDescent
print ('Expected theta values (approx)')
print (' -3.6303  1.1664\n\n')





