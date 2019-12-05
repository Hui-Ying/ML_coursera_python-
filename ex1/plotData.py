#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


data1 = np.loadtxt(fname = "ex1data1.txt", delimiter="," )
X = data1[:,0]
y = data1[:,1]
m = len(y)  # number of training examples

plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.plot(X,y,'rx', markersize=10)

