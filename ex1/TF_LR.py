#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
rng = np.random

data = np.loadtxt(fname = "ex1data1.txt", delimiter="," ) 

# Parameters
learning_rate = 0.01
training_epochs = 1500
display_step = 50

# Training Data
train_X = data[:,0]
train_Y = data[:,1]
m = train_X.shape[0] # .shape here returns the dimensions of train_X

# tf Graph input
X = tf.placeholder("float") # tf.placeholder is used to feed actual training examples. 
Y = tf.placeholder("float")

# Set model weights
W = tf.Variable(rng.randn(), name = "weight") # tf.Variable is for trainable variables 
b = tf.Variable(rng.randn(), name = "bias")   #such as W and b

# Hypothsis
hyp = tf.add(tf.multiply(X,W), b)  

# Cost function(Mean square error)
cost = tf.reduce_sum(tf.pow(Y - hyp,2))/(2*m)

# .minimize(cost) will minimize the parameters(W,b) automatically
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables
init = tf.global_variables_initializer()

# Start training
#The 'with' block terminates the session as soon 
#as the operations are completed.
# tf.Session() initiates a TensorFlow Graph object 
#in which tensors are processed through operations (or ops).
with tf.compat.v1.Session() as sess:
     # Run the initializer
     sess.run(init)
     #Fit all training data
     for epoch in range(training_epochs):
         for(x,y) in zip(train_X, train_Y):
             sess.run(optimizer, feed_dict = {X:x, Y:y})
             
         if (epoch + 1) % display_step == 0:
             c = sess.run(cost,feed_dict = {X:train_X, Y:train_Y})
             print("Epoch : ", '%0.4d' % (epoch + 1), "Cost = ", "{:.9f}".format(c),\
                   "W=", sess.run(W), "b=", sess.run(b))
     
     print("Finished!")
     training_cost = sess.run(cost, feed_dict = {X: train_X, Y: train_Y})
     print("Training cost = ", training_cost, "W=", sess.run(W), "b=", sess.run(b),'\n')      
         
     plt.plot(train_X, train_Y, 'ro', label='Original Data')
     plt.plot(train_X, sess.run(W)*train_X +sess.run(b), label = 'Fitted line')
     plt.legend()
     plt.show()
            
     
     
     
     
     
     