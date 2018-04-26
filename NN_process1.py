#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 21:57:52 2018

@author: liran
"""

'''
x,y - inputs layer
N3, N4 - hidden layer
N5 - output

x - w1 --> N3
  \      /   \
   w2   /     w5
    \  /       \
     \/         N5 ---> Output
     /\         /
    /  \       /
   w3   \     w6
  /      \   /
y - w4 --> N4

'''

import time
import numpy as np

def sigmoid(x):
    return 1 / ( 1 + np.exp(-x) )

def sigmoid_der(x):
    return x * (1 - x)

w1 = 0.1
w2 = 0.2
w3 = 0.3
w4 = 0.4
w5 = 0.5
w6 = 0.6

xvals = [0,0,1,1]
yvals = [0,1,0,1]
expected = [0,1,1,1]

it = 0
item = 0
flag_print = -1

while True: 
    # forward propogation
    N1val = xvals[item] * w1 + yvals[item] * w3 
    N2val = xvals[item] * w2 + yvals[item] * w4 
    
    N1Out = sigmoid(N1val)
    N2Out = sigmoid(N2val)
    
    N3val = N1Out * w5 + N2Out * w6 
    
    N3Out = sigmoid(N3val)
    
    error = expected[item] - N3Out
    
    if it % 1000 == 0:
        flag_print = 0
        print("============")
        time.sleep(1)
    
    if flag_print >= 0:    
        print (N3Out)
        flag_print +=1
        if flag_print == 4:
            flag_print = -1
    
    
    # back prop
    
    
    learning = 0.1
    
    e5 = learning * error * N1Out * sigmoid_der(N3Out)
    e6 = learning * error * N2Out * sigmoid_der(N3Out)

    
    w5 += e5
    w6 += e6
    
    e1 = learning * error * xvals[item] * sigmoid_der(N1Out)
    e2 = learning * error * xvals[item] * sigmoid_der(N2Out)
    e3 = learning * error * yvals[item] * sigmoid_der(N1Out)
    e4 = learning * error * yvals[item] * sigmoid_der(N2Out)
    
    
    
    w1 += e1
    w2 += e2
    w3 += e3
    w4 += e4
    
    
    it+=1
    item += 1
    item = item % 4





