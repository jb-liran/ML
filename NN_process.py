#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 13:32:08 2018

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
w33 = 0.1
w44 = 0.1
w55 = 0.1


#xvals = [0,5,6,1]
#yvals = [0,2,7,1]
#expected = [0,0.6,0.3,1]


xvals = [0,0,1,1]
yvals = [0,1,0,1]
expected = [0,1,1,0]
bias = 1

it = 0
item = 0
flag_print = -1

while True: 
    # forward propogation
    N3val = xvals[item] * w1 + yvals[item] * w3 + bias * w33
    N4val = xvals[item] * w2 + yvals[item] * w4 + bias * w44
    
    N3Out = sigmoid(N3val)
    N4Out = sigmoid(N4val)
    
    N5val = N3Out * w5 + N4Out * w6 + bias * w55
    
    N5Out = sigmoid(N5val)
    
    error = expected[item] - N5Out
    
    if it % 100 == 0:
        flag_print = 0
        time.sleep(1)
        print("===========")
    
    if flag_print >= 0:    
        print (N5Out)
        flag_print +=1
        if flag_print == 4:
            flag_print = -1
    
    
    # back prop
    
    
    learning = 0.3
    
    e5 = learning * error * N3Out * sigmoid_der(N5Out)
    e6 = learning * error * N4Out * sigmoid_der(N5Out)

    e55 = learning * error * bias * sigmoid_der(N5Out)
    
    w5 += e5
    w6 += e6
    w55 += e55
    
    e1 = learning * error * xvals[item] * sigmoid_der(N3Out)
    e2 = learning * error * xvals[item] * sigmoid_der(N4Out)
    e3 = learning * error * yvals[item] * sigmoid_der(N3Out)
    e4 = learning * error * yvals[item] * sigmoid_der(N4Out)
    
    e33 = learning * error * bias * sigmoid_der(N3Out)
    e44 = learning * error * bias * sigmoid_der(N4Out)
    
    
    w1 += e1
    w2 += e2
    w3 += e3
    w4 += e4
    
    w33 += e33
    w44 += e44
    
    it+=1
    item += 1
    if item == 4:
        item = 0





