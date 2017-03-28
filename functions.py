# Functions for neural networks

import numpy as np

def fun1(x,B):
    return 1/(1+np.exp(-B*x))

def dfun1(x,B):
    return -B*np.exp(-B*x)/(1+np.exp(-B*x))**2

def fun2(x,B):
    return 2/(1+np.exp(-B*x))-1

def dfun2(x,B):
    return -2*B*np.exp(-B*x)/(1+np.exp(-B*x))**2

def fun3(x,B):
    return np.tanh(B*x)

def dfun3(x,B):
    return B*(1-np.tanh(B*x)**2)