# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 23:39:59 2019

@author: Han-Hsien.Seah
"""

import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

Wfh = 0
Wih = 0
Woh = 0
Wfx = 0
Wix = 100
Wox = 100
bf = -100
bi = 100
bo = 0
Wch = -100
Wcx = 50
bc = 0

h = 0
c = 0
#x = [0,0,1,1,1,0]
x = [1,1,0,1,1]

def LSTM(c, h, x):
    f = sigmoid(Wfh*h + Wfx*x + bf)
    i = sigmoid(Wih*h + Wix*x + bi)
    o = sigmoid(Woh*h + Wox*x + bo)
    c = f*c + i*np.tanh(Wch*h+Wcx*x+bc)
    h = o * np.tanh(c)
    h = np.round(h)

    return c, h


for i in range(len(x)):
    c, h = LSTM(c, h, x[i])
    print(c, h)


