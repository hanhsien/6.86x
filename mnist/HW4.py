# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 22:34:48 2019

@author: Han-Hsien.Seah
"""

import numpy as np

W = np.array([[1,0,-1],
              [0,1,-1],
              [-1,0,-1],
              [0,-1,-1]])

V = np.array([[1,1,1,1,0],
              [-1,-1,-1,-1,2]])

X = np.array([[1,4,1]]).T

Z = np.maximum(W@X,0)

U = V @ np.vstack([Z,1]) 

U = np.maximum(U,0)

O = np.exp(U) / np.sum(np.exp(U))