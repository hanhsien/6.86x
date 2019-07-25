# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import gzip
import _pickle

with gzip.open('../Datasets/mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = _pickle.load(f)
    
train_x, train_y = train_set


import matplotlib.cm as cm
import matplotlib.pyplot as plt


plt.imshow(X_train[0:10].reshape((42, 28)), cmap=cm.Greys_r)
plt.show()