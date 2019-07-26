# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import gzip
import pickle
import numpy

with gzip.open('../Datasets/mnist.pkl.gz', 'rb') as f:
    u = pickle.load(f, encoding = 'latin1')
    print(u)
    train_set, valid_set, test_set = u
    
train_x, train_y = train_set


import matplotlib.cm as cm
import matplotlib.pyplot as plt


plt.imshow(train_x[0].reshape((28, 28)), cmap=cm.Greys_r)
plt.show()




f = gzip.open(path_to_data_dir + 'train_multi_digit' + '_mini' + '.pkl.gz', 'rb')
X_train = _pickle.load(f, encoding='latin1')

f = gzip.open('../Datasets/mnist.pkl.gz', 'rb')
X_train = _pickle.load(f, encoding='latin1')
train_x, train_y = train_set

plt.imshow(train_x[0].reshape((28, 28)), cmap=cm.Greys_r)
plt.show()