# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:02:47 2019

@author: Han-Hsien.Seah
"""

import numpy as np
Y = np.empty((4,3))
Y[:] = np.nan
Y[0][0] = 5
Y[0][2] = 7
Y[1][1] = 2
Y[2][0] = 4
Y[3][1] = 3
Y[3][2] = 6

U = np.array([[6, 0, 3, 6]]).T
V = np.array([[4, 2, 1]]).T

X = U@V.T

sq_err = 1/2* np.nansum((Y-X)**2)
print(sq_err)

reg = np.sum(U**2/2)+np.sum(V**2/2)