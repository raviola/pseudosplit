#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 14:44:14 2022

@author: lisandro
"""
import numpy as np
from scipy.optimize import newton_krylov

from pseudospectral import B_matrix_real
from hermite import hermite_function, hermite_nodes

omega = 0.0625
alpha = -1.
n = 1000
s = 0.8

def a(k):
        return (np.abs(k)**(2*s))

A = B_matrix_real(a, n)

def F(Q):
    F_Q = (A + omega * np.diag(np.ones(n))).dot(Q) + alpha * np.abs(Q)**2 * Q
    return F_Q

x = hermite_nodes(n, weights=False)
Q_0 = hermite_function(0, x)

Q_f = newton_krylov(F, Q_0)

