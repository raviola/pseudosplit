#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 23:50:11 2022

@author: lisandro
"""

def neri_step(u, P_A, P_B, dt):

    c1 = c4 = 1 / 2 / (2 - 2**(1 / 3)) * dt
    c2 = c3 = (1 - 2 ** (1 / 3)) / 2 / (2 - 2**(1 / 3)) * dt
    d1 = d3 = 1 / (2 - 2**(1 / 3)) * dt
    d2 = -2 ** (1 / 3) / (2 - 2 ** (1 / 3)) * dt
    d4 = 0.

    u1 = P_A(u, c1)
    u2 = P_B(u1, d1)
    u3 = P_A(u2, c2)
    u4 = P_B(u3, d2)
    u5 = P_A(u4, c3)
    u6 = P_B(u5, d3)
    u7 = P_A(u6, c4)

    return u7

def strang_step(u, P_A, P_B, dt):
    u1 = P_A(u, dt/2)
    u2 = P_B(u1, dt)
    u3 = P_A(u2, dt/2)
    return u3

def ruth_step(u, P_A, P_B, dt):

    c1 = 1. * dt
    c2 = - 2 / 3. * dt
    c3 = 2 / 3. * dt
    d1 = - 1 / 24. * dt
    d2 = 3 / 4. * dt
    d3 = 7 / 24. * dt

    u1 = P_A(u, c1)
    u2 = P_B(u1, d1)
    u3 = P_A(u2, c2)
    u4 = P_B(u3, d2)
    u5 = P_A(u4, c3)
    u6 = P_B(u5, d3)

    return u6
