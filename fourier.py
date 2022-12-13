#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 20:44:39 2022

@author: lisandro
"""
import numpy as np
from numpy import exp, abs, sqrt, pi
from scipy.fft import fft, ifft, fftfreq, fftshift

def P_A_init(alpha, sigma):
    
    
    def P_A(u0, dt):
        
        u = u0 * exp(-1.0j * dt * (alpha * abs(u0) ** (2 * sigma)))
        P_A.calls += 1
        return u
    P_A.calls = 0
    return P_A

def P_B_init(B,         # function b(k) for the pseudodifferential operator
             N=2**10,   # number of Fourier modes       
             P=100):    # spatial period

              
    
    # Define the linear propagator function
     
    def P_B(u0, dt):
        
        k = 2 * np.pi * fftfreq(N, P/N)
        propagator = np.exp(-1.0j * B(k) * dt)
    
        u0_til = 1 / N * fft(u0)
        u0_til_evol = propagator * u0_til
        u = N * ifft(u0_til_evol)
        
        P_B.calls += 1
        
        return u
    
    P_B.dt = []
    P_B.propagator = []
    P_B.calls = 0
        
    return P_B