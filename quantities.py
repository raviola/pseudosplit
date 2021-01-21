"""
Quantities module.

This module provides functions for calculating important quantities (Lp norms, center of mass, variance, energy, Fourier and Hermite coefficients)
using Hermite-Gauss quadrature formulas.

"""
import numpy as np
from numpy import abs, zeros
from pseudospectral import hermite_projection_matrix, B_matrix
from base_functions import hermite_function

# Lp-norm of function u calculated by Hermite-Gass quadrature
# d_i = w_i * exp(x_i ** 2) are the modified weights 
# given w_i (weights) and x_i (nodes)

def norm(u, d, n=2):
        
    return(((abs(u)**n).dot(d))**(1./n))

def center_of_mass(u, d, x):
    
    return((x * abs(u)).dot(d))

def variance(u, d, x):
    CM = center_of_mass(u, d, x)
    
    v = zeros(CM.size)
    
    for j, CM_j in enumerate(CM):
        v[j] = ((x - CM_j)**2 * abs(u[j,:])).dot(d)
        
    return(v)

def localized_2norm(u, d, x, r):
    
    d_aux = abs(x) < r
    
    d_aux = d_aux * d
    
    return((abs(u)).dot(d_aux))

def hermite_coeffs(u):
    
    _, dim_base = u.shape
    U = hermite_projection_matrix(dim_base)
    
    return(u.dot(U.transpose()))

def fourier_coeffs(u, kmax=20, numpoints=100):
    _, dim_base = u.shape
    
    k = np.linspace(-kmax, kmax, numpoints)
    
    phi = np.empty((dim_base, k.size), dtype=np.clongdouble)
    for j in range(dim_base):
        phi[j,:] = (-1.0j)**j * hermite_function(j, k)
    
    beta = hermite_coeffs(u)
    
    return(beta.dot(phi))

def infinite_norm(u):
    return(np.max(u, axis=1))


def kinetic_energy(u, B):
    
    dim_time, dim_base = u.shape
    
    beta = hermite_coeffs(u)
    
    prefactor = np.empty((dim_base, dim_base), dtype=np.clongdouble)

    KE = np.zeros(dim_time, dtype=np.longdouble)
    
    for h in range(dim_base):
        for j in range(dim_base):
            prefactor[h, j] = (-1.0j)**h * (1.0j)**j
    
    c = B_matrix(B, dim_base)
    C = prefactor * c
    
    for i in np.arange(dim_time):
    
        KE[i] = beta[i,:].dot(C.dot(beta[i,:].conj()))
    
    return(KE)

    


    
    