"""
Quantities module.

This module provides functions for calculating important quantities (Lp norms, center of mass, variance, energy, Fourier and Hermite coefficients) using Hermite-Gauss quadrature formulas.

"""
import numpy as np
from numpy import abs, zeros
from pseudospectral import (hermite_to_fourier_matrix,
                            collocation_to_hermite_matrix,
                            hermite_operator_matrix)

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
    U = collocation_to_hermite_matrix(dim_base)
    
    return(u.dot(U.T))

def fourier_coeffs(u, kmax=None, numpoints=1000):
    dim_time, dim_base = u.shape
        
    A = np.empty((dim_time, numpoints), dtype=np.complex128)
   
    h2f, k = hermite_to_fourier_matrix(dim_base, numpoints)
    
    c2h = collocation_to_hermite_matrix(dim_base)
    
    h_t = u.dot(c2h.T)
    
    for i in range(dim_time):
        A[i,:] = h2f.dot(h_t[i])
     
    return A, k

def infinite_norm(u):
    return(np.max(np.abs(u), axis=1))


def kinetic_energy(u, B):
    
    dim_time, dim_base = u.shape
    
    beta = hermite_coeffs(u)
    

    KE = np.zeros(dim_time, dtype=np.double)
        
    C = hermite_operator_matrix(dim_base, B, rep='hermite')
    
    
    for i in np.arange(dim_time):
    
        KE[i] = beta[i,:].dot(C.dot(beta[i,:].conj()))
    
    return(KE)

    


    
    