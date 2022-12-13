"""
Pseudospectral module.

This module provides methods for approximating functions by means of
pseudospectral collocation techniques.
"""

import numpy as np
from numpy import sqrt, exp, abs
from scipy.linalg import expm  # Matrix exponential for linear evolution


# Orthonormal Hermite functions for spectral expansion
from hermite import (hermite_nodes,
                     hermite_modified_weights,
                     hermite_matrix)


def hermite_to_collocation_matrix(n, x=None):
    """Return the matrix that transform the Hermite expansion coefficients
    into collocation values.

    TODO: documentation
    """
    if x is None:
        x = hermite_nodes(n)

    phi = hermite_matrix(n, x)
    return phi.T


def collocation_to_hermite_matrix(n):
    """ Returns the matrix that transform the collocation values into Hermite 
        expansion coefficients.
        TODO: documentation
    """    
    
    phi = hermite_matrix(n)
    
    # Prefactor matrix of modified weights (for element-wise multiplication)
    w = hermite_modified_weights(n)
    
    return phi * w

def collocation_to_interpolation_matrix(n, x):
    c2h = collocation_to_hermite_matrix(n)
    h2i = hermite_to_collocation_matrix(n, x)
    return h2i @ c2h
    

def hermite_evolution_matrix(n, a, dt, n_quad=None):
    """ Returns the multiplication matrix that transforms an initial state in 
        hermite (spectral) representation into a final state in the same 
        representation.
        TODO: documentation
    """
    
    # If the Gaussian quadrature order `n_quad` is not specified, uses 2*n 
    if n_quad is None:
        n_quad = 2 * n
    
    k, w = hermite_nodes(n_quad, weights=True)
    
    prop_k = exp(-1.0j * a(k) * dt)
    
    phi = hermite_matrix(n, k)
    
    # Intermediate matrix
    U = (phi * w * prop_k) @ (phi.T)
    
    for m in range(0, n):
        for j in range(0, n):
            U[m, j] = (-1)**j * (1.0j)**(m + j) * U[m, j]
    
    return U
            
            
def hermite_operator_matrix(n, a, n_quad=None, rep='hermite'):
    """ Returns a matrix that represents the operator given by symbol a 
        (which is a Fourier multiplier) in Hermite (spectral) representation.
        TODO: documentation
    """
    
    # If the Gaussian quadrature order `n_quad` is not specified, uses 2*n 
    if n_quad is None:
        n_quad = 2 * n
    
    k, w = hermite_nodes(n_quad, weights=True)
    
    a_k = a(k)
    
    phi = hermite_matrix(n, k)
   
    # Intermediate matrix
    U = (phi * w * a_k).dot(phi.T).astype(np.complex128)
    
    if rep == 'hermite':
        for m in range(0, n):
            for j in range(0, n):
                U[m, j] = (-1.0j)**j * (1.0j)**m * U[m, j]
    
    return U

def hermite_to_fourier_matrix(n, n_interp=10000):
            
    max_k = sqrt(2 * n + 1)
    k = np.linspace(-max_k, max_k, n_interp)

    phi = hermite_matrix(n, k).astype(np.complex128)

    for i in range(n):
        phi[i, :] = phi[i, :] * (-1.0j)**i
    return phi.T, k


def P_A_init(alpha, sigma, dim_base=100):

    def P_A(u0, dt):
        u = u0 * exp(-1.0j * dt * (alpha * abs(u0) ** (2 * sigma)))
        P_A.calls += 1
        return u
    P_A.calls = 0
    return P_A


def P_B_init(B,                # function b(k) for the pseudodifferential operator
             dim_base=100):    # dimension of the orthogonal functions set for the expansion
             
    
    h2c = hermite_to_collocation_matrix(dim_base)
    c2h = collocation_to_hermite_matrix(dim_base)
    
    C = -1.0j * hermite_operator_matrix(dim_base, B)
     
    
    # Define the linear propagator function
     
    def P_B(u0, dt):
        
        if (dt not in P_B.dt):
            P_B.dt.append(dt)
            # Exponential of C * dt solves u'(t) = C u(t)
            # TODO: Determinar por qué usar la exponencial de C * dt
            # (donde C se aproxima por cuadratura gaussiana) es mejor que 
            # usar el propagador completo obtenido por cuadratura
            expCdt = expm(C * dt)
            # expCdt = hermite_evolution_matrix(dim_base, B, dt)
                        
            P_B.propagator.append(h2c @ (expCdt @ c2h))

        
        i = P_B.dt.index(dt)
        u = P_B.propagator[i] @ u0
        P_B.calls += 1
        
        return u
    
    P_B.dt = []
    P_B.propagator = []
    P_B.calls = 0
        
    return P_B

# 8< ==========================================================================
# TODO: Esto está mal, hay que arreglarlo para que funcione con n grande

def make_c(B, dim_base):
    
    # For improved precision, we use double number of nodes for gaussian quadrature
    #x2 = hermite_nodes(dim_base * 2)

    # TODO: obtain weights in extended double precision to achieve n > 385
    #d2 = hermite_modified_weights(2 * dim_base)
    
    # Obtain linear operator B at nodes
    #b_vector = B(x2)
    
    # Intermediate matrix c = \int_{-\infty}^{\infty} b(k) \varphi_{j}(k) \varphi_{h}(k)\ dk
    #c = np.empty((dim_base, dim_base), dtype=np.double)
    
    c = hermite_operator_matrix(dim_base, B, rep='space')

    # phi = np.empty((dim_base, 2*dim_base), dtype=np.double)
    # phi[0, :] = hermite_function(0, x2)
    # phi[1, :] = hermite_function(1, x2)
        
    #for n in range(2, dim_base):
    #    phi[n, :] = (2/n)**0.5 * x2 * phi[n-1, :] - ((n-1)/n)**0.5 * phi[n-2, :]
    
    #for j in range(dim_base):
    #
    #    phi_j = phi[j, :]
    #    # c is symmetric, so we need to obtain only the diagonal and upper diagonal elements
    #    for h in range(j + 1):
    #    
    #        phi_h = phi[h, :]
    #        
    #        # Gaussian quadrature to approximate integral \int_{-\infty}^{\infty} b(k) \varphi_{j}(k) \varphi_{h}(k)\ dk
    #        c[j,h] = c[h,j] = d2.dot(b_vector * phi_j * phi_h)
    
    return(c)

def B_matrix_real(B, dim_base):
    
    phi = hermite_matrix(dim_base)
        
    U = collocation_to_hermite_matrix(dim_base)
    
    c = make_c(B, dim_base)
        
    pre = np.empty((dim_base, dim_base), dtype=np.complex128)
    for h in range(dim_base):
        for j in range(dim_base):
            pre[h, j] = (-1.0j)**(j - h)
    
    c_prime = pre * c
    
    return ((phi.T).dot((c_prime).dot(U)))

# 8< ==========================================================================
