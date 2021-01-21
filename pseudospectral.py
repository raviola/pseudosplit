"""
Pseudospectral module.

This module provides methods for approximating functions by means of pseudospectral collocation techniques.
"""

import numpy as np
from numpy.polynomial.hermite import hermgauss  # Weights and nodes for Gauss-Hermite quadrature
from numpy import exp, abs
from scipy.linalg import expm  # Matrix exponential for linear evolution (from SciPy)

from base_functions import hermite_function # Orthonormal Hermite functions for spectral expansion

def P_A_init(alpha, sigma):
    
    def P_A(u0, dt):
        
        u = u0 * exp(-1.0j *  alpha * abs(u0) ** (2 * sigma) * dt)
        
        return u
    
    return P_A
    
def P_B_init(B,                # function b(k) for the pseododifferential operator
             dim_base=100):    # dimension of the orthogonal functions set for the expansion
             
    # Projection matrix U: u(x_i) --> \beta_j (Hermite expansion coefficients)
    # Given u at nodes x_i, U.dot(u) returns the Hermite expansion coefficients 

    x, w = hermgauss(dim_base) # nodes and weights of the Hermite-Gauss quadrature
    d = w * exp(x ** 2)        # modified weights of the Hermite-Gauss quadrature
    
    # Matrix of Hermite functions evaluated at nodes 
    # phi[j,l] is the value of Hermite function j at node x_l
    # Its transpose converts \beta_j --> u(x_i)
    
    phi = np.empty((dim_base, dim_base), dtype=np.longdouble)

    for j in range(dim_base):
        phi[j,:] = hermite_function(j, x)
        
    
    U = hermite_projection_matrix(dim_base)
    
    # ===============================================================
    # Define C, the matrix for the linear evolution problem u' = C u
    # from the symbol of the linear operator B
    # ===============================================================
    
    c = B_matrix(B, dim_base)
            
    # Prefactor for c to obtain C 
    pre = np.empty((dim_base, dim_base), dtype=np.clongdouble)
    for h in range(dim_base):
        for j in range(dim_base):
            pre[h, j] = (-1.0j)**(j - h + 1)
    
    C = pre * c
    
    # Define the linear propagator function
    def P_B(u0, dt):
        
        if (P_B.dt != dt):
            P_B.dt = dt
            # Exponential of C * dt solves u'(t) = C u(t)
            P_B.expCdt = expm(C * dt)
            
        # Obtain the coefficients of u0 in the Hermite function basis
        beta = U.dot(u0)
        
        # Obtain the new coefficients after a linear evolution with time dt
        # This is the linear integration step
        beta = P_B.expCdt.dot(beta) # beta + dt * C.dot(beta)
        
        # Back to the initial space (synthesis)
        # u = sum(n) (beta[n] * phi[n,:])
        u = beta.dot(phi)
        
        return u
    
    P_B.dt = 0.0
        
    return P_B


def hermite_projection_matrix(dim_base):
    
    x, w = hermgauss(dim_base) # nodes and weights of the Hermite-Gauss quadrature
    d = w * exp(x ** 2)        # modified weights of the Hermite-Gauss quadrature
    
    # Matrix of Hermite functions evaluated at nodes 
    # phi[j,l] is the value of Hermite function j at node x_l
    # Its transpose converts \beta_j --> u(x_i)
    
    phi = np.empty((dim_base, dim_base), dtype=np.longdouble)

    for j in range(dim_base):
        phi[j,:] = hermite_function(j, x)
    
    # Prefactor matrix of modified weights (for element-wise multiplication)
    D = np.diag(d)

    # Projection matrix U: u(x_i) --> \beta_j (Hermite expansion coefficients)
    # Given u at nodes x_i, U.dot(u) returns the Hermite expansion coefficients 

    U =  phi.dot(D)
    
    return(U)

def B_matrix(B, dim_base):
    
    # For improved precision, we use double number of nodes for gaussian quadrature
    x2, w2 = hermgauss(dim_base * 2)
    d2 = w2 * exp(x2 ** 2)
    
    # Obtain linear operator B at nodes
    b_vector = B(x2)
    
    # Intermediate matrix c = \int_{-\infty}^{\infty} b(k) \varphi_{j}(k) \varphi_{h}(k)\ dk
    c = np.zeros((dim_base, dim_base), dtype=np.longdouble)

    for j in range(dim_base):
    
        phi_j = hermite_function(j, x2)
    
        # c is symmetric, so we need to obtain only the diagonal and upper diagonal elements
        for h in range(j + 1):
        
            phi_h = hermite_function(h, x2)
            
            # Gaussian quadrature to approximate integral \int_{-\infty}^{\infty} b(k) \varphi_{j}(k) \varphi_{h}(k)\ dk
            c[j,h] = c[h,j] = d2.dot(b_vector * phi_j * phi_h)
    
    return(c)
            
    
    
    
    