"""
Base functions module.
Currently only implements Hermite functions (orthonormal basis).
"""

# Polinomios de Hermite
from numpy.polynomial import Hermite  # Polinomios de Hermite (sin normalizar)


# Funciones matemáticas generales (vectoriales)
from numpy import pi, sqrt, exp
from scipy.special import factorial

### Definiciones generales para la base de funciones

# Polinomio de Hermite de orden n (no normalizado) 
def hermite_poly(n, x):
    """
    Function that returns the (unnormalized) n-th Hermite polynomial 
    evaluated at x.
    
    Parameters
    ----------
    n : integer
        Degree of Hermite polynomial.
    
    x : ndarray
        Array of values.
    
    Returns
    -------
    
    ndarray 
        The n-th Hermite polynomial evaluated at x.
    
    """
    
    return Hermite.basis(n)(x) 


# Polinomio de Hermite de orden n (normalizado)
def normalized_hermite_poly(n, x):
    """
    Function that returns the (normalized) n-th Hermite polynomial 
    evaluated at x.
    
    Parameters
    ----------
    n : integer
        Degree of Hermite polynomial.
    
    x : ndarray
        Array of values.
    
    Returns
    -------
    
    ndarray 
        The n-th Hermite polynomial evaluated at x.
    
    """
    return hermite_poly(n, x) / sqrt(sqrt(pi) * 2 ** n * factorial(n))

# Función de Hermite de orden n (normalizada)
def hermite_function(n, x):
    """
    Function that returns the (normalized) n-th Hermite function 
    evaluated at x.
    
    Parameters
    ----------
    n : integer
        Degree of Hermite function.
    
    x : ndarray
        Array of values.
    
    Returns
    -------
    
    ndarray 
        The n-th (normalized) Hermite function evaluated at x.
    
    """
    return normalized_hermite_poly(n, x) * exp(- x ** 2 / 2)
