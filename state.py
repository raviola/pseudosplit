"""
Created on Sat Oct  1 21:16:15 2022.

@author: lisandro
"""

import numpy as np
from numpy import sqrt
from scipy.fft import fft, ifft, fftfreq, fftshift
from scipy.integrate import simps
from hermite import hermite_nodes, hermite_matrix


class HermiteState():
    def __init__(self, dim, u=None, param=None, spectral=False):

        self.rep = 'hermite'
        self.dim = dim

        if param is not None:
            self.param = self.scaling = param
        else:
            self.param = self.scaling = 1.0

        if spectral:
            raise NotImplementedError("Spectral definition not implemented.")

        self.nodes, self.weights = hermite_nodes(self.dim, weights=True)
        if u is not None:
            if callable(u):
                self.values = u(self.nodes).astype('complex')
            else:
                raise ValueError("u must be a NumPy ufunc.")
        else:  # Deferred definition of values
            self.values = np.zeros(self.dim, dtype='complex')

        self.coeffs = None

    def set_values(self, data, spectral=False):
        if spectral:
            self.coeffs = data
            self.values = self.IDHT_matrix @ self.coeffs
        else:
            self.values = data

    def norm(self, n=2):
        """
        Return the state n-norm.

        Parameters
        ----------
        n : int, optional
            Order of norm. The default is 2.

        Returns
        -------
        norm : float
            n- norm of the state.

        """
        # Get norm using Gauss-Hermite quadrature formula
        norm = (self.weights @ np.abs(self.values)**n)**(1 / n)
        return norm

    def get_coeffs(self):
        """
        Return the spectral coefficients for the Hermite representation.

        Returns
        -------
        coeffs : tuple
            The first element of the tuple is an array of indices, and the
            second is the corresponding array of spectral coefficients.

        """
        if self.coeffs is None:
            # Calculate the matrix of scaled Hermite functions at nodes
            self.hermite_matrix = hermite_matrix(self.dim,
                                                 self.scaling * self.nodes)
            self.hermite_matrix *= sqrt(self.scaling)

            # Calculate the analysis matrix - Discrete Hermite Transform (DHT)
            self.DHT_matrix = self.hermite_matrix * self.weights

            # Calculate the synthesis matrix (inverse DHT)
            # self.IDHT_matrix = self.hermite_matrix.T

            # Calculate the Hermite coefficients

            self.coeffs = self.DHT_matrix @ self.values

        return (np.arange(self.dim, dtype='int'), self.coeffs)

    def dot(self, other):
        """
        Inner product.

        Parameters
        ----------
        other : State
            The state over which to calculate the (discrete) inner product.

        Returns
        -------
        dot_product : complex
            The complex (discrete) inner product

        """
        if (self.rep == other.rep
                and self.dim == other.dim
                and self.param == other.param):

            dot_product = self.weights @ (self.values * np.conj(other.values))
            return dot_product
        else:
            raise ValueError("Representations don't match"
                             "(type, dim, scaling)")


class FourierState():
    def __init__(self, dim, u=None, param=None, spectral=False):

        self.rep = 'fourier'
        self.dim = dim

        if param is not None:
            self.param = self.period = param
        else:
            self.param = self.period = 2 * np.pi

        if spectral:
            raise NotImplementedError("Spectral definition not implemented.")

        self.nodes = np.arange(-self.period / 2,
                               self.period / 2,
                               self.period / self.dim)
        if u is not None:
            if callable(u):
                self.values = u(self.nodes).astype('complex')
            else:
                raise ValueError("u must be a NumPy ufunc.")
        else:
            self.values = np.zeros(self.dim, dtype='complex')

        # # Calculate the Fourier coefficients using FFT
        # if u is not None:
        #     self.coeffs = 1 / self.dim * fft(self.values)
        # else:
        #     self.coeffs = np.zeros(self.dim, dtype='complex')

        self.coeffs = None

    def set_values(self, data, spectral=False):
        if spectral:
            self.coeffs = data
            self.values = self.dim * ifft(self.coeffs)
        else:
            self.values = data
            # self.coeffs = 1 / self.dim * fft(self.values)

    def norm(self, n=2):
        """
        Return norm of state.

        Parameters
        ----------
        n : integer, optional
            Order of norm. The default is 2.

        Returns
        -------
        norm : float
            The n- norm.

        """
        # Get the norm of state using the Simpson quadrature formula
        norm = (simps(np.abs(self.values)**n, self.nodes))**(1 / n)
        return norm

    def get_coeffs(self):
        """
        Return the spectral coefficients for the Fourier representation.

        Returns
        -------
        coeffs : tuple
            The first element of the tuple is an array of wacvenumbers, and the
            second is the corresponding complex array of spectral coefficients.
        """
        if self.coeffs is None:
            self.coeffs = 1 / self.dim * fft(self.values)

        return (2*np.pi*fftshift(fftfreq(self.dim, self.period/self.dim)),
                fftshift(self.coeffs))

    def dot(self, other):
        """
        Inner product.

        Parameters
        ----------
        other : State
            The state over which to calculate the (discrete) inner product.

        Returns
        -------
        dot_product : complex
            The complex (discrete) inner product

        """
        if (self.rep == other.rep
                and self.dim == other.dim
                and self.param == other.param):
            dot_product = self.period / self.dim * (self.values
                                                    @ np.conj(other.values))
            return dot_product
        else:
            raise ValueError("Representations don't match"
                             "(type, dim, period)")


def State(rep, dim, u=None, param=None, spectral=False):
    """
    Return an instance of a State class based on the given representation.

    Parameters
    ----------
    rep : str
        The name of the representation (basis). Currently it can be 'hermite'
        or 'fourier'.
    u : Numpy ufunc
        A function of one array argument that returns the value of the state
        over an array of grid points.
    dim : integer
        Dimension of the representation (number of basis modes).
    param : float, optional
        Additional parameter that defines the scaling.
    spectral : boolean, optional
        If False (default) the function `u` is defined over the 'physical'
        grid. If True, `u` is defined over the `spectral` grid.

    Raises
    ------
    ValueError, NotImplementedError

    Returns
    -------
    state : an instace of HermiteState or FourierState
        An instance of the correct representation.

    """
    if rep == 'hermite':
        return HermiteState(dim, u, param, spectral)
    elif rep == 'fourier':
        return FourierState(dim, u, param, spectral)
    else:
        raise ValueError("Incorrect representation")
