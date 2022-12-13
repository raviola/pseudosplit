"""
Models for evolution equations.

@author: lisandro
"""
from numpy import pi, abs, exp, log, sum
from scipy.linalg import expm
from scipy.fft import fft, ifft, fftfreq
from pseudospectral import (hermite_to_collocation_matrix,
                            collocation_to_hermite_matrix,
                            hermite_operator_matrix,
                            hermite_evolution_matrix)


class NLS():
    """NLS model."""

    def __init__(self,
                 a=1.0,
                 b=1.0,
                 sigma=1.0,
                 frac_exp=1.0
                 ):

        self.a = a
        self.b = b
        self.sigma = sigma
        self.frac_exp = frac_exp
        self.P_A = None
        self.P_B = None
        self.rep = None
        self.dim = None
        self.param = None

        self.a_symbol = lambda k: self.a * abs(k)**(2*self.frac_exp)

    def get_hamiltonian(self):
        """
        Return the hamiltonian function of the fNLS model.

        Returns
        -------
        H : Numpy ufunc expecting a State as argument

        """
        def H(u):
            if u.rep != 'fourier':
                raise NotImplementedError("Only Fourier.")
            else:
                k = 2 * pi * fftfreq(u.dim, u.param / u.dim)
                multiplier = (-1.0j * abs(k))**self.frac_exp
                if u.coeffs is None:
                    u.coeffs = 1 / u.dim * fft(u.values)
                # TODO: Implement arbitrary power law
                K = u.period * sum(abs(multiplier * u.coeffs)**2)
                V = self.b * u.norm(4)**4
                return 0.5*(K + V)
        return H

    def get_propagators(self, rep, dim, param):
        """
        Return the partial propagators in the given representation.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if self.rep == rep and self.dim == dim and self.param == param:
            return self.P_A, self.P_B

        self.rep = rep
        self.dim = dim
        self.param = param

        # The Kerr propagator is identical in both representations
        # (nonlinear part)
        def P_A(u0, dt):
            u = u0 * exp(-1.0j * dt * self.b * abs(u0) ** (2 * self.sigma))
            P_A.calls += 1
            return u
        P_A.calls = 0
        self.P_A = P_A

        # Symbol for minus Laplacian (linear part)
        def a_symbol(k):
            return self.a * abs(k)**(2*self.frac_exp)

        # Calculate the linear propagator depending on representation
        if self.rep == 'hermite':
            self.h2c = hermite_to_collocation_matrix(self.dim)
            self.c2h = collocation_to_hermite_matrix(self.dim)

            # Generator of evolution
            C = -1.0j * hermite_operator_matrix(self.dim, a_symbol)

            # Define the linear propagator function

            def P_B(u0, dt):

                if (dt not in P_B.dt):
                    P_B.dt.append(dt)
                    # Exponential of C * dt solves u'(t) = C u(t)
                    # TODO: Determinar por qué usar la exponencial de C * dt
                    # (donde C se aproxima por cuadratura gaussiana) es mejor
                    # que usar el propagador completo obtenido por cuadratura
                    expCdt = expm(C * dt)
                    # expCdt = hermite_evolution_matrix(dim_base, B, dt)

                    P_B.propagator.append(self.h2c @ (expCdt @ self.c2h))

                i = P_B.dt.index(dt)
                u = P_B.propagator[i] @ u0
                P_B.calls += 1
                return u

            self.P_B = P_B
            P_B.dt = []
            P_B.propagator = []
            P_B.calls = 0

        elif self.rep == 'fourier':
            def P_B(u0, dt):

                k = 2 * pi * fftfreq(self.dim, self.param / self.dim)
                propagator = exp(-1.0j * a_symbol(k) * dt)
                u0_til = 1 / self.dim * fft(u0)
                u0_til_evol = propagator * u0_til
                u = self.dim * ifft(u0_til_evol)
                P_B.calls += 1

                return u

            P_B.dt = []
            P_B.propagator = []
            P_B.calls = 0
        else:
            raise ValueError("A representation must be given.")

        return P_A, P_B

class CGLE():
    def __init__(self, a=1.0, b=1.0, sigma=1.0, frac_exp=1.0):
        self.a = a
        self.b = b
        self.sigma = sigma
        self.frac_exp = frac_exp
        self.P_A = None
        self.P_B = None
        self.rep = None
        self.dim = None
        self.param = None

    def get_propagators(self, rep, dim, param):
        if self.rep == rep and self.dim == dim and self.param == param:
            return self.P_A, self.P_B

        self.rep = rep
        self.dim = dim
        self.param = param

        # The Kerr propagator is identical in both representations
        # (nonlinear part). The nonlinear partial propagator is exact.
        def P_A(u0, dt):
            u = u0 * exp(-1.0j / (-2.0 * self.sigma) *
                         (exp(-2.0 * self.sigma * dt) - 1) * (-1.0j + self.b) *
                         abs(u0) ** (2 * self.sigma))
            P_A.calls += 1
            return u
        P_A.calls = 0
        self.P_A = P_A

        # Symbol for dispersion and diffusion (linear part)
        def a_symbol(k):
            return (1.0j - self.a) * (-1) * abs(k)** (2*self.frac_exp) + 1.0j

        # Calculate the linear propagator depending on representation
        if self.rep == 'hermite':
            self.h2c = hermite_to_collocation_matrix(self.dim)
            self.c2h = collocation_to_hermite_matrix(self.dim)

            # Generator of evolution
            C = -1.0j * hermite_operator_matrix(self.dim, a_symbol)

            # Define the linear propagator function

            def P_B(u0, dt):

                if (dt not in P_B.dt):
                    P_B.dt.append(dt)
                    # Exponential of C * dt solves u'(t) = C u(t)
                    # TODO: Determinar por qué usar la exponencial de C * dt
                    # (donde C se aproxima por cuadratura gaussiana) es mejor
                    # que usar el propagador completo obtenido por cuadratura
                    expCdt = expm(C * dt)
                    # expCdt = hermite_evolution_matrix(dim_base, B, dt)

                    P_B.propagator.append(self.h2c @ (expCdt @ self.c2h))

                i = P_B.dt.index(dt)
                u = P_B.propagator[i] @ u0
                P_B.calls += 1
                return u

            self.P_B = P_B
            P_B.dt = []
            P_B.propagator = []
            P_B.calls = 0

        elif self.rep == 'fourier':
            def P_B(u0, dt):

                k = 2 * pi * fftfreq(self.dim, self.param / self.dim)
                propagator = exp(-1.0j * a_symbol(k) * dt)
                u0_til = 1 / self.dim * fft(u0)
                u0_til_evol = propagator * u0_til
                u = self.dim * ifft(u0_til_evol)
                P_B.calls += 1

                return u

            P_B.dt = []
            P_B.propagator = []
            P_B.calls = 0
        else:
            raise ValueError("A representation must be given.")

        return P_A, P_B


class CGLE_opt():
    def __init__(self, beta=1.0, epsilon=1.0, delta=1.0,
                 sigma=1.0, frac_exp=1.0):
        self.beta = beta
        self.epsilon = epsilon
        self.delta = delta
        self.sigma = sigma
        self.frac_exp = frac_exp
        self.P_A = None
        self.P_B = None
        self.rep = None
        self.dim = None
        self.param = None

    def get_propagators(self, rep, dim, param):
        if self.rep == rep and self.dim == dim and self.param == param:
            return self.P_A, self.P_B

        self.rep = rep
        self.dim = dim
        self.param = param

        # The Kerr propagator is identical in both representations
        # (nonlinear part). The nonlinear partial propagator is exact.
        def P_A(u0, dt):
            u = u0 * exp(-(1.0j + self.epsilon) / (2*self.epsilon) *
                         log(1 - 2*self.epsilon * abs(u0)**2 * dt))

            P_A.calls += 1
            return u
        P_A.calls = 0
        self.P_A = P_A

        # Symbol for dispersion and diffusion (linear part)
        def a_symbol(k):
            return ((1.0j * self.beta - 1/2) * (-1) * abs(k)**(2*self.frac_exp)+
                    1.0j * self.delta)

        # Calculate the linear propagator depending on representation
        if self.rep == 'hermite':
            self.h2c = hermite_to_collocation_matrix(self.dim)
            self.c2h = collocation_to_hermite_matrix(self.dim)

            # Generator of evolution
            C = -1.0j * hermite_operator_matrix(self.dim, a_symbol)

            # Define the linear propagator function

            def P_B(u0, dt):

                if (dt not in P_B.dt):
                    P_B.dt.append(dt)
                    # Exponential of C * dt solves u'(t) = C u(t)
                    # TODO: Determinar por qué usar la exponencial de C * dt
                    # (donde C se aproxima por cuadratura gaussiana) es mejor
                    # que usar el propagador completo obtenido por cuadratura
                    expCdt = expm(C * dt)
                    # expCdt = hermite_evolution_matrix(dim_base, B, dt)

                    P_B.propagator.append(self.h2c @ (expCdt @ self.c2h))

                i = P_B.dt.index(dt)
                u = P_B.propagator[i] @ u0
                P_B.calls += 1
                return u

            self.P_B = P_B
            P_B.dt = []
            P_B.propagator = []
            P_B.calls = 0

        elif self.rep == 'fourier':
            def P_B(u0, dt):

                k = 2 * pi * fftfreq(self.dim, self.param / self.dim)
                propagator = exp(-1.0j * a_symbol(k) * dt)
                u0_til = 1 / self.dim * fft(u0)
                u0_til_evol = propagator * u0_til
                u = self.dim * ifft(u0_til_evol)
                P_B.calls += 1

                return u

            P_B.dt = []
            P_B.propagator = []
            P_B.calls = 0
        else:
            raise ValueError("A representation must be given.")

        return P_A, P_B


def NLS_propagators(a_coeff=1.0,
                    b_coeff=1.0,
                    sigma=1.0,
                    rep='hermite',
                    dim=500,
                    period=None
                    ):
    """
    Return a tuple of propagators for the NLS model in given representation.

    Parameters
    ----------
    a_coeff : TYPE, optional
        DESCRIPTION. The default is 1.0.
    b_coeff : TYPE, optional
        DESCRIPTION. The default is 1.0.
    sigma : TYPE, optional
        DESCRIPTION. The default is 1.0.
    rep : TYPE, optional
        DESCRIPTION. The default is 'hermite'.
    dim : TYPE, optional
        DESCRIPTION. The default is 500.
    period : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    # The Kerr propagator is identical in both representations (nonlinear part)
    def P_A(u0, dt):
        u = u0 * exp(-1.0j * dt * b_coeff * abs(u0) ** (2 * sigma))
        P_A.calls += 1
        return u
    P_A.calls = 0

    # Symbol for minus Laplacian (linear part)
    def a(k):
        return a_coeff * abs(k)**2

    # Calculate the linear propagator depending on representation
    if rep == 'hermite':
        h2c = hermite_to_collocation_matrix(dim)
        c2h = collocation_to_hermite_matrix(dim)

        # Generator of evolution
        C = -1.0j * hermite_operator_matrix(dim, a)

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

    elif rep == 'fourier':
        def P_B(u0, dt):

            k = 2 * pi * fftfreq(dim, period/dim)
            propagator = exp(-1.0j * a(k) * dt)
            u0_til = 1 / dim * fft(u0)
            u0_til_evol = propagator * u0_til
            u = dim * ifft(u0_til_evol)
            P_B.calls += 1

            return u

        P_B.dt = []
        P_B.propagator = []
        P_B.calls = 0

    else:
        raise ValueError("Incorrect representation. "
                         "Use 'fourier' or 'hermite'")

    return P_A, P_B


def fNLS_propagators(a_coeff=1.0,
                     b_coeff=1.0,
                     sigma=1.0,
                     rep='hermite',
                     dim=500,
                     period=None,
                     s=1.0
                     ):
    """
    Return a tuple of propagators for the fNLS model in given representation.

    Parameters
    ----------
    a_coeff : TYPE, optional
        DESCRIPTION. The default is 1.0.
    b_coeff : TYPE, optional
        DESCRIPTION. The default is 1.0.
    sigma : TYPE, optional
        DESCRIPTION. The default is 1.0.
    rep : TYPE, optional
        DESCRIPTION. The default is 'hermite'.
    dim : TYPE, optional
        DESCRIPTION. The default is 500.
    period : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    # The Kerr propagator is identical in both representations (nonlinear part)
    def P_A(u0, dt):
        u = u0 * exp(-1.0j * dt * b_coeff * abs(u0) ** (2 * sigma))
        P_A.calls += 1
        return u
    P_A.calls = 0

    # Symbol for minus Laplacian (linear part)
    def a(k):
        return a_coeff * abs(k)**(2*s)

    # Calculate the linear propagator depending on representation
    if rep == 'hermite':
        h2c = hermite_to_collocation_matrix(dim)
        c2h = collocation_to_hermite_matrix(dim)

        # Generator of evolution
        C = -1.0j * hermite_operator_matrix(dim, a)

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

    elif rep == 'fourier':
        def P_B(u0, dt):

            k = 2 * pi * fftfreq(dim, period/dim)
            propagator = exp(-1.0j * a(k) * dt)
            u0_til = 1 / dim * fft(u0)
            u0_til_evol = propagator * u0_til
            u = dim * ifft(u0_til_evol)
            P_B.calls += 1

            return u

        P_B.dt = []
        P_B.propagator = []
        P_B.calls = 0

    else:
        raise ValueError("Incorrect representation. "
                         "Use 'fourier' or 'hermite'.")
    return P_A, P_B


def CGLE_propagators(a_coeff=1.0 - 1.0j,
                     b_coeff=1.0 - 1.0j,
                     sigma=1.0,
                     rep='hermite',
                     dim=500,
                     period=None,
                     s=1.0
                     ):
    """
    Return a tuple of propagators for the fNLS model in given representation.

    Parameters
    ----------
    a_coeff : TYPE, optional
        DESCRIPTION. The default is 1.0.
    b_coeff : TYPE, optional
        DESCRIPTION. The default is 1.0.
    sigma : TYPE, optional
        DESCRIPTION. The default is 1.0.
    rep : TYPE, optional
        DESCRIPTION. The default is 'hermite'.
    dim : TYPE, optional
        DESCRIPTION. The default is 500.
    period : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    # The Kerr propagator is identical in both representations (nonlinear part)
    def P_A(u0, dt):
        u = u0 * exp(-1.0j * dt * b_coeff * abs(u0) ** (2 * sigma))
        P_A.calls += 1
        return u
    P_A.calls = 0

    # Symbol for linear part of CGLE
    def a(k):
        return a_coeff * abs(k)**(2*s) + 1.0j

    # Calculate the linear propagator depending on representation
    if rep == 'hermite':
        h2c = hermite_to_collocation_matrix(dim)
        c2h = collocation_to_hermite_matrix(dim)

        # Generator of evolution
        C = -1.0j * hermite_operator_matrix(dim, a)

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

    elif rep == 'fourier':
        def P_B(u0, dt):

            k = 2 * pi * fftfreq(dim, period/dim)
            propagator = exp(-1.0j * a(k) * dt)
            u0_til = 1 / dim * fft(u0)
            u0_til_evol = propagator * u0_til
            u = dim * ifft(u0_til_evol)
            P_B.calls += 1

            return u

        P_B.dt = []
        P_B.propagator = []
        P_B.calls = 0

    else:
        raise ValueError("Incorrect representation. "
                         "Use 'fourier' or 'hermite'.")
    return P_A, P_B
