"""
Splitting module.

The splitting module provides functions for solving evolution equations using
split-step schemes.
"""
import numpy as np

from affine import affineS2_step, affineS4_step, affineS6_step, affineS8_step
from symplectic import strang_step, ruth_step, neri_step

integration_step = {'lie-trotter': None,
                    'strang': strang_step,
                    'ruth': ruth_step,
                    'neri': neri_step,
                    'affineS2': affineS2_step,
                    'affineS4': affineS4_step,
                    'affineS6': affineS6_step,
                    'affineS8': affineS8_step}


def evolve(u0,             # initial datum
           dt,             # time step
           T,              # total evolution time
           P_A,            # propagator for operator A
           P_B,            # propagator for operator B
           param=None,     # additional parameters for propagators
           callback=None,  # callback function called at the end of every step
           method='strang'):
    r"""
    Calculate the solution of an evolution equation using a split-step scheme.

    Given the evolution equation

    .. math:: \partial_t u = A(u) + B(u)

    this function calculates its solution given the initial datum u0=u(t=0) and
    the propagators for A and B.

    Parameters
    ----------
    u0 : array_like
        Initial state using the representation expected by `P_A` and `P_B`.

    dt : float
        Time step for the integration.

    T : float
        Total time for the evolution.

    P_A : function of 2 variables
        Propagator function which solves (exactly or not) the evolution problem
        for operator A only.

        `P_A` (u, dt) takes a state u and a time step dt and returns the
        evolved state.

    P_B : function of 2 variables
        Propagator function which solves (exactly or not) the evolution problem
        for operator B only.

        Idem `P_A`.

    param : dict, optional (#TODO, experimental)
        Additional parameters to be passed to each propagator.

    callback : function, optional (#TODO, experimental)
        A function to be called at the end of each integration step.

    Returns
    -------
    ndarray
        A matrix with shape (STEPS, np.size(u0)), where STEPS is the number of
        integration steps and np.size(u0) is the dimension of the
        representation of the solution.
    """
    STEPS = int(T / dt)    # number of integration steps
    t = np.arange(0., T, dt)

    # define the matrix for storing the solution at each step and initialize
    # the first entry with initial datum (u0)
    sol = np.zeros((STEPS, np.size(u0)), dtype=np.complex128)
    local_error = np.zeros((STEPS,), dtype=np.float64)
    sol[0, :] = u0
    u = u0
    local_error[0] = 0.

    # split-step main loop
    for step in range(1, STEPS):

        u = integration_step[method](u, P_A, P_B, dt)

        sol[step, :] = u

    return t, sol
