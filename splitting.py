"""
Splitting module.

The splitting module provides functions for solving evolution equations using split-step schemes.
"""
import numpy as np

def evolve(u0,             # initial datum
           dt,             # time step
           T,              # total evolution time
           P_A,            # propagator for operator A
           P_B,            # propagator for operator B
           param=None,     # additional parameters for propagators
           callback=None): # callback function to be called at the end of every evolution step
    
    r""" 
    Calculate the solution of an evolution equation using a split-step scheme.
    
    Given the evolution equation
    
    .. math:: \partial_t u = A(u) + B(u)
    
    this function calculates its solution given the initial datum u0=u(t=0) and the propagators for A and B.
    
    Parameters
    ----------
    
    u0 : array_like
        Initial datum (state) using the representation expected by `P_A` and `P_B`.
      
    dt : float
        Time step for the integration.
      
    T : float
        Total time for the evolution.
        
    P_A : function of 2 variables
        Propagator function which solves (exactly or not) the evolution problem for operator A only.
        
        `P_A` (u, dt) takes a state u and a time step dt and returns the evolved state.
    
    P_B : function of 2 variables
        Propagator function which solves (exactly or not) the evolution problem for operator B only.
        
        Idem `P_A`.
    
    param : dict, optional (TODO, experimental)
        Additional parameters to be passed to each propagator.
    
    callback : function, optional (TODO, experimental)
        A function to be called at the end of each integration step.
        
    Returns
    -------
    
    ndarray
        A matrix with shape (STEPS, np.size(u0)), where STEPS is the number of integration steps and np.size(u0) is the dimension of the representation of the solution.
    """
    
    STEPS = int(T / dt)    # number of integration steps
    t = np.arange(0., T, dt)

    # define the matrix for storing the solution at each step and initialize the first entry with initial datum (u0)
    sol = np.zeros((STEPS, np.size(u0)), dtype=np.clongdouble)
    sol[0,:] = u0
    u = u0

    # split-step main loop (currently implementing the Strang 2nd order scheme, but this can be generalized - TODO)
    for step in range(1, STEPS):
        u1 = P_A(u, dt/2)
        u2 = P_B(u1, dt)
        u = P_A(u2, dt/2)
        sol[step, :] = u
    
    return t, sol    
