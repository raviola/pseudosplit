#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cubic Nonlinear Schrödinger Equation (NLSE3) -  Soliton evolution.

Calculate the evolution of a soliton with different split-step
symplectic and affine integrators. Uses Fourier pseudospectral representation.
Store various metrics (Inf error, L_2 norm error, computing time)
"""

import numpy as np
import time

from state import State
from scheme import Scheme
from models import NLS
from solver import Solver

scheme_list = [#'strang',
               #'affineS2',
               #'ruth',
               'neri',
               'affineS4',
               'affineS6',
               'affineS8']

N = [100*i for i in range(6, 10)]

dt_list = [5.0e-1, 2.5e-1, 1.0e-1,
           5.0e-2, 2.5e-2 , 1.0e-2]
           #5.0e-3, 2.5e-3, 1.0e-3]

# Set the parameters of the NLSE3 soliton solution
c = 0.0
eta = 1.0
omega = 0.5*(c**2 - eta**2)


# Define the soliton function
def soliton(x):
    """
    Soliton solution for the NLSE3.

    Parameters
    ----------
    x : array-like
        Space coordinates.

    Returns
    -------
    array-like
        Soliton values at the given space coordinates.

    """
    return eta / np.cosh(eta*(x)) * np.exp(1.0j*(c*x))


# Time stamp for file identification
time_stamp = time.localtime()
time_stamp_str = time.strftime("%Y-%m-%d-%H:%M:%S", time_stamp)

# Format string for screen output
screen_template = "{0:4d} {1:e} {2:8s} {3:2.3f} {4:e} {5:e} {6:e}"
# Format string for file output
file_template = "{0:d},{1:e},{2:s},{3:e},{4:e},{5:e},{6:e}\n"

T = 10.

with open("NLSE3_hermite_soliton-" + time_stamp_str + '.csv', 'w') as fout:
    fout.write("N,delta_t,method,T,err_inf,err_2,comp_time\n")
    for dim in N:
        for sch in scheme_list:
            for dt in dt_list:
                # Initialize the CGLE model
                model = NLS(a=0.5, b=-1.0)
                # Get the pseudospectral representation of stable CGLE3 soliton
                u0 = State(rep='hermite', dim=dim, u=soliton)
                # Set initial and final times for simulation
                t0 = 0.0
                tf = T
                # Select the split-step scheme
                scheme = Scheme(sch)
                # Create the solver
                solver = Solver(model, scheme)
                # A trajectory is a list of tuples (time, state)
                trajectory = [(t0, u0)]
                # Start the solver
                solver.start(u0, t0, tf)
                # Time at the beginning of calculations
                comp_time_init = time.time()

                # Evolution of a CGLE3 soliton
                while solver.active:
                    u = solver.step(dt)
                    t = solver.sim_time
                    trajectory.append((t, u))

                # Time at the end of calculations
                comp_time_final = time.time()
                # Time elapsed during calculations
                comp_time = comp_time_final - comp_time_init

                x = u0.nodes
                # Theoretical final state
                ufval_t = eta / np.cosh(eta*(x - c*t)) * np.exp(1.0j*(c*x - omega*t))
                # Numerical final state
                ufval = trajectory[-1][1].values

                L2_norm = u0.norm()
                err_L2_norm = np.abs(L2_norm - u.norm()) / L2_norm

                err_inf = np.amax(np.abs(ufval_t-ufval))

                print(screen_template.format(dim, dt, sch, T, err_inf,
                                             err_L2_norm, comp_time))

                fout.write(file_template.format(dim, dt, sch, T, err_inf,
                                                err_L2_norm, comp_time))
