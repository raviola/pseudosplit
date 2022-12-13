"""
Complex Ginzburg-Landau soliton evolution.

Calculate the evolution of a soliton with different split-step
symplectic and affine integrators. Uses Fourier pseudospectral representation.
Store various metrics (Inf error, L_2 norm error, computing time)
"""

import numpy as np
import time

from state import State
from scheme import Scheme
from models import CGLE_opt
from solver import Solver

scheme_list = ['strang',
               'affineS2',
               'affineS4',
               'affineS6',
               'affineS8']

N = [2**i for i in range(9, 15)]

dt_list = [5.0e-1, 2.5e-1, 1.0e-1,
           5.0e-2, 2.5e-2, 1.0e-2,
           5.0e-3, 2.5e-3, 1.0e-3]

# Set the parameters of the CGLE3 stable soliton solution (Akhmediev 1996)
beta = 0.25
lamb = (1 + 4 * beta**2)**0.5
epsilon = beta * (3 * lamb - 1) / (4 + 18 * beta**2)
delta = 0.0
d = (lamb - 1) / (2 * beta)
F = (d * lamb / (2 * epsilon))**0.5
G = 1.0
omega = -d * lamb**2 / (2 * beta) * G**2


# Define the soliton function (Akhmediev 1996)
def soliton(x):
    """
    Soliton solution for the CGLE.

    Parameters
    ----------
    x : array-like
        Space coordinates.

    Returns
    -------
    array-like
        Soliton values at the given space coordinates.

    """
    return G*F/np.cosh(G*x)*np.exp(1.0j*d*np.log(G*F/np.cosh(G*x)))


# Time stamp for file identification
time_stamp = time.localtime()
time_stamp_str = time.strftime("%Y-%m-%d-%H:%M:%S", time_stamp)

# Format string for screen output
screen_template = "{0:4d} {1:e} {2:8s} {3:2.3f} {4:e} {5:e} {6:e}"
# Format string for file output
file_template = "{0:d},{1:e},{2:s},{3:e},{4:e},{5:e},{6:e}\n"

T = 10.

with open("output/CGLE3_fourier_soliton-" +
          time_stamp_str + '.csv', 'w') as fout:
    fout.write("N,delta_t,method,T,err_inf,err_2,comp_time\n")
    for dim in N:
        for sch in scheme_list:
            for dt in dt_list:
                # Initialize the CGLE model
                model = CGLE_opt(beta=beta, epsilon=epsilon, delta=delta)
                # Get the pseudospectral representation of stable CGLE3 soliton
                period = 200*np.pi
                u0 = State(rep='fourier', dim=dim, u=soliton, param=period)
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

                u0val = u0.values
                # Theoretical final state
                ufval_t = u0val*np.exp(-1.0j*(omega)*t)
                # Numerical final state
                ufval = trajectory[-1][1].values

                L2_norm = u0.norm()
                err_L2_norm = np.abs(L2_norm - u.norm()) / L2_norm

                err_inf = np.amax(np.abs(ufval_t-ufval))

                print(screen_template.format(dim, dt, sch, T, err_inf,
                                             err_L2_norm, comp_time))

                fout.write(file_template.format(dim, dt, sch, T, err_inf,
                                                err_L2_norm, comp_time))
