"""
Cubic Fractional Nonlinear Schrödinger Equation (fNLSE3) - Station. state evol.

Calculate the evolution of a stationary state with different split-step
symplectic and affine integrators. Uses Fourier pseudospectral representation.
Store various metrics (Inf error, L_2 norm error, Hamiltonian, computing time)
"""

import numpy as np
import time
from scipy.optimize import newton_krylov
from scipy.fft import fft, ifft, fftfreq

from state import State
from scheme import Scheme
from models import NLS
from solver import Solver

scheme_list = ['strang',
               'ruth',
               'neri',
               'affineS2',
               'affineS4',
               'affineS6',
               'affineS8']

N = [2**i for i in range(15,16)]

dt_list = [5.0e-1, 2.5e-1, 1.0e-1,
           5.0e-2, 2.5e-2, 1.0e-2,
           5.0e-3, 2.5e-3, 1.0e-3]


s = 0.7

omega = -1
gamma = -1
P = 200 * np.pi


def NLS_ground_state(x):
    return 1.0 / np.cosh(x)


# Time stamp for file identification
time_stamp = time.localtime()
time_stamp_str = time.strftime("%Y-%m-%d-%H:%M:%S", time_stamp)

# Format string for screen output
screen_template = "{0:4d} {1:e} {2:8s} {3:2.3f} {4:e} {5:e} {6:e} {7:e}"
# Format string for file output
file_template = "{0:d},{1:e},{2:s},{3:e},{4:e},{5:e},{6:e},{7:e}\n"

T = 10.

with open("output/fNLSE3_fourier_stationary_" + str(s) + "-" +
          time_stamp_str + '.csv', 'w') as fout:

    fout.write("N,delta_t,method,T,err_inf,err_2,err_H,comp_time\n")

    for dim in N:
        period = 200 * np.pi

        # Nonlinear functional to obtain the ground state numerically
        def F(Q):
            Q_hat = (1 / dim) * fft(Q)
            k = 2 * np.pi * fftfreq(dim, period / dim)
            Lap_s_Q_hat = 0.5 * np.abs(k)**(2*s) * Q_hat
            Lap_s_Q = dim * ifft(Lap_s_Q_hat)
            F_Q = Lap_s_Q - omega * Q + gamma * np.abs(Q)**2 * Q
            return F_Q

        # Initial guess for Newton-Krylov solver (NLSE3 ground state)
        u0 = State(rep='fourier', dim=dim,
                   u=NLS_ground_state, param=period)

        x = u0.nodes
        u0values = u0.values
        # Solve nonlinear system
        Qf = newton_krylov(F, u0values, f_rtol=1e-11)
        # Reset initial state for evolution (now it's a ground state)
        u0.set_values(Qf)

        for sch in scheme_list:
            for dt in dt_list:
                # Initialize the CGLE model
                model = NLS(a=0.5, b=-1.0, frac_exp=s)
                hamiltonian = model.get_hamiltonian()

                H0 = hamiltonian(u0)
                L2_norm = u0.norm()

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

                # Evolution of the NLSE3 soliton
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
                ufval_t = u0.values * np.exp(-1.0j * omega * t)
                # Numerical final state
                ufval = trajectory[-1][1].values

                err_H = abs(hamiltonian(u) - H0) / abs(H0)
                err_L2_norm = np.abs(L2_norm - u.norm()) / L2_norm

                err_inf = np.amax(np.abs(ufval_t - ufval))
                print(screen_template.format(dim, dt, sch, T, err_inf,
                                             err_L2_norm, err_H, comp_time))

                fout.write(file_template.format(dim, dt, sch, T, err_inf,
                                                err_L2_norm, err_H, comp_time))
