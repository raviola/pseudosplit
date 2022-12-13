#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 12:25:24 2022.

@author: lisandro
"""

from state import State
from affine import (affineS2_step,
                    affineS4_step,
                    affineS6_step,
                    affineS8_step)
from symplectic import (strang_step,
                        ruth_step,
                        neri_step)


class Scheme():
    """Time-splitting integration schemes."""

    def __init__(self, name, P_A=None, P_B=None):
        """
        Scheme object for time-splitting methods.

        Parameters
        ----------
        name : TYPE, optional
            DESCRIPTION. The default is None.
        P_A : TYPE, optional
            DESCRIPTION. The default is None.
        P_B : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        self.name = name
        if name == "affineS2":
            self.step_function = affineS2_step
        elif name == "affineS4":
            self.step_function = affineS4_step
        elif name == "affineS6":
            self.step_function = affineS6_step
        elif name == "affineS8":
            self.step_function = affineS8_step
        elif name == "strang":
            self.step_function = strang_step
        elif name == "ruth":
            self.step_function = ruth_step
        elif name == "neri":
            self.step_function = neri_step
        if P_A is not None:
            self.P_A = P_A
        if P_B is not None:
            self.P_B = P_B

    def set_propagators(self, P_A, P_B):
        """
        Set propagators for the splitting scheme.

        Parameters
        ----------
        P_A : TYPE
            DESCRIPTION.
        P_B : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.P_A = P_A
        self.P_B = P_B

    def step(self, u0, dt):
        """
        Advance one integration step of time `dt` from initial state `u0`.

        Parameters
        ----------
        u0 : State
            Initial state.
        dt : float
            Step size.

        Returns
        -------
        u : State
            State after evolution
        """
        u_values = self.step_function(u0.values,
                                      self.P_A,
                                      self.P_B,
                                      dt)
        u = State(rep=u0.rep, dim=u0.dim, param=u0.param)
        u.set_values(u_values)
        return u
