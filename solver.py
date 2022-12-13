"""
Created on Wed Oct 19 17:49:01 2022.

@author: lisandro
"""


class Solver():
    """Class for solving 1D evolution equations.

    The solver uses a split-step scheme on a problem discretized by means of a
    pseudospectral (or other discrete) representation in space.
    """

    def __init__(self, model, scheme):

        self.model = model
        self.scheme = scheme
        self.state = None
        self.active = False

    def start(self, init_state, init_time, end_time):
        self.state = init_state
        self.sim_time = init_time
        self.end_time = end_time

        self._rep = init_state.rep
        self._dim = init_state.dim
        self._param = init_state.param

        self.P_A, self.P_B = self.model.get_propagators(self._rep,
                                                        self._dim,
                                                        self._param)

        self.scheme.set_propagators(self.P_A, self.P_B)
        self.active = True

    def step(self, dt):
        if self.state is not None:
            if self.active:
                if self.sim_time >= self.end_time:
                    self.active = False
                    return self.state
                elif self.sim_time + dt >= self.end_time:
                    dt = self.end_time - self.sim_time
                    self.active = False
                self.state = self.scheme.step(self.state, dt)
                self.sim_time += dt
                return self.state
        else:
            raise ValueError("The state hasn't been specified")

    def solve(init_state, time_points, model, scheme):
        pass
