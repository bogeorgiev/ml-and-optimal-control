
import numpy as np


class MPC:

    def __init__(self, x0, bounds, h, num_trajectory, procedure, max_iter=100):
        self.x0 = x0
        self.bounds = bounds
        self.h = h
        self.num_trajectory = num_trajectory
        self.procedure = procedure
        self.max_iter = max_iter

    def solve(self):
        trajectory = []
        controls = []
        p0 = self.x0
        for it in range(self.max_iter):
            res = self.procedure(p0, self.bounds)
            controls.append(res.x[0])
            trajectory.append(self.num_trajectory(res.x[0], p0, self.h))
            p0 = trajectory[-1][1]
        return controls, trajectory
