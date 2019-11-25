import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from new_lbfgsb import solve as nmpc_solver

class NMPC:

    def __init__(self, calc_trajectory, objective_f, x0, u0, nmpc_it=100 , N=50, m=5, lower_b=None, upper_b=None, solver_it=15):
        self.calc_trajectory = calc_trajectory
        self.obj_f = objective_f
        self.x0 = x0
        self.u0 = u0
        self.m = m
        self.N = N
        self.h = 1/self.N
        self.solver_it = solver_it
        self.nmpc_it = nmpc_it
        if not lower_b:
            self.lower_b = - np.ones_like(u0)*np.finfo(float).max
        if not upper_b:
            self.upper_b = np.ones_like(u0)*np.finfo(float).max
        self.controls = []
        self.plotting = True


    def solve(self):

        x_k = self.x0
        u_k = self.u0

        for k in range(self.nmpc_it):
            res = nmpc_solver(partial(self.obj_f, x_k, self.N, self.h), u_k, l=self.lower_b, u=self.upper_b, m=self.m, max_iters=self.solver_it)
            u_k = res
            self.controls.append(res[0])
            x_k = self.calc_trajectory(u_k, self.N, self.h)[0]
            if self.plotting:
                plt.scatter(k, u_k)
                plt.pause(0.05)

        plt.show()


def calc_trajectory(control, x0, N, h):

    x = np.zeros(2*N + 1)
    y = np.zeros(2*N + 1)
    z = np.zeros(2*N + 1)

    x[0] = x0[0]
    y[0] = x0[1]
    z[0] = x0[2]

    u = control[:2*N+1]
    v = control[2*N+1:]
    for i in range(0, 2*N, 2):
        x[i+1] = x[i] + 0.5*h*u[i+1]
        x[i+2] = x[i] + h*u[i+1]
        y[i+1] = y[i] + 0.5*h*v[i+1]
        y[i+2] = y[i] + h*v[i+1]
        z[i+1]  = z[i] + 0.5*h*(x[i]*v[i+1] - y[i]*u[i+1])
        z[i+2]  = z[i] + h*(x[i+1]*v[i+1] - y[i+1]*u[i+1])

    return x,y,z


def brockett_integrator(control, x0, N, h):
    
    x,y,z = calc_trajectory(control, N, h, x0)

    return 0.5*h*np.sum(x[1::2]*x[1::2] + y[1::2]*y[1::2] + z[1::2]*z[1::2])


N = 50
x0 = np.zeros(2*N+1)
u0 = np.zeros(4*N+2)

nmpc = NMPC(calc_trajectory, brockett_integrator, x0, u0)
nmpc.solve()