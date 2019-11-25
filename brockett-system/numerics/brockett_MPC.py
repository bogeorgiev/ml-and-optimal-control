import numpy as np
from scipy.optimize import minimize


N=100
h=1/N
t = np.arange(0,1+h/2,h/2)

x = np.zeros_like(t)
y = np.zeros_like(t)
z = np.zeros_like(t)

x[0] = y[0] = z[0] = 1
u_v = np.zeros(4*N + 2)


def num_trajectory(control, p0, h):
    phase = np.zeros((3,3))
    x = phase[:,0]
    y = phase[:,1]
    z = phase[:,2]
    x[0] = p0[0]
    y[0] = p0[1]
    z[0] = p0[2]
    u = control[0::2]
    v = control[1::2]

    x[1] = x[0] + 0.5*h*u[1]
    x[2] = x[0] + h*u[1]
    y[1] = y[0] + 0.5*h*v[1]
    y[2] = y[0] + h*v[1]
    z[1]  = z[0] + 0.5*h*(x[0]*v[1] - y[0]*u[1])
    z[2]  = z[0] + h*(x[1]*v[1] - y[1]*u[1])

    return phase[-1,:]


def trajectory(control, p0, h):
    phase = np.zeros((len(control)//2, 3))
    x = phase[:,0]
    y = phase[:,1]
    z = phase[:,2]
    x[0] = p0[0]
    y[0] = p0[1]
    z[0] = p0[2]
    u = control[::2]
    v = control[1::2]
    for i in range(0, len(x)-1, 2):
        x[i+1] = x[i] + 0.5*h*u[i+1]
        x[i+2] = x[i] + h*u[i+1]
        y[i+1] = y[i] + 0.5*h*v[i+1]
        y[i+2] = y[i] + h*v[i+1]
        z[i+1]  = z[i] + 0.5*h*(x[i]*v[i+1] - y[i]*u[i+1])
        z[i+2]  = z[i] + h*(x[i+1]*v[i+1] - y[i+1]*u[i+1])
    return phase


def brockett_integrator(control, p0, h):
    
    phase = trajectory(control, p0, h)
    x = phase[:,0]
    y = phase[:,1]
    z = phase[:,2]

    return 0.5*h*np.sum(x[1::2]*x[1::2] + y[1::2]*y[1::2] + z[1::2]*z[1::2])


def opt_procedure(control, p0):
    
    return minimize(brockett_integrator, u_v, args=p0, method="L-BFGS-B", bounds=None)


print(trajectory(np.ones(4), np.ones(3), .01))