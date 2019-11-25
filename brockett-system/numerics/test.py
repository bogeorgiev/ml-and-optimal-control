import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize


num_grid_points = 50 
h = 1 / num_grid_points

t = np.arange(0, 1 + h/2, h/2)

def gt_u(t):
    return (2*(np.exp(3*t) - np.exp(3)))/(np.exp(3*t/2)*(np.exp(3) + 2))

def gt_u1(t):
    t1 = t[t< .5]
    t2 = t[t>= .5]
    res = np.zeros_like(t)
    res[t< .5] = 1
    res[t>= .5] = (np.exp(1)**t2 - np.exp(1)**(2-t2))/(np.sqrt(np.exp(1))*(1 - np.exp(1)))
    return res

def trajectory(u, init_point=1):
    x = np.zeros_like(u)
    x[0] = init_point 
    for i in range(1, len(x), 2):
        x[i] = x[i-1] + 0.5 * h *(u[i] + 0.5 * x[i-1])
        x[i+1] = x[i-1] + h * (u[i] + 0.5 * x[i])
    return x

def trajectory1(u, init_point=(1 + 3 * np.exp(1)) / (2 * ( 1 - np.exp(1)))):
    x = np.zeros_like(u)
    x[0] = init_point
    for i in range(1, len(x), 2):
        x[i] = x[i-1] + 0.5*h*u[i]
        x[i+1] = x[i-1] + h*u[i]
    return x
    
def gt_f1(t):
    res = np.zeros_like(t)
    for i in range(len(t)):
        if t[i] < .5:
            res[i] = t[i] + (1 + 3*np.exp(1))/(2*(1-np.exp(1)))
        else:
            res[i] = (np.exp(1)**t[i] + np.exp(1)**(2-t[i]))/(np.sqrt(np.exp(1))*(1-np.exp(1)))
    return res


def objective_func(u, init_point):
    x = trajectory(u, init_point)
    result = 0.25 * h * (0.5 * np.sum(2 * x**2) + np.sum(u**2) )
    return result

def objective_func_inter(u, init_point):
    x = trajectory(u, init_point)
    result = 0.25 * h * np.sum(2 * x[1::2]**2 + u[1::2]**2) 
    return result


def obj_fun1(u, init_point):
    if init_point is not None:
        x = trajectory1(u, init_point=init_point)
    return 0.5 * h * np.sum(0.5 * x**2 + u**2)

num_samples = 10
init_points = 20 * np.random.rand(num_samples)
control = np.zeros_like(t)
bounds = None #[(-1, 1)] * (2 * num_grid_points + 1)
method = "L-BFGS-B" #"CG"
obj_func = objective_func_inter
traj = trajectory

result = []
for init_point in init_points:
    result += [minimize(obj_func, control, args=(init_point), method=method, bounds=bounds).x]

#result = minimize(obj_fun1, control, method=method, bounds=bounds).x
#result_inter = minimize(objective_func_inter, control, method="L-BFGS-B", ).x

#print(result)

fig, ax = plt.subplots(num_samples, 2)
for sample in range(num_samples):
    ax[sample][1].plot(result[sample][1::2])
    ax[sample][0].plot(traj(result[sample]))

#ax[1][1].plot(result_inter)
#ax[1][1].plot(gt_u(t))
#ax[1][0].plot(trajectory(gt_u(t)))
#ax[1][0].plot(trajectory(result_inter))

plt.show()
