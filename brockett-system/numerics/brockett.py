import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.optimize import minimize

np.random.seed(101)

N = 70
h = 1/N
t = np.arange(0,1+h/2,h/2)

batch_size = 200
num_samples = 10_000

init_points = 1 * np.random.rand(num_samples, 3)
np.savetxt("./data/init_points.csv", init_points, delimiter=",")

control = np.zeros(4 * N + 2)
# bounds = [(-2, 2)]*(4 * N+2)
method = "L-BFGS-B" #"CG" "Newton-CG"


def trajectory(control, init_point):
    u = control[:2*N+1]
    v = control[2*N+1:]

    x[0] = init_point[0]
    y[0] = init_point[1]
    z[0] = init_point[2]

    for i in range(0, 2*N, 2):
        x[i+1] = x[i] + 0.5*h*u[i+1]
        x[i+2] = x[i] + h*u[i+1]
        y[i+1] = y[i] + 0.5*h*v[i+1]
        y[i+2] = y[i] + h*v[i+1]
        z[i+1]  = z[i] + 0.5*h*(x[i]*v[i+1] - y[i]*u[i+1])
        z[i+2]  = z[i] + h*(x[i+1]*v[i+1] - y[i+1]*u[i+1])
    return x,y,z


def brockett_obj_func(control, init_point):
    
    x,y,z = trajectory(control, init_point)

    return 0.5*h*np.sum(x[1::2]*x[1::2] + y[1::2]*y[1::2] + z[1::2]*z[1::2])

obj_func = brockett_obj_func
traj = trajectory

result_controls = np.array([])
result_traj = None

for num, init_point in tqdm(enumerate(init_points)):

    x = np.zeros_like(t)
    y = np.zeros_like(t)
    z = np.zeros_like(t)
    current_res = minimize(obj_func, control, args=(init_point), method=method).x
    result_controls = np.append(result_controls, current_res[:2*N+1][1::2])
    result_controls = np.append(result_controls, current_res[2*N+1:][1::2])
    opt_x, opt_y, opt_z = traj(current_res, init_point)
    # opt_x, opt_y, opt_z = opt_x.reshape(-1,1), opt_y.reshape(-1,1), opt_z.reshape(-1,1)
    current_traj = np.stack([opt_x, opt_y, opt_z], axis=1)
    if num == 0:
        result_traj = current_traj
    else:
        result_traj = np.concatenate([result_traj, current_traj], axis=0)
        tqdm.write("{}, {}".format(result_traj.shape, result_controls.shape))

    if (num + 1) % batch_size == 0:
        np.savetxt("./data/controls.csv", result_controls, delimiter=",")
        np.savetxt("./data/trajectories.csv", result_traj, delimiter=",")
    
    # print(current_res[:2*N+1][1::2])
    # print(current_res[2*N+1:][1::2])

#result = minimize(obj_fun1, control, method=method, bounds=bounds).x
#result_inter = minimize(objective_func_inter, control, method="L-BFGS-B", ).x

#print(result)

# fig1, ax = plt.subplots(num_samples, 2)
# for sample in range(num_samples):
#     ax[sample][0].plot(result[sample][:2*N+1][1::2])
#     ax[sample][1].plot(result[sample][2*N+1:][1::2])


# fig2, ax = plt.subplots(num_samples, 3)
# for sample in range(num_samples):
#     opt_x, opt_y, opt_z = traj(result[sample], init_points[sample])
#     ax[sample][0].plot(x)
#     ax[sample][1].plot(y)
#     ax[sample][2].plot(z)    

 

# from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure(figsize=plt.figaspect(0.5))
# for sample in range(num_samples):
#     ax = fig.add_subplot(num_samples, 1, sample + 1, projection='3d')
#     opt_x, opt_y, opt_z = traj(result[sample], init_points[sample])
#     print(opt_x[-1], opt_y[-1], opt_z[-1])
#     ax.plot(opt_x, opt_y, opt_z, '-b')


# plt.show()     

#ax[1][1].plot(result_inter)
#ax[1][1].plot(gt_u(t))
#ax[1][0].plot(trajectory(gt_u(t)))
#ax[1][0].plot(trajectory(result_inter))

# x[0] = y[0] = z[0] = 1

# u_v = np.zeros(4*N + 2)


# procedure = "L-BFGS-B"
# bound = [(-100,100)]*(4*N+2)
# res = minimize(brockett_integrator, u_v, method=procedure, bounds=bound)
# opt_x, opt_y, opt_z = trajectory(res.x)

# # lower = -np.ones(4*N + 2)*100
# # upper = np.ones(4*N + 2)*100
# # res = solve(brockett_integrator, u_v, l=lower, u=upper, m=5, max_iters=35)
# # opt_x, opt_y, opt_z = trajectory(res)


# from mpl_toolkits.mplot3d import Axes3D

# fig, ax = plt.subplots(2, 1)
# ax[0].plot(res.x[1:2*N + 1:2])
# ax[0].grid(True)
# ax[1].plot(res.x[2*N + 2::2])
# ax[1].grid(True)

# fig1, ax = plt.subplots(3,1)
# ax[0].plot(opt_x)
# ax[0].grid(True)
# ax[1].plot(opt_y)
# ax[1].grid(True)
# ax[2].plot(opt_z)
# ax[2].grid(True)

# fig2 = plt.figure()
# ax = plt.axes(projection='3d')

# ax.plot(opt_x, opt_y, opt_z, '-b')

# plt.show()
