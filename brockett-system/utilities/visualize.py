import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import io
import torch


N = 70
h = 1/N

def trajectory(controls, init_points, device, num_points=70):
    """
        controls: Batch x 2 x num_points
    """
    batch_size = controls.shape[0]
    u = controls[:, 0, :].clone()
    v = controls[:, 1, :].clone()
    x = torch.zeros(batch_size, 2*num_points + 1).to(device)
    y = torch.zeros(batch_size, 2*num_points + 1).to(device)
    z = torch.zeros(batch_size, 2*num_points + 1).to(device)

    x[:, 0] = init_points[:, 0].clone()
    y[:, 0] = init_points[:, 1].clone()
    z[:, 0] = init_points[:, 2].clone()

    for i in range(0, 2*num_points, 2):
        x[:, i+1] = x[:, i] + 0.5*h*u[:, i//2].clone()
        x[:, i+2] = x[:, i] + h*u[:, i//2].clone()
        y[:, i+1] = y[:, i] + 0.5*h*v[:, i//2].clone()
        y[:, i+2] = y[:, i] + h*v[:, i//2].clone()
        z[:, i+1]  = z[:, i] + 0.5*h*(x[:, i].clone()*v[:, i//2].clone() - y[:, i].clone()*u[:, i//2].clone())
        z[:, i+2]  = z[:, i] + h*(x[:, i+1].clone()*v[:, i//2].clone() - y[:, i+1].clone()*u[:, i//2].clone())

    return torch.cat([x[:, :-1][:, ::2].unsqueeze(2),
                    y[:, :-1][:, ::2].unsqueeze(2),
                    z[:, :-1][:, ::2].unsqueeze(2)], dim=2)


def numpy_trajectory(u, v, init_point):

    x = np.zeros(2*N + 1)
    y = np.zeros(2*N + 1)
    z = np.zeros(2*N + 1)

    x[0] = init_point[0]
    y[0] = init_point[1]
    z[0] = init_point[2]

    for i in range(0, 2*N, 2):
        x[i+1] = x[i] + 0.5*h*u[i//2]
        x[i+2] = x[i] + h*u[i//2]
        y[i+1] = y[i] + 0.5*h*v[i//2]
        y[i+2] = y[i] + h*v[i//2]
        z[i+1]  = z[i] + 0.5*h*(x[i]*v[i//2] - y[i]*u[i//2])
        z[i+2]  = z[i] + h*(x[i+1]*v[i//2] - y[i+1]*u[i//2])

    return np.stack([x[:-1][::2],y[:-1][::2],z[:-1][::2]], axis=1)



def plot_trajectory_3d(pred_traj, target_traj=None, save_path=None, show=False, buff=False):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = plt.axes(projection='3d')
    ax.plot(pred_traj[:,0], pred_traj[:,1], pred_traj[:,2], '-b')
    buf = None
    if target_traj is not None:
        ax.plot(target_traj[:,0], target_traj[:,1], target_traj[:,2], '-r')
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    if buff:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
    numpy_data = get_img_from_fig(fig)
    plt.close()
    return numpy_data 

def get_img_from_fig(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = torch.tensor(data)
    data = data.view(data.shape[0], data.shape[1], 3)
    return data 

def plot_controls(pred_control, target_control, save_path=None, show=False):
    fig, ax = plt.subplots(2, 1, figsize=plt.figaspect(0.5))
    ax[0].plot(pred_control[0], '-b')
    ax[0].plot(target_control[0], '-r')
    ax[1].plot(pred_control[1], '-b')
    ax[1].plot(target_control[1], '-r')
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()

def plot_single_controls(u_control, v_control, save_path=None, show=False):
    _, ax = plt.subplots(2, 1, figsize=plt.figaspect(0.5))
    ax[0].plot(u_control)
    ax[1].plot(v_control)
    if save_path:
        plt.savefig(save_path)
    if show: 
        plt.show() 




if __name__ == "__main__":
    import pdb

    # trajectories = np.genfromtxt("./data/trajectories.csv", delimiter=',')
    # trajectories = trajectories.reshape(1000, 141, 3)
    # plot_trajectory_3d(trajectories[1,:,0], trajectories[1,:,1], trajectories[1,:,2])

    controls = np.genfromtxt("../data/controls.csv", delimiter=',')
    controls = controls.reshape(1000, 2, 70)
    # plot_controls(controls[0,0,:], controls[0,1,:])
    # init_points = np.genfromtxt("../data/init_points.csv", delimiter=',')
    i = 1
    # traj1 = trajectory(controls[i,0,:], controls[i,1,:], init_points[i])
    # traj2 = trajectory(controls[i+1,0,:], controls[i+1,1,:], init_points[i+1])
    
    # buf  = plot_trajectory_3d(traj1, traj2, buff=True)

    # pdb.set_trace()
    plot_pred_n_target_controls(controls[i], controls[i+1], save_path="./img.png")
