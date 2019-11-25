import sys
sys.path.append("..")

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from nn_models.dataloader import BrockettTrajectoryDataset
from utilities.visualize import plot_trajectory_3d, trajectory, numpy_trajectory

if __name__ == "__main__":
    data_size = 2 
    traj_size = 70
    sample = 0
    ds = BrockettTrajectoryDataset(size=data_size)
    control_norm = ds.control_norm
    loader = DataLoader(ds, batch_size=data_size, shuffle=True) 
    it = iter(loader)
    init_points, controls, traj = next(it)
    print(init_points.shape, controls.shape, traj.shape)
    controls = controls * control_norm
    target = controls[:, :, :traj_size].to('cuda')
    #target = target.view(data_size, traj_size, 2)
   

    for sample in range(data_size):
        predicted_traj = numpy_trajectory(controls[sample, 0, :],
                                    controls[sample, 1, :], init_points[sample]) 
        plot_trajectory_3d(pred_traj=predicted_traj, 
                            target_traj=traj[sample, :, :].cpu().numpy(),
                            save_path="./img/numpy_trajectory_np.png",
                            show=True)

        predicted_traj = trajectory(target, init_points, device='cuda',
                                    num_points=traj_size)

        plot_trajectory_3d(pred_traj=predicted_traj[sample, :, :].detach().cpu().numpy(),
                            target_traj=traj[sample, :, :].cpu().numpy(),
                            save_path="./img/torch_trajectory.png",
                            show=True)


    N = 70

    initial_points_path = "../data/init_points.csv"
    controls_path = "../data/controls.csv"
    trajectories_path = "../data/trajectories.csv"

    initial_points = np.genfromtxt(initial_points_path, delimiter=',')
    controls = np.genfromtxt(controls_path, delimiter=',')
    trajectories = np.genfromtxt(trajectories_path, delimiter=',')

    controls = controls.reshape((-1, 2, N))
    trajectories = trajectories.reshape((-1, 2*N+1 ,3))

    for i in range(5):
        traj = numpy_trajectory(controls[i, 0, :], controls[i, 1, :], initial_points[i])
        plot_trajectory_3d(pred_traj=traj,
                        target_traj=trajectories[i, :, :], 
                        save_path="./img/Combined_{}.png".format(i))
        plot_trajectory_3d(pred_traj=traj,
                        save_path="./img/Predicted_{}.png".format(i))
