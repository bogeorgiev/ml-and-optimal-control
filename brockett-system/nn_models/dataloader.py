import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from mpl_toolkits.mplot3d import Axes3D
from random import randint

initial_points_path = "../data/init_points.csv"
controls_path = "../data/controls.csv"
trajectories_path = "../data/trajectories.csv"


class BrockettTrajectoryDataset(Dataset):
    
    def __init__(self, size=1000, initial_points_path=initial_points_path,
            controls_path=controls_path, trajectories_path=trajectories_path):

        self.size = size
        self.initial_points = np.genfromtxt(initial_points_path, delimiter=',')
        self.controls = np.genfromtxt(controls_path, delimiter=',')
        self.trajectories = np.genfromtxt(trajectories_path, delimiter=',')

        self.initial_points = torch.tensor(self.initial_points[:1000]).float()
        self.controls = torch.tensor(self.controls.reshape(1000, 2, 70)).float()
        self.control_norm = self.controls.max()
        self.controls = self.controls / self.control_norm
        self.trajectories = torch.tensor(self.trajectories.reshape(1000, 141, 3)).float()

        index_list = [2 * i for i in range(71)]
        self.trajectories = self.trajectories[:, index_list, :] 
        time_index = torch.arange(71).float().unsqueeze(0).unsqueeze(2)
        time_index = time_index.expand(1000, 71, 1)
        self.trajectories = torch.cat([self.trajectories, time_index], dim=2).contiguous()

    def __len__(self):
        #return self.inital_points.shape[0]
        return self.size

    def __getitem__(self, idx):
        return [self.initial_points[idx],
                self.controls[idx, :, :],
                self.trajectories[idx, :-1, :]]

if __name__ == "__main__":

    bds = BrockettTrajectoryDataset(size=50)
    print(bds.controls[:, 0, 0].std())
    print(bds.controls[:, 0, 0].mean())
    print(bds.controls[:, 1, 0].std())
    print(bds.controls[:, 1, 0].mean())
    num_samples = 2 
    indices = [randint(0, len(bds)) for i in range(num_samples)]
    #print("{}, {}, {}".format(initial_points.shape, controls.shape, trajectories.shape))
    for index in indices:
        fig = plt.figure(figsize=plt.figaspect(0.5))
        _, _, trajectories = bds[index]
        ax = plt.axes(projection='3d')
        ax.plot(trajectories[:, 0].numpy(), trajectories[:, 1].numpy(), trajectories[:, 2].numpy(), '-b')
        plt.show()
