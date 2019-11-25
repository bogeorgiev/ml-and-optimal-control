import torch
import sys
sys.path.append("..")
import torch.optim as optim
import torch.nn as nn
import argparse
from nn_models.mlp_models import MLPBrockett
from nn_models.dataloader import BrockettTrajectoryDataset
from utilities.visualize import plot_trajectory_3d, trajectory

model_load_path = "path to trained model"
dataset_size = 800
traj_size = 2
l1 = 1
l2 = 10
device = 'cuda'

model = MLPBrockett(hidden_nodes=70).to(device)
checkpoint = torch.load(model_load_path)
model.load_state_dict(checkpoint['model_state_dict'])
loss_func = nn.MSELoss()

ds = BrockettTrajectoryDataset(size=dataset_size)
control_norm = ds.control_norm
loader = torch.utils.data.DataLoader(ds, batch_size=dataset_size, shuffle=True)

it = iter(loader)
init_points, controls, traj = next(it)

init_points = init_points.to(device)
inputs = traj[:, :traj_size, :].contiguous().to(device)
inputs = inputs.view(-1, 4)
output = model(inputs)
output = output.view(dataset_size, 2, -1)

target = controls[:, :, :traj_size].to(device)
predicted_traj = trajectory(output * control_norm, init_points, device=device,
                        num_points=traj_size)
inputs = inputs[:, :-1]
inputs = inputs.view(dataset_size, traj_size, -1)
loss = l1 * loss_func(output, target) + l2 * loss_func(inputs, predicted_traj)

pw_diff = inputs - predicted_traj #B x pts x dim
pw_diff = pw_diff.norm(dim=2)
print(pw_diff.mean(dim=0))
print(pw_diff.std(dim=0))

traj = inputs
for sample in range(0):
    plot_trajectory_3d(pred_traj=predicted_traj[sample, :, :].detach().cpu().numpy(),
                            target_traj=traj[sample, :traj_size, :].cpu().numpy(), 
                            save_path="./img/MLP_trajectory.png", show=True)
