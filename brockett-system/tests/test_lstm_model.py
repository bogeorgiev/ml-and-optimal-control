import torch
import sys
sys.path.append("..")
import torch.optim as optim
import torch.nn as nn
import argparse
from nn_models.mlp_models import MLPBrockett, LSTMBrockett
from nn_models.dataloader import BrockettTrajectoryDataset
from utilities.visualize import plot_trajectory_3d, trajectory, plot_controls
from matplotlib import pyplot as plt

#model_load_path = "path to trained model"
dataset_size = 800
batch_size = 800
traj_start = 0
traj_size = 20
l1 = 1
l2 = 10
device = 'cuda'

model = LSTMBrockett(input_size=3, hidden_size=10, output_size=2, device=device).to(device)
checkpoint = torch.load(model_load_path)
model.load_state_dict(checkpoint['model_state_dict'])
loss_fn = nn.MSELoss()

ds = BrockettTrajectoryDataset(size=dataset_size)
control_norm = ds.control_norm
loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)

it = iter(loader)
init_points, controls, traj = next(it)

init_points = init_points.to(device)
inputs = traj[:, traj_start:traj_size, :-1].contiguous().to(device)
init_points = traj[:, traj_start, :-1].contiguous().to(device)
output = model(inputs)
output = output.view(batch_size, 2, -1)

target = controls[:, :, traj_start:traj_size].to(device)
predicted_traj = trajectory(output * control_norm, init_points, device=device,
                        num_points=traj_size-traj_start)
loss = l1 * loss_fn(output, target) + l2 * loss_fn(inputs, predicted_traj)

pw_diff = inputs - predicted_traj #B x pts x dim
pw_diff = pw_diff.norm(dim=2)
print(pw_diff.mean(dim=0))
print(pw_diff.std(dim=0))

traj = inputs
for sample in range(0):
    plot_trajectory_3d(pred_traj=predicted_traj[sample, :, :].detach().cpu().numpy(),
            target_traj=traj[sample, :, :].cpu().numpy(), 
            save_path="./img/init_LSTM_trajectory" + str(sample) + ".png", show=False)
    plot_controls(output[sample, :, :].detach().cpu().numpy(), target[sample, :, :].detach().cpu().numpy(), save_path="./img/init_LSTM_controls" + str(sample) + ".png", show=False)
    plt.close()
