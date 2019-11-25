import numpy as np
import torch
import sys
sys.path.append("..")
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
from mlp_models import MLPBrockett, LSTMBrockett
from dataloader import BrockettTrajectoryDataset
from logger import Logger
import time
import tensorflow as tf
from utilities.visualize import plot_trajectory_3d, trajectory, get_img_from_fig

step = 0

def train(args, model, device, train_loader, optimizer, loss_fn, epoch, logger, exp_name):
    control_norm = train_loader.dataset.control_norm
    model.train()
    for batch_idx, data in enumerate(train_loader):

        init_points, controls, traj = data
        optimizer.zero_grad()

        init_points = init_points.to(device)
        inputs = traj[:, args.traj_start:args.traj_size, :-1].contiguous().to(device)
        init_points = traj[:, args.traj_start, :-1].contiguous().to(device)
        output = model(inputs)
        output = output.view(args.batch_size, 2, -1)

        target = controls[:, :, args.traj_start:args.traj_size].to(device)
        predicted_traj = trajectory(output * control_norm, init_points, device=device,
                                num_points=args.traj_size-args.traj_start)
        loss = args.l1 * loss_fn(output, target) + args.l2 * loss_fn(inputs, predicted_traj)
        
        loss.backward()
        optimizer.step()
        global step
        step = step + 1
        if (step + 1) % 50 == 0 and args.enable_logging:
            info = {"loss_steps_" + exp_name: loss.item()}
            for tag, value in info.items():
                logger.add_scalar(tag, value, step + 1)
                #print(predicted_traj[0, :, :].detach().cpu().numpy().shape)
                traj = inputs
                #print(traj[0, :, :].cpu().numpy().shape)
                buff = plot_trajectory_3d(pred_traj=predicted_traj[0, :, :].detach().cpu().numpy(),
                        target_traj=traj[0, :, :].cpu().numpy(), 
                        save_path="./img/trajectory_" + exp_name + ".png", buff=True)
                logger.add_image("trajectories" + exp_name, buff, step + 1, dataformats='HWC') 

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, loss_fn, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Brockett LSTM model')

    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=40, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--dataset-size', type=int, default=100,
                        help='size of train dataset')
    parser.add_argument('--l1', type=float, default=1.0e1)
    parser.add_argument('--l2', type=float, default=1.0e2)
    parser.add_argument('--traj-start', type=int, default=0,
                        help='trajectory starting index')
    parser.add_argument('--traj-size', type=int, default=60,
                        help='number of trajectory points')
    parser.add_argument('--epochs', type=int, default=300000, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1.0e-5, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--enable-logging', type=bool, default=True,
                        help='enables results plotting and tensorboard')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='enable model saving')
    parser.add_argument('--load-model', type=bool, default=False,
                        help='loads a model from model load path')
    parser.add_argument('--model-load-path', type=str,
            default='./saved_models/brockett_lstm_predictor1574258572.8664896.pt')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    exp_name = str(time.time())

    Brockett_ds = BrockettTrajectoryDataset(size=args.dataset_size) 
    logger = SummaryWriter("./logs")
           
    train_loader = torch.utils.data.DataLoader(Brockett_ds,
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(Brockett_ds,
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = LSTMBrockett(input_size=3, hidden_size=10, output_size=2, device=device).to(device)

    loss_fn = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print(args)
    #args.load_model = False
    if args.load_model:
        checkpoint = torch.load(args.model_load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        #optimizer.load_state_dict(checkpoint['optim_state_dict'])


    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, loss_fn, epoch,
            logger, exp_name)

        if (args.save_model):
            torch.save({
                    "model_state_dict" : model.state_dict(),
                    "optim_state_dict" : optimizer.state_dict(),
                    }, "./saved_models/brockett_lstm_predictor" + exp_name + ".pt")

        #test(args, model, device, loss_fn, test_loader) 
    
if __name__=="__main__":
    main()
