import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random

class LSTMBrockett(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(LSTMBrockett, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device
        self.lstm_cell = nn.LSTMCell(self.input_size, self. hidden_size)
       	self.hidden_nodes = 30
        self.l1 = nn.Linear(self.hidden_size, self.hidden_nodes, bias=True)
        self.l2 = nn.Linear(self.hidden_nodes, self.hidden_nodes, bias=False)
        self.l3 = nn.Linear(self.hidden_nodes, self.hidden_nodes, bias=False)
        self.l4 = nn.Linear(self.hidden_nodes, self.hidden_nodes, bias=False)
        self.l5 = nn.Linear(self.hidden_nodes, 2, bias=False)
        self.batch_norm = nn.BatchNorm1d(self.hidden_nodes)
       	self.relu = nn.ReLU() 
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.mlp = nn.Sequential(self.l1, self.batch_norm, self.relu,
                                self.l2, self.batch_norm, self.relu,
                                self.l3, self.batch_norm, self.relu,
                                self.l4, self.batch_norm, self.relu,
                                self.l5)


    def forward(self, x, future=0, y=None):
        outputs = []
        h_t = torch.zeros(x.size(0), self.hidden_size).float().to(self.device)
        c_t = torch.zeros(x.size(0), self.hidden_size).float().to(self.device)

        time_steps = x.shape[1]
        for time_step in range(time_steps):
            h_t, c_t = self.lstm_cell(x[:, time_step, :], (h_t, c_t))
            output = self.mlp(h_t)
            outputs += [output.unsqueeze(2)]

        return torch.cat(outputs, dim=2)

class MLPBrockett(nn.Module):
    def __init__(self, hidden_nodes=40, input_size=4):
        super(MLPBrockett, self).__init__()
        self.hidden_nodes = hidden_nodes
        self.input_size = input_size
        self.l1 = nn.Linear(self.input_size, self.hidden_nodes, bias=True)
        self.l2 = nn.Linear(self.hidden_nodes, self.hidden_nodes, bias=False)
        self.l3 = nn.Linear(self.hidden_nodes, self.hidden_nodes, bias=False)
        self.l4 = nn.Linear(self.hidden_nodes, self.hidden_nodes, bias=False)
        self.l5 = nn.Linear(self.hidden_nodes, 2, bias=False)
        self.batch_norm = nn.BatchNorm1d(self.hidden_nodes)
        self.relu = nn.ReLU() 
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.mlp = nn.Sequential(self.l1, self.batch_norm, self.relu,
                                self.l2, self.batch_norm, self.relu,
                                self.l3, self.batch_norm, self.relu,
                                self.l4, self.batch_norm, self.relu,
                                self.l5)

    def forward(self, x):
        x = self.mlp(x)
        return x


def loss_func(predicted, target):
    """
        predicted: B x 2
        target: B x 2
    """
    pass
