import torch
from torch import nn


class Linear(nn.Module):
    def __init__(self, NN_dims: list):
        super().__init__()
        self.NN_dims = NN_dims

        layers = []
        for i in range(len(NN_dims) - 2):
            layers.append(nn.Linear(NN_dims[i], NN_dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(NN_dims[-2], NN_dims[-1]))
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x


class Linear_mix(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

        self.layer_stack = nn.Sequential(
            *[nn.Linear(input_dim, 2000),
              nn.ReLU(),
              nn.Linear(2000, 2000),
              nn.Tanh(),
              nn.Linear(2000, 2000),
              nn.Tanh(),
              nn.Linear(2000, 1)
              ]
        )

    def forward(self, x):
        x = self.layer_stack(x)
        return x
