import torch.nn as nn
import torch


class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class PINN(nn.Module):
    def __init__(self, input_size=2, hidden_sizes=(64,64,64), output_size=1, activation=nn.Tanh()):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_sizes[0]), activation]
        for i in range(len(hidden_sizes)-1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(activation)
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, xy):
        return self.net(xy)
