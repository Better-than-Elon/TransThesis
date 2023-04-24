import torch
import torch.nn as nn
from torch.autograd import Variable


class ATRPOAgent(nn.Module):
    def __init__(self, observ_shape, n_actions):
        super(ATRPOAgent, self).__init__()
        self.linear = nn.Sequential(nn.Linear(observ_shape, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())

        self.action_mean = nn.Linear(64, n_actions)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.log_std = nn.Parameter(torch.zeros(1, n_actions))

    def forward(self, x):
        x = self.linear(x)

        action_mean = self.action_mean(x)
        action_log_std = self.log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std

    def act(self, x):
        with torch.no_grad():
            action_mean, _, action_std = self.forward(torch.tensor(x).unsqueeze(0))
            action = torch.normal(action_mean, action_std)
        return action


class Value(nn.Module):
    def __init__(self, observ_shape):
        super(Value, self).__init__()
        self.linear = nn.Sequential(nn.Linear(observ_shape, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())

        self.value_head = nn.Linear(64, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        return self.value_head(self.linear(x))
