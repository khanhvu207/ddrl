import torch
from torch import distributions
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, device):
        super(ActorNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(128, action_size)
        self.fc5 = nn.Linear(128, action_size)
        self.gate = torch.tanh

        self.lstm = nn.LSTM(input_size=action_size, hidden_size=64, batch_first=True)
        self.std_offset = 0
        self.std_controlling_minimal = 0

    def _init_weights_and_bias(self):
        for name, layer in self._modules.items():
            if "lstm" in name:
                continue
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, state, prev_actions, action=None):
        x = self.gate(self.fc1(state))
        x = self.gate(self.fc2(x))
        x = self.gate(self.fc3(x))
        y, _ = self.lstm(prev_actions)
        y = y[:, -1, :]
        z = torch.cat([x, y], dim=1)
        mean, std_logit = self.fc4(z), self.fc5(z)
        std = nn.Softplus()(std_logit + self.std_offset) + self.std_controlling_minimal
        dist = torch.distributions.Normal(loc=mean, scale=std, validate_args=False)
        if action is None:
            action = torch.tanh(dist.sample())
        log_prob = dist.log_prob(value=action).sum(dim=1)
        entropy = dist.entropy().sum(dim=1)
        return action, log_prob, entropy


class CriticNetwork(nn.Module):
    def __init__(self, state_size, seed=2021):
        super(CriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.gate = torch.tanh

    def _init_weights_and_bias(self):
        for name, layer in self._modules.items():
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, state):
        x = self.gate(self.fc1(state))
        x = self.gate(self.fc2(x))
        x = self.gate(self.fc3(x))
        x = self.fc4(x)
        return x