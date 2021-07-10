import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        device,
        seed=2021,
    ):
        super(ActorNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.device = device
        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, action_size)
        self.gate = F.relu

        self.lstm = nn.LSTM(input_size=action_size, hidden_size=64, batch_first=True)

        self.cov_var = torch.full(size=(action_size,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var).to(device)

    def forward(self, state, prev_actions, action=None):
        # Action head
        x = self.gate(self.fc1(state))
        x = self.gate(self.fc2(x))
        x = self.gate(self.fc3(x))
        z = torch.tanh(self.fc4(x))

        # Action context head
        # y, _ = self.lstm(prev_actions)
        # y = y[:, -1, :]

        # Concat
        # z = torch.cat([x, y], dim=1)
        # z = torch.tanh(self.fc4(z))
        
        dist = torch.distributions.MultivariateNormal(z, self.cov_mat)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob


class CriticNetwork(nn.Module):
    def __init__(
        self, state_size, seed=2021
    ):
        super(CriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)
        self.gate = F.relu

    def forward(self, state):
        x = self.gate(self.fc1(state))
        x = self.gate(self.fc2(x))
        x = self.gate(self.fc3(x))
        x = self.fc4(x)
        return x
