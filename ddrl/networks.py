import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ActorNetwork(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        seed=2021,
        hidden_size1=128,
        hidden_size2=128,
        gru_layers=1,
    ):
        super(ActorNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.fc1 = nn.Linear(state_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2 * 2, action_size)

        self.gru_layers = gru_layers
        self.gru = nn.GRU(input_size=action_size, hidden_size=hidden_size2, num_layers=gru_layers, batch_first=True)

        self.cov_var = torch.full(size=(action_size,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var).to(device)

    def forward(self, state, prev_actions, action=None):
        # Action head
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        # Action context head
        h0 = torch.zeros(self.gru_layers, prev_actions.size(0), self.hidden_size2).to(device)
        y, _ = self.gru(prev_actions, h0)
        y = y[:, -1, :]

        # Concat [x, y]
        z = torch.cat([x, y], dim=1)
        z = torch.tanh(self.fc3(z))
        
        dist = torch.distributions.MultivariateNormal(z, self.cov_mat)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob


class CriticNetwork(nn.Module):
    def __init__(
        self, state_size, seed=2021, hidden_size1=256, hidden_size2=256
    ):
        super(CriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
