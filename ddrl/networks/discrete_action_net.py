import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions
from torch.nn.modules.activation import ReLU


class BaseNet(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.gate = F.relu
        self.policy_head = nn.Linear(64, action_size)
        self.value_head = nn.Linear(64, 1)

    def forward(self, state):
        x = self.gate(self.fc1(state))
        x = self.gate(self.fc2(x))
        x = self.gate(self.fc3(x))
        return {
            "p": F.softmax(self.policy_head(x), dim=-1),
            "v": self.value_head(x)
        }
    
    def get_action(self, prob):
        dist = distributions.Categorical(prob)
        action = dist.sample()
        return action
    
    def get_best_action(self, prob):
        _, indices = prob.max(dim=1)
        return indices

    def eval_action(self, prob, action):
        dist = distributions.Categorical(prob)
        return {
            "log_prob": dist.log_prob(action),
            "entropy": dist.entropy()
        }

