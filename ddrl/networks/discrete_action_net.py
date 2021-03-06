import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions
from torch.nn.modules.activation import ReLU


class BaseNet(nn.Module):
    def __init__(self, state_size, action_size):
        """
        A shared network to learn the representation of game's states
        """
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        self.gate = nn.LeakyReLU()
        self.shared_net = nn.Sequential(
            nn.Linear(state_size, 256), self.gate,
            nn.Linear(256, 128), self.gate,
            nn.Linear(128, 64), self.gate,
        )
        
        # Different heads
        self.policy_head = nn.Linear(64, action_size)
        self.value_head = nn.Linear(64, 1)

    def forward(self, state):
        x = self.shared_net(state)
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

class BaseActor(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.gate = nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(state_size, 256), self.gate,
            nn.Linear(256, 128), self.gate,
            nn.Linear(128, 64), self.gate,
            nn.Linear(64, action_size),
            nn.Softmax(dim=-1),
        )

    def forward(self, state):
        return self.net(state)
    
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

class BaseCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.gate = nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(state_size, 256), self.gate,
            nn.Linear(256, 128), self.gate,
            nn.Linear(128, 64), self.gate,
            nn.Linear(64, 1)
        )
    
    def forward(self, state):
        return self.net(state)

