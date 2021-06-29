import torch
import random
from collections import deque, namedtuple

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Batcher:
    def __init__(self, config):
        self.buffer_size = config['learner']['batcher']['buffer_size']
        self.gamma = config['learner']['batcher']['gamma']
        self.batch_size = config['learner']['network']['batch_size']

        self.memory = deque(maxlen=self.buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "log_prob", "reward", "value", "done"])
        
    def add(self, trajectory):
        rw2go = self._compute_reward_to_go(trajectory['rewards'])
        for i in range(len(rw2go)):
            e = self.experience(trajectory['states'][i], trajectory['actions'][i], trajectory['log_probs'][i], rw2go[i], trajectory['values'][i], trajectory['dones'][i])
            self.memory.append(e)
        print("Buffer updated")

    def __len__(self):
        return len(self.memory)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.Tensor([e.state for e in experiences if e is not None]).to(device)
        actions = torch.Tensor([e.action for e in experiences if e is not None]).to(device)
        log_probs = torch.Tensor([e.log_prob for e in experiences if e is not None]).to(device)
        values = torch.Tensor([e.value for e in experiences if e is not None]).to(device)
        rewards = torch.Tensor([e.reward for e in experiences if e is not None]).to(device)

        values = torch.squeeze(values)
        log_probs = torch.squeeze(log_probs)
        return states, actions, log_probs, values, rewards
        

    def _compute_reward_to_go(self, rewards):
        rw2go = []
        for eps_reward in reversed(rewards):
            discounted_reward = 0
            for reward in reversed(eps_reward):
                discounted_reward = reward + discounted_reward * self.gamma
                rw2go.append(discounted_reward)
        return rw2go[::-1]
    

