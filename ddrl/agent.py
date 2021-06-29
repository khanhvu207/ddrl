from .models import *
from .utils import *

import torch
import numpy as np
from torch.optim import Adam

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparams
        self.gamma = 0.99
        self.eps = 1e-10
        self.learning_steps = 5
        self.clip = 0.15
        self.lr = 0.0001

        # Networks
        self.Actor = ActorNetwork(state_size=state_size, action_size=action_size).to(device)
        self.Critic = CriticNetwork(state_size=state_size).to(device)

        # Optimizers
        self.actor_optim = Adam(self.Actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.Critic.parameters(), lr=self.lr)

        # Exploration noise
        self.ou_noise = OUNoise(size=action_size, seed=2021)

    def noise_reset(self):
        self.ou_noise.reset()

    def act(self, state, is_train=False):
        state = torch.Tensor(state).to(device)
        state = torch.unsqueeze(state, dim=0)
        
        # Get action and state's value
        self.Actor.eval()
        self.Critic.eval()
        with torch.no_grad():
            action, log_prob = self.Actor(state)
            value = self.Critic(state)
        self.Actor.train()
        self.Critic.train()

        action = torch.squeeze(action).cpu().detach().numpy()
        if is_train: 
            action += self.ou_noise.sample()
        log_prob = log_prob.cpu().detach().numpy()
        value = value.cpu().detach().numpy()
        return action, log_prob, value
    
    def learn(self, minibatch):
        """
            Proximal Policy Optimization
            Pseudocode: https://spinningup.openai.com/en/latest/algorithms/ppo.html
        """
        states, actions, log_probs, values, rewards2go  = minibatch
        
        # 5. Compute avantages
        advantages = rewards2go - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)
        advantages.to(device)
        
        # 6. Multi-step gradient descent
        for _ in range(self.learning_steps):
            cur_values, cur_log_probs = self.evaluate(states, actions)

            # Compute the ratio between current probs and old probs
            ratios = torch.exp(cur_log_probs - log_probs)

            # PPO-Clip objective
            actor_loss = -torch.min(ratios * advantages, torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages)
            actor_loss = actor_loss.mean()
            critic_loss = nn.MSELoss()(cur_values, rewards2go)

            # Backpropagation & Gradient descent
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

    def evaluate(self, states, actions):
        cur_values = self.Critic(states)
        _, cur_log_probs = self.Actor(states, actions)
        cur_values = torch.squeeze(cur_values)
        cur_log_probs = torch.squeeze(cur_log_probs)
        return cur_values, cur_log_probs

    def compute_reward_to_go(self, rewards):
        rw2go = []
        for eps_reward in reversed(rewards):
            discounted_reward = 0
            for reward in reversed(eps_reward):
                discounted_reward = reward + discounted_reward * self.gamma
                rw2go.append(discounted_reward)
        return torch.Tensor(rw2go[::-1])
    
    def save_weights(self):
        torch.save(self.Actor.state_dict(), 'actor_weight.pth')
        torch.save(self.Critic.state_dict(), 'critic_weight.pth')
    
    def get_weights(self):
        return self.Actor.state_dict(), self.Critic.state_dict()

    def sync(self, actor_weight, critic_weight):
        self.Actor.load_state_dict(actor_weight)
        self.Critic.load_state_dict(critic_weight)

        
