from .networks import *
from .utils import *

import torch
import threading
import numpy as np
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast

class Agent:
    def __init__(self, state_size, action_size, config, device):
        self.state_size = state_size
        self.action_size = action_size
        self.eps = 1e-10
        self.learning_steps = config["learner"]["network"]["learning_steps"]
        self.clip = config["learner"]["network"]["clip"]
        self.lr = config["learner"]["network"]["lr"]
        self.seed = config["learner"]["utils"]["seed"]
        self.max_grad_norm = 1.0
        self.device = device

        # Networks
        self.Actor = ActorNetwork(state_size=state_size, action_size=action_size, device=device).to(
            device
        )
        self.Critic = CriticNetwork(state_size=state_size).to(device)

        # Optimizers
        self.actor_optim = Adam(self.Actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.Critic.parameters(), lr=self.lr)

        # Threadlocks
        self._weights_lock = threading.Lock()
        self.synced = False
    
    def eval_mode(self):
        self.Actor.eval()
        self.Critic.eval()
    
    def train_mode(self):
        self.Actor.train()
        self.Critic.train()

    def act(self, state, prev_actions):
        prev_actions = torch.Tensor(prev_actions).to(self.device)
        state = torch.Tensor(state).to(self.device)
        prev_actions = torch.unsqueeze(prev_actions, dim=0)
        state = torch.unsqueeze(state, dim=0)

        # Get action and state's value
        with self._weights_lock:
            self.eval_mode()
            with torch.no_grad():
                action, log_prob = self.Actor(state, prev_actions)
                value = self.Critic(state)
            self.train_mode()

            action = torch.squeeze(action).cpu().detach().numpy()
            log_prob = log_prob.cpu().detach().numpy()
            value = value.cpu().detach().numpy()

        return action, log_prob, value

    def learn(self, minibatch):
        """
        Proximal Policy Optimization
        Pseudocode: https://spinningup.openai.com/en/latest/algorithms/ppo.html
        """
        states, actions, prev_actions, log_probs, rewards2go = minibatch

        with torch.no_grad():
            self.Critic.eval()
            values = self.Critic(states)
            self.Critic.train()
        
        values = torch.squeeze(values)

        # 5. Compute avantages
        advantages = rewards2go - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)
        advantages.to(self.device)

        # 6. Multi-step gradient descent
        with self._weights_lock:
            for _ in range(self.learning_steps):
                cur_values, cur_log_probs = self.evaluate(states, actions, prev_actions)

                # Compute the ratio between current probs and old probs
                ratios = torch.exp(cur_log_probs - log_probs)

                # PPO-Clip objective
                actor_loss = -torch.min(
                    ratios * advantages,
                    torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages,
                )
                actor_loss = actor_loss.mean()
                critic_loss = nn.MSELoss()(cur_values, rewards2go)

                self.actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Actor.parameters(), self.max_grad_norm)
                self.actor_optim.step()
                
                self.critic_optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Critic.parameters(), self.max_grad_norm)
                self.critic_optim.step()

    def evaluate(self, states, actions, prev_actions):
        cur_values = self.Critic(states)
        _, cur_log_probs = self.Actor(states, prev_actions, actions)

        cur_values = torch.squeeze(cur_values)
        cur_log_probs = torch.squeeze(cur_log_probs)
        return cur_values, cur_log_probs

    def save_weights(self):
        with self._weights_lock:
            torch.save(self.Actor.state_dict(), "actor_weight.pth")
            torch.save(self.Critic.state_dict(), "critic_weight.pth")

    def get_weights(self):
        with self._weights_lock:
            return self.Actor.state_dict(), self.Critic.state_dict()

    def sync(self, actor_weight, critic_weight):
        with self._weights_lock:
            self.Actor.load_state_dict(actor_weight)
            self.Critic.load_state_dict(critic_weight)
            self.synced = True
