from .networks import *
from .utils import *

import torch
import threading
import numpy as np
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self, state_size, action_size, config):
        self.state_size = state_size
        self.action_size = action_size
        self.eps = 1e-10
        self.learning_steps = config["learner"]["network"]["learning_steps"]
        self.clip = config["learner"]["network"]["clip"]
        self.lr = config["learner"]["network"]["lr"]
        self.seed = config["learner"]["utils"]["seed"]
        self.max_grad_norm = 1.0

        # Networks
        self.Actor = ActorNetwork(state_size=state_size, action_size=action_size).to(
            device
        )
        self.Critic = CriticNetwork(state_size=state_size).to(device)

        # Optimizers
        self.actor_optim = Adam(self.Actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.Critic.parameters(), lr=self.lr)

        # Threadlocks
        self._weights_lock = threading.Lock()
        self.synced = False

    def act(self, state):
        state = torch.Tensor(state).to(device)
        state = torch.unsqueeze(state, dim=0)

        # Get action and state's value
        with self._weights_lock:
            self.Actor.eval()
            self.Critic.eval()
            with torch.no_grad():
                action, log_prob = self.Actor(state)
                value = self.Critic(state)
            self.Actor.train()
            self.Critic.train()

            action = torch.squeeze(action).cpu().detach().numpy()
            log_prob = log_prob.cpu().detach().numpy()
            value = value.cpu().detach().numpy()

        return action, log_prob, value

    def learn(self, minibatch):
        """
        Proximal Policy Optimization
        Pseudocode: https://spinningup.openai.com/en/latest/algorithms/ppo.html
        """
        states, actions, log_probs, rewards2go = minibatch

        with torch.no_grad():
            self.Critic.eval()
            values = self.Critic(states)
            self.Critic.train()
        
        values = torch.squeeze(values)

        # 5. Compute avantages
        advantages = rewards2go - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)
        advantages.to(device)

        # 6. Multi-step gradient descent
        with self._weights_lock:
            for _ in range(self.learning_steps):
                cur_values, cur_log_probs = self.evaluate(states, actions)

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

    def evaluate(self, states, actions):
        cur_values = self.Critic(states)
        _, cur_log_probs = self.Actor(states, actions)
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
            print("Syncing...")
            self.Actor.load_state_dict(actor_weight)
            self.Critic.load_state_dict(critic_weight)
            self.synced = True
