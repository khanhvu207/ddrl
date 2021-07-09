from .networks import *
from .utils import *

import gym
import math
import time
import torch
import threading
import numpy as np
from collections import deque

from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast


class Agent:
    def __init__(self, state_size, action_size, config, device, neptune):
        self.state_size = state_size
        self.action_size = action_size
        self.eps = 1e-10
        self.learning_steps = config["learner"]["network"]["learning_steps"]
        self.clip = config["learner"]["network"]["clip"]
        self.lr = config["learner"]["network"]["lr"]
        self.seed = config["learner"]["utils"]["seed"]
        self.max_grad_norm = 1.0
        self.device = device

        self.env_name = config["env"]["env-name"]
        self.env = gym.make(self.env_name)
        self.gru_seq_len = config["learner"]["network"]["gru_seq_len"]
        self.max_t = config["worker"]["max_t"]
        self.best_score = -math.inf

        # Networks
        self.Actor = ActorNetwork(
            state_size=state_size, action_size=action_size, device=device
        ).to(device)
        self.Critic = CriticNetwork(state_size=state_size).to(device)

        # Optimizers
        self.actor_optim = Adam(self.Actor.parameters(), lr=self.lr, weight_decay=1e-5)
        self.critic_optim = Adam(
            self.Critic.parameters(), lr=self.lr, weight_decay=1e-5
        )

        # Threadlocks
        self._weights_lock = threading.Lock()
        self.synced = False

        # Neptune
        self.neptune = neptune

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
        states, actions, prev_actions, log_probs, vs, advantages = minibatch

        for _ in range(self.learning_steps):

            cur_values, cur_log_probs = self.compute(states, actions, prev_actions)

            ratios = torch.exp(cur_log_probs - log_probs)

            # PPO-Clip objective
            actor_loss = -torch.min(
                ratios * advantages,
                torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages,
            )
            actor_loss = actor_loss.mean()
            critic_loss = nn.MSELoss()(cur_values, vs)

            with self._weights_lock:
                self.actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.Actor.parameters(), self.max_grad_norm
                )
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.Critic.parameters(), self.max_grad_norm
                )
                self.critic_optim.step()

    def compute(self, states, actions, prev_actions):
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

    def evaluate(self):
        scores = []
        for _ in range(5):
            score = 0
            state = self.env.reset()
            prev_actions = deque(
                [np.zeros(self.action_size) for _ in range(self.gru_seq_len)],
                maxlen=self.gru_seq_len,
            )

            for t in range(self.max_t):
                action, log_prob, value = self.act(state, prev_actions=prev_actions)
                observation, reward, done, info = self.env.step(action)
                score += reward
                state = observation
                if done:
                    break

            scores.append(score)

        mean_score = np.mean(scores)
        print(f"Evaluation score: {mean_score}")
        if self.neptune is not None:
            self.neptune["eval/score"].log(mean_score)
        if mean_score > self.best_score:
            print(f"The agent has improved from the last evaluation! Saving weight...")
            self.best_score = mean_score
            self.save_weights()
