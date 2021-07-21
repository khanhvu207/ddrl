from .networks.lunar_net import *
from .utils import *

import gym
import math
import time
import torch
import threading
import numpy as np
from collections import deque

from torch.optim import Adam


class Trainer:
    def __init__(self, state_size, action_size, config, device, neptune):
        self.state_size = state_size
        self.action_size = action_size
        self.eps = 1e-10

        self.learning_steps = config["learner"]["network"]["learning_steps"]
        self.clip = config["learner"]["network"]["clip"]
        self.actor_lr = config["learner"]["network"]["actor_lr"]
        self.critic_lr = config["learner"]["network"]["critic_lr"]
        self.entropy_regularization = config["learner"]["network"][
            "entropy_regularization"
        ]
        self.entropy_regularization_decay = config["learner"]["network"][
            "entropy_regularization_decay"
        ]
        self.max_grad_norm = config["learner"]["network"]["max_grad_norm"]
        self.device = device

        self.env_name = config["env"]["env-name"]
        self.env = gym.make(self.env_name)
        self.rnn_seq_len = config["learner"]["network"]["rnn_seq_len"]
        self.max_t = config["worker"]["max_t"]
        self.best_score = -math.inf
        self.means = []
        self.scores_window = deque(maxlen=100)

        # Networks
        self.Actor = ActorNetwork(
            state_size=state_size, action_size=action_size, device=device
        ).to(device)
        self.Critic = CriticNetwork(state_size=state_size).to(device)

        # Initialize weights
        # self.Actor._init_weights_and_bias()
        # self.Critic._init_weights_and_bias()

        # Optimizers
        self.actor_optim = Adam(
            self.Actor.parameters(), lr=self.actor_lr, weight_decay=1e-5
        )
        self.critic_optim = Adam(
            self.Critic.parameters(), lr=self.critic_lr, weight_decay=1e-5
        )

        # Critic loss
        self.critic_loss = nn.MSELoss()  # nn.SmoothL1Loss()

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
            with torch.no_grad():
                self.eval_mode()
                action, log_prob, _ = self.Actor(state, prev_actions)
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

            cur_values, cur_log_probs, dist_entropy = self.compute(
                states, actions, prev_actions
            )

            ratios = torch.exp(cur_log_probs - log_probs)

            # PPO-Clip objective
            actor_loss = -torch.min(
                ratios * advantages,
                torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages,
            )
            entropy_loss = -self.entropy_regularization * dist_entropy
            actor_loss = actor_loss + entropy_loss
            actor_loss = actor_loss.mean()
            critic_loss = 0.5 * self.critic_loss(cur_values, vs)

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

        self.entropy_regularization = max(
            0.0, self.entropy_regularization * self.entropy_regularization_decay
        )

    def compute(self, states, actions, prev_actions):
        cur_values = self.Critic(states)
        _, cur_log_probs, entropy = self.Actor(states, prev_actions, actions)

        cur_values = torch.squeeze(cur_values)
        cur_log_probs = torch.squeeze(cur_log_probs)
        return cur_values, cur_log_probs, entropy

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
        for _ in range(1):
            score = 0
            state = self.env.reset()
            prev_actions = deque(
                [np.zeros(self.action_size) for _ in range(self.rnn_seq_len)],
                maxlen=self.rnn_seq_len,
            )

            for t in range(self.max_t):
                action, log_prob, value = self.act(state, prev_actions=prev_actions)
                observation, reward, done, info = self.env.step(action)
                score += reward
                state = observation
                if done:
                    break

            self.scores_window.append(score)

        mean_score = np.mean(self.scores_window)
        self.means.append(mean_score)
        print(f"Evaluation score: {mean_score}")
        if self.neptune is not None:
            self.neptune["eval/score"].log(mean_score)