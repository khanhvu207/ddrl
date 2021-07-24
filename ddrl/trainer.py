from .networks.discrete_action_net import BaseActor, BaseCritic
from .utils import OUNoise

import os
import sys
import gym
import math
import time
import torch
import pickle
import threading
import numpy as np
from datetime import datetime
from collections import deque

import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler


class Trainer:
    def __init__(self, state_size, action_size, config, device, neptune):
        self.state_size = state_size
        self.action_size = action_size
        self.neptune = neptune
        # Configs
        self.env_config = config["env"]
        self.batcher_config = config["batcher"]
        self.learner_config = config["learner"]

        self.seed = self.learner_config["seed"]
        self.learning_steps = self.learner_config["learning_steps"]
        self.clip = self.learner_config["clip"]
        self.lr = self.learner_config["lr"]
        self.lr_decay = self.learner_config["lr_decay"]
        self.lr_decay_every = self.learner_config["lr_decay_every"]
        self.entropy_regularization = self.learner_config["entropy_regularization"]
        self.entropy_regularization_decay = self.learner_config[
            "entropy_regularization_decay"
        ]
        self.max_grad_norm = self.learner_config["max_grad_norm"]
        self.solve_score = self.learner_config["solve_score"]
        self.device = device

        self.env_name = self.env_config["env-name"]
        self.env = gym.make(self.env_name)
        self.env.seed(self.seed)

        # Neptune.ai logging
        if self.neptune is not None:
            self.neptune["lr_decay"] = self.learner_config["lr_decay"]
            self.neptune["batch size"] = self.batcher_config["batch_size"]
            self.neptune["learning steps"] = self.learner_config["learning_steps"]
            self.neptune["loss target"] = self.learner_config["loss_target"]
            self.neptune["lambda"] = self.learner_config["lambda"]
            self.neptune["gamma"] = self.learner_config["gamma"]
            self.neptune["policy clip"] = self.learner_config["clip"]

        # Evaluation metrics
        self.start_time = time.time()
        self.best_score = -math.inf
        self.scores = []
        self.means = []
        self.timepoints = []
        self.scores_window = deque(maxlen=100)
        self.save_dir = self.learner_config["result_dir"]

        # Networks
        self.actor = BaseActor(state_size=state_size, action_size=action_size).to(
            device
        )
        self.critic = BaseCritic(state_size=state_size, action_size=action_size).to(
            device
        )

        # Noise
        self.noise = OUNoise(size=action_size, seed=config["learner"]["seed"])

        # Optimizers
        self.actor_optim = Adam(self.actor.parameters(), lr=2e-4)
        self.critic_optim = Adam(self.critic.parameters(), lr=2e-3)

        # Scheduler
        # self.scheduler = lr_scheduler.StepLR(
        #     self.optim, step_size=self.lr_decay_every, gamma=self.lr_decay
        # )

        # Critic loss
        self.critic_loss = nn.MSELoss()  # nn.SmoothL1Loss()

        # Threadlocks
        self._weights_lock = threading.Lock()
        self.synced = False

        # Neptune
        self.neptune = neptune

    def eval_mode(self):
        self.actor.eval()
        self.critic.eval()

    def train_mode(self):
        self.actor.train()
        self.critic.train()

    def act(self, state):
        state = torch.Tensor(state).unsqueeze(0).to(self.device)
        with self._weights_lock:
            with torch.no_grad():
                self.eval_mode()
                logit = {
                    "p": self.actor(state),
                    "v": self.critic(state)
                }
                value = logit["v"]
                action = self.actor.get_action(logit["p"])
                log_prob = self.actor.eval_action(logit["p"], action)["log_prob"]
                self.train_mode()
            action = action.cpu().detach().numpy()
            log_prob = log_prob.cpu().detach().numpy()
            value = value.cpu().detach().numpy()
        return action, log_prob, value

    def learn(self, minibatch):
        """
        Proximal Policy Optimization
        Pseudocode: https://spinningup.openai.com/en/latest/algorithms/ppo.html
        """
        states, actions, log_probs, vs, advantages = minibatch

        # Normalize advantages is a good idea?
        # https://costa.sh/blog-the-32-implementation-details-of-ppo.html
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        for _ in range(self.learning_steps):
            cur_values, cur_log_probs, entropy = self.compute(states, actions)

            # Compute the policy loss
            ratios = torch.exp(cur_log_probs - log_probs)
            surrogate_loss1 = ratios * advantages
            surrogate_loss2 = (
                torch.clamp(ratios, 1.0 - self.clip, 1.0 + self.clip) * advantages
            )
            actor_loss = torch.min(surrogate_loss1, surrogate_loss2)

            # Optimize the policy loss along with the entropy term to encourage exploration
            actor_loss = actor_loss + self.entropy_regularization * entropy
            actor_loss = actor_loss.mean()
            actor_loss = -actor_loss

            # MSE loss
            critic_loss = self.critic_loss(cur_values, vs)

            # Compose the total loss 
            # Maximize: actor_loss
            # Minimize: critic_loss
            # total_loss = -actor_loss + critic_loss

            with self._weights_lock:
                self.actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.max_grad_norm
                )
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm
                )
                self.critic_optim.step()

            if self.neptune is not None:
                self.neptune["actor loss"].log(actor_loss)
                self.neptune["critic loss"].log(critic_loss)

        # self.scheduler.step()
        self.entropy_regularization = max(
            0.0, self.entropy_regularization * self.entropy_regularization_decay
        )
        if self.neptune is not None:
            # self.neptune["lr"].log(self.scheduler.get_last_lr())
            self.neptune["entropy coeff"].log(self.entropy_regularization)

    def compute(self, states, actions):
        logit = {
            "p": self.actor(states),
            "v": self.critic(states)
        }
        cur_values = logit["v"]
        x = self.actor.eval_action(logit["p"], actions.squeeze())
        cur_log_probs = x["log_prob"]
        entropy = x["entropy"]
        return cur_values.squeeze(), cur_log_probs.squeeze(), entropy

    def evaluate(self):
        score = 0
        state = self.env.reset()

        with torch.no_grad():
            with self._weights_lock:
                self.eval_mode()
                while True:
                    s = torch.Tensor(state).unsqueeze(0).to(self.device)
                    logit = {
                        "p": self.actor(s),
                        "v": self.critic(s)
                    }
                    action = self.actor.get_best_action(logit["p"]).detach().cpu().numpy()
                    observation, reward, done, _ = self.env.step(np.squeeze(action))
                    score += reward
                    state = observation
                    if done:
                        break
                self.train_mode()

        self.timepoints.append(time.time() - self.start_time)
        self.scores.append(score)
        self.scores_window.append(score)
        mean_score = np.mean(self.scores_window)
        self.means.append(mean_score)
        self.best_score = max(self.best_score, mean_score)
        print(f"Running mean: {mean_score}, >Best={self.best_score}")

        if self.neptune is not None:
            self.neptune["eval/score"].log(score)
            self.neptune["eval/running mean"].log(mean_score)

        if mean_score >= self.solve_score:
            print("SOLVED!")
            self.save_results()
            self.neptune.stop()
            sys.exit(0)  # <- Terminate this process

    def save_results(self):
        print("Saving results...")
        os.makedirs(self.save_dir, exist_ok=True)
        res = {
            "scores": self.scores,
            "means": self.means,
            "timepoints": self.timepoints,
        }
        filepath = os.path.join(
            self.save_dir,
            f"result-seed-{self.seed}-" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
        )
        with open(f"{filepath}.pickle", "wb") as f:
            pickle.dump(res, f)

        self.save_weights()

    def save_weights(self):
        with self._weights_lock:
            torch.save(
                self.actor.state_dict(),
                os.path.join(self.save_dir, "actor.pth"),
            )
            torch.save(
                self.critic.state_dict(),
                os.path.join(self.save_dir, "critic.pth"),
            )

    def get_weights(self):
        with self._weights_lock:
            return {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict()
            }

    def sync(self, weight):
        with self._weights_lock:
            self.actor.load_state_dict(weight["actor"])
            self.critic.load_state_dict(weight["critic"])
            self.synced = True
