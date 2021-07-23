import time
import torch
import random
import threading
import itertools
import numpy as np
from collections import deque, namedtuple

from .losses import compute_target

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Buffer:
    def __init__(self, config, agent):
        # Configs
        self.config = config
        self.batcher_config = config["batcher"]
        self.learner_config = config["learner"]

        self.buffer_size = self.batcher_config["buffer_size"]
        self.gamma = self.learner_config["gamma"]
        self.batch_size = self.batcher_config["batch_size"]
        self.agent = agent

        # Buffer
        self.memory = deque(maxlen=self.buffer_size)
        self.experience = namedtuple(
            "Experience",
            field_names=["states", "actions", "log_probs", "vs", "advantages"],
        )
        
        # Threadlock
        self._buffer_lock = threading.Lock()
        self.update_count = 0

    def add(self, trajectory):

        for i in range(len(trajectory["states"])):
            states = trajectory["states"][i]
            actions = trajectory["actions"][i]
            log_probs = trajectory["log_probs"][i]
            rewards = trajectory["rewards"][i]

            # SANITY CHECK! Should you clip the rewards? It depends...
            # rewards = np.clip(rewards, -1, 1) 

            # Or normalize rewards? And then clip [-1, 1]
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9)
            rewards = np.clip(rewards, -1, 1)

            returns = self._compute_returns(rewards)

            with self._buffer_lock:
                self.memory.append((states, actions, log_probs, returns, rewards))
        
        self.update_count += 1

    def __len__(self):
        return len(self.memory)

    def sample(self):
        total_sample = 0
        t_states = []
        t_actions = []
        t_log_probs = []
        t_vs = []
        t_advantages = []
        # Stack the experiences until it exceeds the batch size
        while total_sample < self.batch_size:
            with self._buffer_lock:
                # Pick a random trajectory from the buffer
                experiences = random.sample(population=self.memory, k=1)

            states, actions, log_probs, returns, rewards = experiences[0]

            total_sample += len(states)

            with self.agent._weights_lock:
                with torch.no_grad():
                    self.agent.eval_mode()
                    cur_values, cur_log_probs, _ = self.agent.compute(
                        states=torch.Tensor(states).to(device),
                        actions=torch.Tensor(actions).to(device),
                    )
                    self.agent.train_mode()

            cur_values = cur_values.cpu().detach().numpy()
            cur_log_probs = cur_log_probs.cpu().detach().numpy()

            unclipped_rhos = np.exp(cur_log_probs - np.squeeze(np.array(log_probs)))
            rhos = np.clip(unclipped_rhos, 0.0, 1.0)
            cs = np.clip(unclipped_rhos, 0.0, 1.0)


            vs, advantages = compute_target(
                loss=self.learner_config["loss_target"],
                values=cur_values,
                returns=returns,
                rewards=rewards,
                gamma=self.gamma,
                rhos=rhos,
                cs=cs,
                lmbd=self.learner_config["lambda"],
            )

            states = torch.Tensor(states).to(device)
            actions = torch.Tensor(actions).to(device)
            log_probs = torch.Tensor(log_probs).to(device)
            vs = torch.Tensor(vs).to(device)
            advantages = torch.Tensor(advantages).to(device)
            log_probs = torch.squeeze(log_probs)
            t_states.append(states)
            t_actions.append(actions)
            t_log_probs.append(log_probs)
            t_vs.append(vs)
            t_advantages.append(advantages)

        t_states = torch.cat([x for x in t_states])
        t_actions = torch.cat([x for x in t_actions])
        t_log_probs = torch.cat([x for x in t_log_probs])
        t_vs = torch.cat([x for x in t_vs])
        t_advantages = torch.cat([x for x in t_advantages])

        return t_states, t_actions, t_log_probs, t_vs, t_advantages

    def _compute_returns(self, rewards):
        returns = []
        discounted_reward = 0
        for reward in reversed(rewards):
            discounted_reward = reward + discounted_reward * self.gamma
            returns.append(discounted_reward)
        return returns[::-1]
