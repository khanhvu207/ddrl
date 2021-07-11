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
        self.config = config
        self.buffer_size = config["learner"]["batcher"]["buffer_size"]
        self.gamma = config["learner"]["batcher"]["gamma"]
        self.batch_size = config["learner"]["network"]["batch_size"]
        self.agent = agent

        self.memory = deque(maxlen=self.buffer_size)
        self.experience = namedtuple(
            "Experience",
            field_names=["states", "actions", "prev_actions", "log_probs", "vs", "advantages"],
        )

        self._buffer_lock = threading.Lock()

    def add(self, trajectory):
        
        for i in range(len(trajectory["states"])):
            states = trajectory["states"][i]
            actions = trajectory["actions"][i]
            prev_actions = trajectory["prev_actions"][i]
            log_probs = trajectory["log_probs"][i]
            rewards = trajectory["rewards"][i]
            returns = self._compute_returns(rewards)
            
            with self._buffer_lock:
                self.memory.append((states, actions, prev_actions, log_probs, returns, rewards))

    def __len__(self):
        return len(self.memory)

    def sample(self):
        total_sample = 0
        t_states = []
        t_actions = []
        t_prev_actions = []
        t_log_probs = []
        t_vs = []
        t_advantages = []
        while total_sample < self.batch_size:
            if len(self.memory) == 0:
                break
            with self._buffer_lock:
                experiences = random.sample(population=self.memory, k=1)    

            states, actions, prev_actions, log_probs, returns, rewards = experiences[0]
            # rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-10)
            
            total_sample += len(states)

            with torch.no_grad():
                with self.agent._weights_lock:
                    cur_values, cur_log_probs, _ = self.agent.compute(
                        states=torch.Tensor(states).to(device),
                        actions=torch.Tensor(actions).to(device),
                        prev_actions=torch.Tensor(prev_actions).to(device),
                    )

            cur_values = cur_values.cpu().detach().numpy()
            cur_log_probs = cur_log_probs.cpu().detach().numpy()
            unclipped_rhos = np.exp(cur_log_probs - np.squeeze(np.array(log_probs)))
            rhos = np.clip(unclipped_rhos, 0.0, 1.0)
            cs = np.clip(unclipped_rhos, 0.0, 1.0)
            
            vs, advantages = compute_target(
                loss=self.config["learner"]["network"]["loss_target"],
                values=cur_values,
                returns=returns,
                rewards=rewards,
                gamma=self.gamma,
                rhos=rhos,
                cs=cs,
                lmbd=self.config["learner"]["network"]["lambda"]
            )

            states = torch.Tensor(states).to(device)
            actions = torch.Tensor(actions).to(device)
            prev_actions = torch.Tensor(prev_actions).to(device)
            log_probs = torch.Tensor(log_probs).to(device)
            vs = torch.Tensor(vs).to(device)
            advantages = torch.Tensor(advantages).to(device)
            log_probs = torch.squeeze(log_probs)
            t_states.append(states)
            t_actions.append(actions)
            t_prev_actions.append(prev_actions)
            t_log_probs.append(log_probs)
            t_vs.append(vs)
            t_advantages.append(advantages)

        t_states = torch.cat([x for x in t_states])
        t_actions = torch.cat([x for x in t_actions])
        t_prev_actions = torch.cat([x for x in t_prev_actions])
        t_log_probs = torch.cat([x for x in t_log_probs])
        t_vs = torch.cat([x for x in t_vs])
        t_advantages = torch.cat([x for x in t_advantages])

        return t_states, t_actions, t_prev_actions, t_log_probs, t_vs, t_advantages

    def _compute_returns(self, rewards):
        returns = []
        discounted_reward = 0
        for reward in reversed(rewards):
            discounted_reward = reward + discounted_reward * self.gamma
            returns.append(discounted_reward)
        
        # returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-10)
        return returns[::-1]
