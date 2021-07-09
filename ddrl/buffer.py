import time
import torch
import random
import threading
import numpy as np
from collections import deque, namedtuple

from .losses import vtrace

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Buffer:
    def __init__(self, config, agent):
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
            returns = self._compute_returns(trajectory["rewards"][i])
            rewards = trajectory["rewards"][i]

            with torch.no_grad():
                with self.agent._weights_lock:
                    cur_values, cur_log_probs = self.agent.evaluate(
                        states=torch.Tensor(states).to(device),
                        actions=torch.Tensor(actions).to(device),
                        prev_actions=torch.Tensor(prev_actions).to(device),
                    )

            cur_values = cur_values.cpu().detach().numpy()
            cur_log_probs = cur_log_probs.cpu().detach().numpy()
            unclipped_rhos = np.exp(cur_log_probs - np.squeeze(np.array(log_probs)))
            rhos = np.clip(unclipped_rhos, 0.0, 1.0)
            cs = np.clip(unclipped_rhos, 0.0, 1.0)
            
            vs, advantages = vtrace(
                values=cur_values,
                returns=returns,
                rewards=rewards,
                gamma=self.gamma,
                rhos=rhos,
                cs=cs,
            )
            
        
            for j in range(len(states)):
                e = self.experience(
                    states[i],
                    actions[i],
                    prev_actions[i],
                    log_probs[i],
                    vs[i],
                    advantages[i]
                )
                with self._buffer_lock:
                    self.memory.append(e)

    def __len__(self):
        return len(self.memory)

    def sample(self):
        with self._buffer_lock:
            experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.Tensor([e.states for e in experiences if e is not None]).to(
            device
        )
        actions = torch.Tensor([e.actions for e in experiences if e is not None]).to(
            device
        )
        prev_actions = torch.Tensor(
            [e.prev_actions for e in experiences if e is not None]
        ).to(device)
        log_probs = torch.Tensor([e.log_probs for e in experiences if e is not None]).to(
            device
        )
        vs = torch.Tensor([e.vs for e in experiences if e is not None]).to(
            device
        )
        advantages = torch.Tensor([e.advantages for e in experiences if e is not None]).to(
            device
        )

        log_probs = torch.squeeze(log_probs)
        return states, actions, prev_actions, log_probs, vs, advantages

    def _compute_returns(self, rewards):
        returns = []
        discounted_reward = 0
        for reward in reversed(rewards):
            discounted_reward = reward + discounted_reward * self.gamma
            returns.append(discounted_reward)
        return returns[::-1]
