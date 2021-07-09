import torch
import random
import threading
from collections import deque, namedtuple

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
            field_names=["state", "action", "prev_actions", "log_prob", "reward"],
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

            for j in range(len(states)):
                e = self.experience(
                    states[i],
                    actions[i],
                    prev_actions[i],
                    log_probs[i],
                    returns[i],
                )
                with self._buffer_lock:
                    self.memory.append(e)
        print("Buffer updated")

    def __len__(self):
        return len(self.memory)

    def sample(self):
        with self._buffer_lock:
            experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.Tensor([e.state for e in experiences if e is not None]).to(
            device
        )
        actions = torch.Tensor([e.action for e in experiences if e is not None]).to(
            device
        )
        prev_actions = torch.Tensor([e.prev_actions for e in experiences if e is not None]).to(
            device
        )
        log_probs = torch.Tensor([e.log_prob for e in experiences if e is not None]).to(
            device
        )
        rewards = torch.Tensor([e.reward for e in experiences if e is not None]).to(
            device
        )

        log_probs = torch.squeeze(log_probs)
        return states, actions, prev_actions, log_probs, rewards

    def _compute_returns(self, rewards):
        returns = []
        discounted_reward = 0
        for reward in reversed(rewards):
            discounted_reward = reward + discounted_reward * self.gamma
            returns.append(discounted_reward)
        return returns[::-1]
