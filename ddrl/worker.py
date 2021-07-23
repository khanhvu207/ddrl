import concurrent.futures
import os
import pickle
import socket
import sys
import threading
import time
from collections import deque
from datetime import datetime

import gym
import numpy as np

from .trainer import *
from .utils import set_seed

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Worker:
    def __init__(self, config, seed):
        # Set seed
        self.seed = int(seed)
        set_seed(seed=self.seed)

        # Configs
        self.config = config
        self.env_config = config["env"]
        self.worker_config = config["worker"]
        self.socket_config = config["socket"]

        self.env_name = self.env_config["env-name"]
        self.ip = self.socket_config["ip"]
        self.port = self.socket_config["port"]
        self.max_t = self.worker_config["max_t"]
        self.buffer_size = self.worker_config["buffer_size"]

        self.env = gym.make(self.env_name)
        self.env.seed(self.seed)
        self.obs_dim = self.env.observation_space.shape[0]
        try:
            self.act_dim = self.env.action_space.shape[0]
        except:
            self.act_dim = self.env.action_space.n

        # Agent
        self.agent = Trainer(
            state_size=self.obs_dim,
            action_size=self.act_dim,
            config=config,
            device=device,
            neptune=None,
        )

        # Socket configs
        self.msg_buffer_len = 65536
        self.msg_length_padding = 15

        # Threads handler
        self.executor = concurrent.futures.ThreadPoolExecutor()

        # Socket
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._connect_to_server()
        self.executor.submit(self._listen_to_server)

        # Threadlock
        self.lock = threading.Lock()

        # Monitoring
        self.eps_count = 0
        self.has_weight = False
        self.scores = []
        self.means = []
        self.scores_window = deque(maxlen=100)

        # Eps-greedy
        self.eps_greedy = 0.0
        self.eps_greedy_decay = 0.99

    def _connect_to_server(self):
        """
        Connect to the learner, retry if fail
        """
        while True:
            try:
                self.s.connect((self.ip, self.port))
                break
            except:
                pass

    def _listen_to_server(self):
        data = b""
        new_msg = True
        msg_len = 0
        while True:
            try:
                msg = self.s.recv(self.msg_buffer_len)
                if len(msg):
                    if new_msg:
                        msg_len = int(msg[: self.msg_length_padding])
                        msg = msg[self.msg_length_padding :]
                        new_msg = False
                    data += msg
                    if len(data) == msg_len:
                        state_dicts = pickle.loads(data)
                        self._sync(state_dicts)
                        data = b""
                        new_msg = True
            except:
                pass

    def _sync(self, network_weights):
        """
        Neural network weights' synchronization
        """
        with self.lock:
            self.agent.sync(
                weight=network_weights["net"],
            )
            print("Weights synced!")

            if not self.has_weight:
                self.has_weight = True

    def run(self):
        while True:
            if not self.agent.synced:
                continue
            trajectory = {
                "states": [],
                "actions": [],
                "log_probs": [],
                "rewards": [],
            }
            total_t = 0
            while total_t < self.buffer_size:
                score = 0
                state = self.env.reset()
                states = []
                actions = []
                log_probs = []
                eps_reward = []

                with self.lock:
                    self.agent.noise.reset()

                    for t in range(self.max_t):
                        # Exploration
                        # if np.random.rand() <= self.eps_greedy:
                        #     action = [np.random.choice(3)]
                        #     log_prob = [np.log(self.eps_greedy)] # prob = 1.0 -> ln(1.0) = 0
                        # else:
                        action, log_prob, _ = self.agent.act(state)
                        observation, reward, done, _ = self.env.step(
                            np.squeeze(action)
                        )
                        states.append(state)
                        actions.append(action)
                        log_probs.append(log_prob)
                        eps_reward.append(reward)
                        score += reward
                        total_t += 1
                        state = observation
                        if done:
                            break

                    self.eps_greedy = max(0.0, self.eps_greedy * self.eps_greedy_decay)

                    trajectory["states"].append(states)
                    trajectory["actions"].append(actions)
                    trajectory["log_probs"].append(log_probs)
                    trajectory["rewards"].append(eps_reward)
                    self.scores.append(score)
                    self.scores_window.append(score)
                    # self.neptune["score"].log(score)

                    mean_score = np.mean(self.scores_window)
                    self.means.append(mean_score)
                    # self.neptune["avg_score"].log(mean_score)
                    self.eps_count += 1

            # print(f"Average score: {self.means[-1]:.2f}")
            # print(f"Epsilon-greedy: {self.eps_greedy:.5f}")
            self._send_collected_experience(trajectory)
            del trajectory

    def _send_collected_experience(self, trajectory):
        data = pickle.dumps(trajectory)
        msg = bytes(f"{len(data):<{15}}", "utf-8") + data
        self.s.sendall(msg)
