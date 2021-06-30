import gym
import time
import pickle
import socket
import threading
import numpy as np
import concurrent.futures
from collections import deque

from .agent import *


class Worker:
    def __init__(self, config):
        self.config = config
        self.env_name = config["env"]["env-name"]
        self.ip = config["learner"]["socket"]["ip"]
        self.port = config["learner"]["socket"]["port"]
        self.max_t = config["worker"]["max_t"]
        self.batch_size = config["worker"]["batch_size"]
        self.gru_seq_len = config["learner"]["network"]["gru_seq_len"]

        self.env = gym.make(self.env_name)
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.agent = Agent(
            state_size=self.obs_dim, action_size=self.act_dim, config=config
        )
        self.msg_buffer_len = 65536
        self.msg_length_padding = 15

        self.executor = concurrent.futures.ThreadPoolExecutor()
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._connect_to_server()
        self.executor.submit(self._listen_to_server)

        # Monitoring
        self.eps_count = 0
        self.has_weight = False
        self.scores = []
        self.means = []
        self.scores_window = deque(maxlen=100)

    def _connect_to_server(self):
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
        self.agent.sync(
            actor_weight=network_weights["actor"],
            critic_weight=network_weights["critic"],
        )
        print("Weights synced!")

        if not self.has_weight:
            self.has_weight = True
            self.executor.submit(self.run)

    def run(self):
        while True:
            if not self.agent.synced:
                continue
            time.sleep(0)
            trajectory = {
                "states": [],
                "actions": [],
                "prev_actions": [],
                "log_probs": [],
                "rewards": [],
                "dones": [],
            }
            total_t = 0
            while total_t < self.batch_size:
                eps_reward = []
                score = 0
                state = self.env.reset()
                prev_actions = deque(
                    [np.zeros(self.act_dim) for _ in range(self.gru_seq_len)],
                    maxlen=self.gru_seq_len,
                )

                for t in range(self.max_t):
                    action, log_prob, value = self.agent.act(
                        state, prev_actions=prev_actions
                    )
                    observation, reward, done, info = self.env.step(action)
                    prev_actions.append(action)

                    trajectory["states"].append(state)
                    trajectory["actions"].append(action)
                    trajectory["prev_actions"].append(prev_actions)
                    trajectory["log_probs"].append(log_prob)
                    trajectory["dones"].append(done)
                    eps_reward.append(reward)
                    score += reward
                    total_t += 1
                    state = observation
                    if done:
                        break

                trajectory["rewards"].append(eps_reward)
                self.scores.append(score)
                self.scores_window.append(score)

                mean_score = np.mean(self.scores_window)
                self.means.append(mean_score)
                print(
                    f"Average score: {mean_score:.2f}, Buffer: {len(trajectory['states'])}/{self.batch_size}"
                )
                self.eps_count += 1

                if self.eps_count % self.config["worker"]["save_every"] == 0:
                    self.agent.save_weights()
                    print("Save weights")

            self._send_collected_experience(trajectory)

    def _send_collected_experience(self, trajectory):
        data = pickle.dumps(trajectory)
        msg = bytes(f"{len(data):<{15}}", "utf-8") + data
        self.s.sendall(msg)
