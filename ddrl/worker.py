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
# device = 'cpu'

import neptune.new as neptune


class Worker:
    def __init__(self, config, seed, debug):
        # Set seed
        self.seed = int(seed)
        set_seed(seed=self.seed)

        self.config = config
        self.env_name = config["env"]["env-name"]
        self.ip = config["learner"]["socket"]["ip"]
        self.port = config["learner"]["socket"]["port"]
        self.max_t = config["worker"]["max_t"]
        self.batch_size = config["worker"]["batch_size"]
        self.rnn_seq_len = config["learner"]["network"]["rnn_seq_len"]
        self.save_dir = config["worker"]["result_dir"]
        self.save_res_every = config["worker"]["save_res_every"]
        self.solve_score = config["worker"]["solve_score"]

        # Neptune.ai
        self.neptune = neptune.init(
            project=config["neptune"]["project"],
            api_token=config["neptune"]["api_token"],
            mode="debug" if debug else "async",
        )

        self.neptune["environment"] = self.env_name
        self.neptune["batch size"] = config["learner"]["network"]["batch_size"]
        self.neptune["learning steps"] = config["learner"]["network"]["learning_steps"]
        self.neptune["loss target"] = config["learner"]["network"]["loss_target"]
        self.neptune["lambda"] = config["learner"]["network"]["lambda"]
        self.neptune["gamma"] = config["learner"]["batcher"]["gamma"]
        self.neptune["policy clip"] = config["learner"]["network"]["clip"]
        self.neptune["seed"] = self.seed

        self.env = gym.make(self.env_name)
        self.env.seed(self.seed)
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.agent = Trainer(
            state_size=self.obs_dim,
            action_size=self.act_dim,
            config=config,
            device=device,
            neptune=None,
        )
        self.msg_buffer_len = 65536
        self.msg_length_padding = 15

        self.executor = concurrent.futures.ThreadPoolExecutor()
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._connect_to_server()
        self.executor.submit(self._listen_to_server)

        self.lock = threading.Lock()

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
        with self.lock:
            self.agent.sync(
                actor_weight=network_weights["actor"],
                critic_weight=network_weights["critic"],
            )
            print("Weights synced!")

            if not self.has_weight:
                self.has_weight = True
                # self.executor.submit(self.run)

    def run(self):
        while True:
            if not self.agent.synced:
                continue
            trajectory = {
                "states": [],
                "actions": [],
                "prev_actions": [],
                "log_probs": [],
                "rewards": [],
            }
            total_t = 0
            while total_t < self.batch_size:
                score = 0
                state = self.env.reset()
                states = []
                actions = []
                prev_acts = []
                log_probs = []
                eps_reward = []

                prev_actions = deque(
                    [np.zeros(self.act_dim) for _ in range(self.rnn_seq_len)],
                    maxlen=self.rnn_seq_len,
                )

                with self.lock:
                    for t in range(self.max_t):
                        action, log_prob, value = self.agent.act(
                            state, prev_actions=prev_actions
                        )
                        observation, reward, done, info = self.env.step(action)
                        prev_actions.append(action)
                        states.append(state)
                        actions.append(action)
                        prev_acts.append(prev_actions)
                        log_probs.append(log_prob)
                        eps_reward.append(reward)
                        score += reward
                        total_t += 1
                        state = observation
                        if done:
                            break

                    trajectory["states"].append(states)
                    trajectory["actions"].append(actions)
                    trajectory["prev_actions"].append(prev_acts)
                    trajectory["log_probs"].append(log_probs)
                    trajectory["rewards"].append(eps_reward)
                    self.scores.append(score)
                    self.scores_window.append(score)
                    self.neptune["score"].log(score)

                    mean_score = np.mean(self.scores_window)
                    self.means.append(mean_score)
                    self.neptune["avg_score"].log(mean_score)
                    self.eps_count += 1

            print(f"Average score: {self.means[-1]:.2f}")
            self._send_collected_experience(trajectory)
            if self.means[-1] > self.solve_score:
                self.save_results()
                self.agent.save_weights()
                sys.exit(0)

            del trajectory

    def save_results(self):
        res = {
            "scores": self.scores,
            "scores_window": self.scores_window,
            "means": self.means,
        }
        filepath = os.path.join(
            self.save_dir,
            f"result-{self.seed}-" + datetime.now().strftime("%Y_%m_%d"),
        )

        print("Saving results...")
        with open(f"{filepath}.pickle", "wb") as f:
            pickle.dump(res, f)

    def _send_collected_experience(self, trajectory):
        data = pickle.dumps(trajectory)
        msg = bytes(f"{len(data):<{15}}", "utf-8") + data
        self.s.sendall(msg)
