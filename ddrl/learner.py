import bz2
import gym
import json
import time
import pickle
import socket
import threading
import concurrent.futures
from datetime import datetime

from .agent import Agent
from .buffer import Buffer
from .collector import Collector
from .synchronizer import Synchronizer

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Learner:
    def __init__(self, config):
        # Configs
        self.config = config
        self.env_name = config["env"]["env-name"]
        self.ip = config["learner"]["socket"]["ip"]
        self.port = config["learner"]["socket"]["port"]

        self.env = gym.make(self.env_name)
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.env.close()

        self.agent = Agent(
            state_size=self.obs_dim,
            action_size=self.act_dim,
            config=config,
            device=device,
        )

        self.eps_count = 0

        self.buffer = Buffer(config=config, agent=self.agent)
        self.synchronizer = Synchronizer(config=config, agent=self.agent)
        self.collector = Collector(
            config=config, buffer=self.buffer, synchronizer=self.synchronizer
        )

        self.executor = concurrent.futures.ThreadPoolExecutor()
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._bind_server()

    def _bind_server(self):
        self.server.bind(("", self.port))
        self.server.listen(5)
        print(f"Start a new learner on PORT {self.port}")
        f = self.executor.submit(self._server_listener)
        if self.config["debug"]:
            f.result()

    def _server_listener(self):
        while True:
            client, address = self.server.accept()
            print(f"Client {address[0]}:{address[1]} is connected...")

            # New worker connected
            self.synchronizer.update_weights()
            self.synchronizer.send_weights(client)
            self.collector.got_new_worker(client, address)

    def step(self):
        while True:
            time.sleep(0)
            if len(self.buffer) >= self.config["learner"]["network"]["batch_size"]:
                print(f"Step {self.eps_count}, learning...")
                self.agent.learn(self.buffer.sample())
                self.synchronizer.update_weights()
                self.eps_count += 1
