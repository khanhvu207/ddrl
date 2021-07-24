import os
import bz2
import gym
import json
import time
import pickle
import socket
import threading
import concurrent.futures
from datetime import datetime

from .trainer import Trainer
from .buffer import Buffer
from .collector import Collector
from .synchronizer import Synchronizer
from .utils import set_seed

import torch
import neptune.new as neptune

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Learner:
    def __init__(self, config, debug):
        # Set seed
        self.seed = config["learner"]["seed"]
        set_seed(seed=self.seed)

        # Configs
        self.config = config
        self.env_config = config["env"]
        self.socket_config = config["socket"]
        self.learner_config = config["learner"]

        self.env_name = self.env_config["env-name"]
        self.ip = self.socket_config["ip"]
        self.port = self.socket_config["port"]

        # Get environment's state and action space size
        self.env = gym.make(self.env_name)
        self.obs_dim = self.env.observation_space.shape[0]
        try:
            self.act_dim = self.env.action_space.shape[0]
        except:
            self.act_dim = self.env.action_space.n
        self.env.close()

        # Neptune.ai
        self.neptune = neptune.init(
            project=os.environ["NEPTUNE_PROJECT"],
            api_token=os.environ["NEPTUNE_API_TOKEN"],
            mode="debug" if debug else "async",
        )

        self.neptune["environment"] = self.env_name
        self.neptune["seed"] = self.seed

        # Trainer
        self.agent = Trainer(
            state_size=self.obs_dim,
            action_size=self.act_dim,
            config=config,
            device=device,
            neptune=self.neptune,
        )

        self.eps_count = 0

        # Dependencies
        self.buffer = Buffer(config=config, agent=self.agent)
        self.synchronizer = Synchronizer(config=config, agent=self.agent)
        self.collector = Collector(
            config=config, buffer=self.buffer, synchronizer=self.synchronizer
        )

        # Threads handler
        self.executor = concurrent.futures.ThreadPoolExecutor()

        # Socket endpoint
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._bind_server()

    def _bind_server(self):
        self.server.bind(("", self.port))
        self.server.listen(5)
        print(f"Start a new learner on PORT {self.port}")
        f = self.executor.submit(self._server_listener)

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
            time.sleep(
                0.15  # <- Adjust this accordingly to the number of parallel workers
            )
            if len(self.buffer) > 0: #and self.buffer.update_count == 3:
                print(f"Step {self.eps_count}, learning...")
                self.agent.learn(self.buffer.sample())
                self.synchronizer.update_weights()
                self.eps_count += 1
                self.buffer.update_count = 0

                # Evaluate with the network every K steps
                if self.eps_count % self.config["learner"]["eval_every"] == 0:
                    self.agent.evaluate()
