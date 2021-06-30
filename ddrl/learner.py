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
            state_size=self.obs_dim, action_size=self.act_dim, config=config
        )

        self.buffer = Buffer(config=config)
        self.collector = Collector(config=config, buffer=self.buffer)

        self.executor = concurrent.futures.ThreadPoolExecutor()
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._bind_server()

    def _bind_server(self):
        self.server.bind(("", self.port))
        self.server.listen(5)
        print(f"Start a new learner on PORT {self.port}")
        self.executor.submit(self._server_listener)

    def _get_weights(self):
        actor_weight, critic_weight = self.agent.get_weights()
        weight_dict = {"actor": actor_weight, "critic": critic_weight}
        return weight_dict

    def send_weights(self, client):
        data_string = pickle.dumps(self._get_weights())
        msg = bytes(f"{len(data_string):<{15}}", "utf-8") + data_string
        client.sendall(msg)
        print('Weights sent!')

    def _server_listener(self):
        while True:
            client, address = self.server.accept()
            print(f"Client {address[0]}:{address[1]} is connected...")
            self.send_weights(client)
            self.collector.got_new_worker(client, address)

    def step(self):
        while True:
            time.sleep(0)
            if len(self.buffer) >= self.config["learner"]["network"]["batch_size"]:
                print("Learning...")
                self.agent.learn(self.buffer.sample())

