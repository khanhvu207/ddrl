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
from .batcher import Batcher

class Learner:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.env.close()

        self.agent = Agent(state_size=self.obs_dim, action_size=self.act_dim) 
        self.batcher = Batcher()

        self.executor = concurrent.futures.ThreadPoolExecutor()
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._bind_server()

        # Threadlocks
        self._batcher_lock = threading.Lock()
        self._networks_lock = threading.Lock()

    def _bind_server(self, ip='127.0.0.1', port=23333):
        self.server.bind((ip, port))
        self.server.listen(5)
        print(f'Start a new learner on PORT {port}')
        self.executor.submit(self._server_listener)

    def _get_weights(self):
        with self._networks_lock:
            actor_weight, critic_weight = self.agent.get_weights()
            weight_dict = {
                'actor': actor_weight,
                'critic': critic_weight
            }
        return weight_dict

    def send_weights(self, client):
        data_string = pickle.dumps(self._get_weights())
        msg = bytes(f"{len(data_string):<{15}}", 'utf-8') + data_string
        client.sendall(msg)
        print('Weights sent!')
    
    def _server_listener(self):
        while True:
            client, address = self.server.accept()
            print(f'Client {address[0]}:{address[1]} is connecting...')
            self.send_weights(client)
            self.executor.submit(self._worker_handler, client, address)
            self.executor.submit(self._periodic_synchronizer, client)
    
    def _periodic_synchronizer(self, client):
        while True:
            time.sleep(30)
            self.send_weights(client)

    def _worker_handler(self, client, address):
        client_ip, client_port = address
        new_msg = True
        data = b''
        while True:
            msg = client.recv(4096)
            if len(msg):
                if new_msg:
                    msg_len = int(msg[:15])
                    # print(f'Message length: {msg_len}')
                    msg = msg[15:]
                    new_msg = False  
                
                data += msg
                
                if len(data) == msg_len:
                    # print('Full message received')
                    batch = pickle.loads(data)
                    with self._batcher_lock:
                        self.batcher.add(batch)                    
                    new_msg = True
                    data = b''

        print(f'{client_ip}:{client_port} disconnected!')
        client.close()
    
    def step(self):
        while True:
            time.sleep(0.1)
            with self._batcher_lock:
                with self._networks_lock:
                    if len(self.batcher) >= 1024:
                        print("Learning...")
                        self.agent.learn(self.batcher.sample())
