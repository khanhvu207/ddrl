import gym
import time
import pickle
import socket
import threading
import concurrent.futures 
from collections import deque

from .agent import *

class Worker:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.agent = Agent(state_size=self.obs_dim, action_size=self.act_dim)

        self.executor = concurrent.futures.ThreadPoolExecutor()
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._connect_to_server()
        self.executor.submit(self._listen_to_server)
        self.has_weight = False
        # self.executor.submit(self._send_msg_to_server)

        # Threadlocks
        self._networks_lock = threading.Lock()

    def _connect_to_server(self, ip='127.0.0.1', port=23333):
        self.s.connect((ip, port))
    
    def _listen_to_server(self):
        data = b''
        new_msg = True
        msg_len = 0
        while True:
            try:
                msg = self.s.recv(4096)
                if len(msg):
                    if new_msg:
                        msg_len = int(msg[:15])
                        print(f'Message length: {msg_len}')
                        msg = msg[15:]
                        new_msg = False  
                    data += msg
                    if len(data) == msg_len:
                        print('Full message received')
                        state_dicts = pickle.loads(data)
                        with self._networks_lock:
                            self._sync(state_dicts)
                            data = b''
                            new_msg = True
            except:
                pass
        
    def _sync(self, network_weights):
        self.agent.sync(actor_weight=network_weights['actor'], critic_weight=network_weights['critic'])
        self.has_weight = True
        print('Weights synced!')
    
    def evaluate(self, max_t=256, batch_size=1024):
        while True:
            if not self.has_weight:  
                continue
            time.sleep(1)
            with self._networks_lock:
                print("Evaluating...")  
                scores = []
                means = []
                scores_window = deque(maxlen=100)
                trajectory = {
                    'states': [],
                    'actions': [],
                    'log_probs': [],
                    'values': [],
                    'rewards': [],
                    'dones': [],
                }
                total_t = 0
                while total_t < batch_size:
                    eps_reward = []
                    score = 0
                    state = self.env.reset()
                    self.agent.noise_reset()
                    for t in range(max_t):
                        # env.render()
                        action, log_prob, value = self.agent.act(state, is_train=False)
                        observation, reward, done, info = self.env.step(action)
                        trajectory['states'].append(state)
                        trajectory['actions'].append(action)
                        trajectory['log_probs'].append(log_prob)
                        trajectory['values'].append(value)
                        eps_reward.append(reward)
                        trajectory['dones'].append(done)
                        score += reward
                        total_t += 1
                        state = observation
                        if done:
                            break
                    trajectory['rewards'].append(eps_reward)
                    scores.append(score)
                    scores_window.append(score)
                mean_score = np.mean(scores_window)
                means.append(mean_score)
                print(f"Average score: {mean_score:.2f}")
                self._send_collected_experience(trajectory)
    
    def _send_collected_experience(self, trajectory):
        data = pickle.dumps(trajectory)
        msg = bytes(f"{len(data):<{15}}", 'utf-8') + data
        self.s.sendall(msg)

                
            
    