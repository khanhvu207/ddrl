import fire
import yaml
import gym
import torch
import numpy as np
from collections import deque
from ddrl.agent import Agent

env_name = 'LunarLanderContinuous-v2'

def main(config=None):
    config = yaml.load(open(config, "r"), Loader=yaml.Loader)
    env = gym.make(config['env']['env-name'])
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = Agent(state_size=obs_dim, action_size=act_dim, config=config, device='cpu', neptune=None)

    agent.sync(actor_weight=torch.load('actor_weight.pth'), critic_weight=torch.load('critic_weight.pth'))

    observation = env.reset()
    rnn_seq_len = config["learner"]["network"]["rnn_seq_len"]
    prev_actions = deque([np.zeros(act_dim) for _ in range(rnn_seq_len)], maxlen=rnn_seq_len)
    
    while True:
        env.render()
        action, log_prob, value = agent.act(observation, prev_actions)
        observation, reward, done, info = env.step(action)
        prev_actions.append(action)

if __name__ == '__main__':
    fire.Fire(main)
