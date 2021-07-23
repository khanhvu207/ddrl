import fire
import yaml
import gym
import torch
import numpy as np
from collections import deque
from ddrl.trainer import Trainer
import time

device = "cpu"

def main(cp=None, config=None):
    print(cp, config)
    config = yaml.load(open(config, "r"), Loader=yaml.Loader)
    env = gym.make(config["env"]["env-name"])
    env.seed(2021)

    obs_dim = env.observation_space.shape[0]
    try:
        act_dim = env.action_space.shape[0]
    except:
        act_dim = env.action_space.n

    agent = Trainer(
        state_size=obs_dim,
        action_size=act_dim,
        config=config,
        device=device,
        neptune=None,
    )

    agent.sync(weight=torch.load(cp))

    score = 0
    observation = env.reset()
    while True:
        env.render()
        time.sleep(0.1)
        logit = agent.net(torch.Tensor(observation).unsqueeze(0))
        action = agent.net.get_best_action(logit["p"]).numpy()
        observation, reward, done, _ = env.step(np.squeeze(action))
        score += reward
        if done:
            break
    
    print(score)


if __name__ == "__main__":
    fire.Fire(main)
