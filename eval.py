import gym
import torch
from ddrl.agent import Agent

env_name = 'LunarLanderContinuous-v2'

if __name__ == '__main__':
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = Agent(state_size=obs_dim, action_size=act_dim)

    agent.Actor.load_state_dict(torch.load('actor_weight.pth'))
    agent.Critic.load_state_dict(torch.load('critic_weight.pth'))

    observation = env.reset()
    while True:
        env.render()
        action, log_prob, value = agent.act(observation, is_train=False)
        observation, reward, done, info = env.step(action)
