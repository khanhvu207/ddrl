from ddrl.agent import Agent

import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def run(env_name, max_eps=1000, max_t=512, batch_size=1024):
    scores = []
    means = []
    scores_window = deque(maxlen=100)
    env = gym.make(env_name)  
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = Agent(state_size=obs_dim, action_size=act_dim)
    
    for i in range(max_eps):
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
            state = env.reset()
            agent.noise_reset()
            for t in range(max_t):
                # env.render()
                action, log_prob, value = agent.act(state, is_train=False)
                observation, reward, done, info = env.step(action)
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
        print(f"\rEpisode: {i}, Average score: {mean_score:.2f}")
        if mean_score >= 200:
            print('Solved!')
            break
        agent.learn(trajectory)

    env.close()
    agent.save_weights()
    return means

if __name__ == '__main__':
    scores = run(env_name='LunarLanderContinuous-v2')

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('result.png')