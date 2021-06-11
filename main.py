from ddrl.agent import Agent

import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def run(env_name, max_eps=500, max_t=512):
    scores = []
    means = []
    scores_window = deque(maxlen=100)

    env = gym.make(env_name)  
    agent = Agent(state_size=8, action_size=2)
    
    for i in range(max_eps):
        trajectory = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'values': [],
            'rewards': [],
            'dones': [],
        }
        state = env.reset()
        score = 0
        for t in range(max_t):
            # env.render()
            action, log_prob, value = agent.act(state)
            observation, reward, done, info = env.step(action)
            trajectory['states'].append(state)
            trajectory['actions'].append(action)
            trajectory['log_probs'].append(log_prob)
            trajectory['values'].append(value)
            trajectory['rewards'].append(reward)
            trajectory['dones'].append(done)
            score += reward
            state = observation
            if done:
                break
        
        # Monitor
        scores.append(score)
        scores_window.append(score)
        mean_score = np.mean(scores_window)
        means.append(mean_score)
        print(f"\rEpisode: {i}, Average score: {mean_score:.2f}")

        if mean_score >= 200:
            print('Solved!')
            break

        # Learn
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