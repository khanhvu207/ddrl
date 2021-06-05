import gym
env = gym.make('BipedalWalkerHardcore-v3')
env.reset()

for t in range(100000):
    env.render()
    action = env.action_space.sample() # take a random action
    observation, reward, done, info = env.step(action)
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break

env.close()
