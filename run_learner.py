from ddrl.learner import Learner

if __name__ == '__main__':
    learner = Learner(env_name='LunarLanderContinuous-v2')
    learner.step()