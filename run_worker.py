from ddrl.worker import Worker

if __name__ == '__main__':
    worker = Worker(env_name='LunarLanderContinuous-v2')
    worker.evaluate()