import yaml
import fire
from ddrl.worker import Worker

def main(config=None):
    config = yaml.load(open(config, "r"), Loader=yaml.Loader)
    worker = Worker(config)
    worker.evaluate()

if __name__ == '__main__':
    fire.Fire(main)