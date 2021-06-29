import yaml
import fire
from ddrl.learner import Learner

def main(config=None):
    config = yaml.load(open(config, "r"), Loader=yaml.Loader)
    learner = Learner(config)
    learner.step()

if __name__ == '__main__':
    fire.Fire(main)