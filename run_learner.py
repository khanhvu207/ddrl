import yaml
import fire
from ddrl.learner import Learner
import time


def main(config=None, debug=True):
    config = yaml.load(open(config, "r"), Loader=yaml.Loader)
    learner = Learner(config, debug)
    learner.step()


if __name__ == "__main__":
    fire.Fire(main)
