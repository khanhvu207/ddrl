import yaml
import fire
from ddrl.worker import Worker


def main(config=None, seed=None):
    config = yaml.load(open(config, "r"), Loader=yaml.Loader)
    worker = Worker(config, seed)
    worker.run()
    worker.save_results()


if __name__ == "__main__":
    fire.Fire(main)
