import yaml
import fire
from ddrl.worker import Worker


def main(config=None, debug=False):
    config = yaml.load(open(config, "r"), Loader=yaml.Loader)
    worker = Worker(config, debug=debug)
    worker.run()
    worker.save_results()


if __name__ == "__main__":
    fire.Fire(main)
