import signal
import subprocess
import time
from re import sub

import fire


def main(env_name=None, runtime=1, num_workers=2, worker_seed=14482, debug=True):
    try:
        if runtime is None or env_name is None:
            raise Exception()

        print(f"Starting {env_name} with {num_workers} workers...")
        debug_mode = "True" if debug else "False"

        learner = subprocess.Popen(
            ["python3", "run_learner.py", f"--config=configs//OpenAI//{env_name}.yaml"]
        )

        workers = []
        for _ in range(num_workers):
            worker = subprocess.Popen(
                [
                    "python3",
                    "run_worker.py",
                    f"--config=configs//OpenAI//{env_name}.yaml",
                    f"--seed={worker_seed}",
                    f"--debug={debug_mode}",
                ]
            )
            workers.append(worker)
            debug_mode = True
            worker_seed += 1

        time.sleep(60 * runtime)  # <- 60 seconds x minutes

        learner.send_signal(signal.SIGKILL)
        for worker in workers:
            worker.send_signal(signal.SIGKILL)
        print("Done!")
    except KeyboardInterrupt:
        learner.send_signal(signal.SIGKILL)
        for worker in workers:
            worker.send_signal(signal.SIGKILL)
        print("Interrupted!")


if __name__ == "__main__":
    fire.Fire(main)
