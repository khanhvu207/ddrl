from re import sub
import fire
import subprocess, signal, time

def main(runtime=1, num_workers=2, debug=True):
    try:
        if runtime is None:
            raise Exception()

        print("Starting...")
        debug_mode = 'True' if debug else 'False'
        learner = subprocess.Popen(['python3', 'run_learner.py', '--config=configs//OpenAI//LunarLanderContinuous.yaml'])

        workers = []
        for _ in range(num_workers):
            worker = subprocess.Popen(['python3', 'run_worker.py', '--config=configs//OpenAI//LunarLanderContinuous.yaml', f'--debug={debug_mode}'])
            workers.append(worker)
            debug_mode = True

        time.sleep(60 * runtime) # <- 60 seconds x minutes

        learner.send_signal(signal.SIGKILL)
        for worker in workers:
            worker.send_signal(signal.SIGKILL)
        print("Done!")
    except KeyboardInterrupt:
        learner.send_signal(signal.SIGKILL)
        for worker in workers:
            worker.send_signal(signal.SIGKILL)
        print("Interrupted!")

if __name__ == '__main__':
    fire.Fire(main)