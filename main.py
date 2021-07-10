from re import sub
import fire
import subprocess, signal, time

def main(wallclock_time=None, debug=False):
    if wallclock_time is None:
        raise Exception()

    print("Starting...")
    debug_mode = 'True' if debug else 'False'
    learner = subprocess.Popen(['python3', 'run_learner.py', '--config=configs//OpenAI//LunarLanderContinuous.yaml'])
    worker = subprocess.Popen(['python3', 'run_worker.py', '--config=configs//OpenAI//LunarLanderContinuous.yaml', f'--debug={debug_mode}'])

    time.sleep(60 * wallclock_time) # <- 60 seconds x minutes

    learner.send_signal(signal.SIGKILL)
    worker.send_signal(signal.SIGKILL)
    print("Done!")

if __name__ == '__main__':
    fire.Fire(main)