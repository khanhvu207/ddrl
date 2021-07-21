.PHONY: run_lunar_lander run_bipedal_walker

TIME ?= 60
DEBUG ?= True
NWORKERS ?= 3

run_lunar_lander: 
	python3 main.py --env_name=LunarLanderContinuous-v2 --runtime=${TIME} --num_workers=${NWORKERS} --worker_seed=2021 --debug=${DEBUG}