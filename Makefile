.PHONY: run_lunar_lander run_bipedal_walker

DEBUG ?= True

run_lunar_lander: 
	python3 main.py --env_name=LunarLanderContinuous-v2 --runtime=60 --num_workers=3 --worker_seed=14482 --debug=${DEBUG}

run_bipedal_walker:
	python3 main.py --env_name=BipedalWalker-v3 --runtime=60 --num_workers=3 --worker_seed=2021 --debug=${DEBUG}