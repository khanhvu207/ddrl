.PHONY: run_cartpole run_acrobot

TIME ?= 60
DEBUG ?= True
NWORKERS ?= 3

OPENAI:= benchmarks/openai-gyms
CONFIGS:= configs

# Training
run_cartpole:
	python3 main.py --env_name=CartPole-v1 --runtime=${TIME} --num_workers=${NWORKERS} --worker_seed=2021 --debug=${DEBUG}

run_acrobot:
	python3 main.py --env_name=Acrobot-v1 --runtime=${TIME} --num_workers=${NWORKERS} --worker_seed=2021 --debug=${DEBUG}

run_mountain_car:
	python3 main.py --env_name=MountainCar-v0 --runtime=${TIME} --num_workers=${NWORKERS} --worker_seed=2021 --debug=${DEBUG}

run_lunar:
	python3 main.py --env_name=LunarLander-v2 --runtime=${TIME} --num_workers=${NWORKERS} --worker_seed=2021 --debug=${DEBUG}

# Evaluation
eval_lunar:
	python3 eval.py --cp=${OPENAI}/LunarLander-v2/ --config=${CONFIGS}/OpenAI/LunarLander-v2.yaml

eval_cartpole:
	python3 eval.py --cp=${OPENAI}/CartPole-v1/ --config=${CONFIGS}/OpenAI/CartPole-v1.yaml

eval_acrobot:
	python3 eval.py --cp=${OPENAI}/Acrobot-v1/ --config=${CONFIGS}/OpenAI/Acrobot-v1.yaml

eval_mountain_car:
	python3 eval.py --cp=${OPENAI}/MountainCar-v0/ --config=${CONFIGS}/OpenAI/MountainCar-v0.yaml