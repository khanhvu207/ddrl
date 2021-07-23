.PHONY: run_cartpole run_acrobot

TIME ?= 60
DEBUG ?= True
NWORKERS ?= 3

OPENAI:= benchmarks/openai-gyms
CONFIGS:= configs

${OPENAI}/CartPole-v1/net.pth:
	python3 main.py --env_name=CartPole-v1 --runtime=${TIME} --num_workers=${NWORKERS} --worker_seed=2021 --debug=${DEBUG}

${OPENAI}/Acrobot-v1/net.pth:
	python3 main.py --env_name=Acrobot-v1 --runtime=${TIME} --num_workers=${NWORKERS} --worker_seed=2021 --debug=${DEBUG}

${OPENAI}/LunarLander-v2/net.pth:
	python3 main.py --env_name=LunarLander-v2 --runtime=${TIME} --num_workers=${NWORKERS} --worker_seed=2021 --debug=${DEBUG}

run_cartpole: ${OPENAI}/CartPole-v1/net.pth

run_acrobot: ${OPENAI}/Acrobot-v1/net.pth

run_lunar: ${OPENAI}/LunarLander-v2/net.pth

eval_cartpole: run_cartpole
	python3 eval.py --cp=${OPENAI}/CartPole-v1/net.pth --config=${CONFIGS}/OpenAI/CartPole-v1.yaml

eval_acrobot: run_acrobot
	python3 eval.py --cp=${OPENAI}/Acrobot-v1/net.pth --config=${CONFIGS}/OpenAI/Acrobot-v1.yaml