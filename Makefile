.PHONY: run_cartpole

TIME ?= 60
DEBUG ?= True
NWORKERS ?= 3

OPENAI:= benchmarks/openai-gyms

${OPENAI}/CartPole-v1/net.pth:
	python3 main.py --env_name=CartPole-v1 --runtime=${TIME} --num_workers=${NWORKERS} --worker_seed=2021 --debug=${DEBUG}

run_cartpole: ${OPENAI}/CartPole-v1/net.pth

eval_cartpole: run_cartpole
	python3 eval.py --cp=/home/kvu/ddrl/benchmarks/openai-gyms/CartPole-v1/net.pth --config=/home/kvu/ddrl/configs/OpenAI/CartPole-v1.yaml