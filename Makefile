.PHONY: run

run:
	python3 run_learner.py --config=configs/OpenAI/LunarLanderContinuous.yaml &
	python3 run_worker.py --config=configs/OpenAI/LunarLanderContinuous.yaml