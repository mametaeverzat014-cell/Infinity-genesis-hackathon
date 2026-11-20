PY ?= python3

.PHONY: check test verify splits trajectories demo docker docker-check

check:            ## the single official check
	$(PY) -m auditgym check

test:
	$(PY) -m unittest discover -s tests -t . -v

verify:
	$(PY) -m auditgym verify

splits:           ## regenerate the 10 train + 10 dev open episodes
	$(PY) -m auditgym build-splits

trajectories:     ## train-mode trajectories of all baselines on the train split
	$(PY) -m auditgym run --split train --mode train --policies random,reconciler,flag_all,oracle,expert --out runs/train_split
	mkdir -p trajectories && cp runs/train_split/trajectories.jsonl runs/train_split/results.jsonl runs/train_split/summary.json runs/train_split/report.md trajectories/

demo:
	$(PY) -m auditgym demo --policy expert --seed 1009 --config interaction

docker:
	docker build -t auditgym:1.0.0 .

docker-check: docker
	docker run --rm --network none --cpus 2 --memory 4g auditgym:1.0.0
