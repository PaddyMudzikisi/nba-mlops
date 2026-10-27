IMAGE    := nba-mlops:0.3.0
IMAGE_ID  = $(shell docker image inspect -f '{{.Id}}' $(IMAGE) 2>/dev/null)
JOB       = IMAGE_ID=$(IMAGE_ID) docker compose run --rm app

.PHONY: build up down data eda portal explore train reproduce evaluate monitor report serve test

build:      ## build the pinned environment image
	docker compose build

up:         ## start the MLflow server on http://localhost:5001
	docker compose up -d mlflow

down:
	docker compose --profile portal --profile serve down

data:       ## version data/raw with DVC (after adding or changing the dataset)
	$(JOB) dvc add data/raw
	$(JOB) dvc push

eda:        ## Chapter 3 data understanding: figures and tests -> reports/eda/
	$(JOB) python -m nba.eda

portal:     ## portal (upload, explore, train, validate, deploy) on http://localhost:8501, API on :8000
	IMAGE_ID=$(IMAGE_ID) docker compose --profile portal up -d mlflow api portal

explore: portal

train:      ## train all action models (commit first so the run is tied to a commit)
	$(JOB) python -m nba.train

reproduce:  ## make reproduce RUN_ID=<parent run id>
	$(JOB) python -m nba.reproduce --run-id $(RUN_ID)

evaluate:   ## make evaluate RUN_ID=<parent run id>  (off-policy value, calibration, fairness)
	$(JOB) python -m nba.evaluate --run-id $(RUN_ID)

monitor:    ## make monitor RUN_ID=<parent run id>  (PSI drift of features and scores)
	$(JOB) python -m nba.monitor --run-id $(RUN_ID)

report:     ## figures and results tables from reports/*.json
	$(JOB) python -m nba.report

serve:      ## scoring API on http://localhost:8000: RUN_ID=<id> to pin a run, else the deployed run
	SERVE_RUN_ID=$(RUN_ID) docker compose --profile serve up -d api

test:
	docker compose run --rm --no-deps app pytest -q
