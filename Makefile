IMAGE    := nba-mlops:0.1.0
IMAGE_ID  = $(shell docker image inspect -f '{{.Id}}' $(IMAGE) 2>/dev/null)
JOB       = IMAGE_ID=$(IMAGE_ID) docker compose run --rm app

.PHONY: build up down data train reproduce

build:      ## build the pinned environment image
	docker compose build

up:         ## start the MLflow server on http://localhost:5000
	docker compose up -d mlflow

down:
	docker compose down

data:       ## version data/raw with DVC (after adding or changing the dataset)
	$(JOB) dvc add data/raw
	$(JOB) dvc push

train:      ## train all action models (commit first so the run is tied to a commit)
	$(JOB) python -m nba.train

reproduce:  ## make reproduce RUN_ID=<parent run id>
	$(JOB) python -m nba.reproduce --run-id $(RUN_ID)
