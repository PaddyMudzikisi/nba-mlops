# Reproducible MLOps pipeline for Next-Best-Action recommendation

MSc Big Data Analytics project. One XGBoost acceptance model per offer, combined
into a next-best-action (NBA) policy that recommends the eligible offer with the
highest expected value (calibrated acceptance probability x offer value). Every
training run is tied to a git commit, a DVC data snapshot and a Docker image,
and a reproducibility test regenerates it from those records.

Results for Chapter 4: [reports/results.md](reports/results.md).

## Data and privacy

Only anonymised, masked or synthetic data is used. Customer identifiers are
pseudonymous keys (`SYN...`); no names, national IDs, account numbers, phone
numbers or e-mail addresses. The dataset is versioned with DVC and is **not**
stored in git.

Input is one CSV with one row per offer made to a customer. Column roles are
mapped in [params.yaml](params.yaml) under `dataset:`:

| Role | Column in `mydata.csv` | Use |
|---|---|---|
| customer key | `customer_id_pseudo` | pseudonymous id |
| offer date | `offer_date` | out-of-time split, monitoring windows |
| action | `offer_name` | one model per offer |
| accepted | `target_accepted` | label |
| value | `estimated_offer_value_usd` | reward for ranking and evaluation, not a feature |
| propensity | `logged_propensity` | off-policy evaluation only, not a feature |

`gender` is deliberately not a feature; it is used only for the fairness
check. `consent_marketing` is an eligibility rule (no recommendation without
consent).

## Architecture

All services run in Docker on one machine, from a single pinned image
(`requirements.lock` is the full `pip freeze`, not just top-level packages).

| Component | Role |
|---|---|
| Git / GitHub | code version; every MLflow run records the commit |
| DVC (local remote `~/nba-mlops-dvcstore`) | data version; runs record the dataset SHA-256 |
| MLflow server, http://localhost:5001 | experiments, artefacts, model registry (`nba_<offer>`) |
| Job container (`make train` etc.) | training, reproduction, evaluation, monitoring, reports |
| Scoring API, http://localhost:8000 | `POST /recommend`, decision log with run and model versions |

MLflow is on port 5001 because macOS AirPlay Receiver occupies 5000.

## Running it

Requires Docker Desktop.

```
make build                 # build the pinned image
make up                    # start MLflow on http://localhost:5001
make data                  # after adding/changing data/raw: dvc add + dvc push
git commit ...             # commit first, so the run is tied to an exact commit
make train                 # trains all offer models; prints the parent run id
make reproduce RUN_ID=...  # retrain from the recorded params and compare predictions
make evaluate RUN_ID=...   # off-policy value, calibration, fairness -> reports/evaluation.json
make monitor RUN_ID=...    # data and outcome drift -> reports/drift.json
docker compose run --rm --no-deps app python experiments/retrain_after_drift.py
make report                # figures and reports/results.md
make serve RUN_ID=...      # scoring API on http://localhost:8000
make test                  # unit tests (toy data; no dataset or MLflow needed)
make down                  # stop MLflow and the API
```

Example request:

```
curl -s -X POST http://localhost:8000/recommend -H 'Content-Type: application/json' -d '{
  "customer_key": "SYN000109",
  "features": {"offer_channel": "Mobile_App", "age_band": "35-44", "employment_type": "Formal_employment",
               "income_band": "1000-2499", "region": "Harare", "balance_band": "1501-5000",
               "contact_count": 1, "tenure_months": 88, "products_held_count": 3, "has_loan": 0,
               "has_card": 1, "txn_count_30d": 9, "txn_value_30d": 383.47, "salary_inflow_flag": 1,
               "mobile_logins_30d": 6}}'
```

## Method notes

- **Labelling.** Each offer model learns only from customers who were actually
  offered that product.
- **Out-of-time split.** Train on offers up to `split.train_end` (2025-12-31),
  test on Jan-Jun 2026.
- **Calibration.** Models are trained without class weighting: weighting
  inflated the probabilities and the odds correction then under-predicted,
  which distorts a ranking across offers
  ([experiments/calibration_comparison.py](experiments/calibration_comparison.py)).
- **Off-policy evaluation.** IPS, SNIPS, direct method and doubly robust
  estimates of value per offer and acceptance rate, with bootstrap 95%
  intervals, against the bank's logged offers, "always the best single offer",
  "always the most offered" and random.
- **Monitoring.** PSI of features and model scores, plus observed vs predicted
  acceptance per offer; cumulative counters (`contact_count`) are reported as
  expected trends and do not trigger retraining.
- **Reproducibility test.** `make reproduce` checks data hash, commit, image,
  platform and library versions, retrains from the recorded `params.yaml`, and
  passes only if every test prediction matches within `1e-6`.

## Layout

```
params.yaml            dataset schema, split, model and monitoring settings
src/nba/features.py    load CSV, map columns, encode categories
src/nba/train.py       train and register models, log provenance and policy.json
src/nba/policy.py      expected-value next-best-action policy
src/nba/reproduce.py   reproducibility test
src/nba/evaluate.py    off-policy evaluation, calibration, fairness
src/nba/monitor.py     drift monitoring
src/nba/serve.py       FastAPI scoring service
src/nba/report.py      figures and results tables
experiments/           calibration comparison, retraining after drift
tests/                 unit tests on toy data
reports/               generated results (no customer-level data)
```
