from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
PARAMS_PATH = REPO / "params.yaml"
RAW_DIR = REPO / "data" / "raw"
CUSTOMERS_PATH = RAW_DIR / "customers.parquet"
OFFERS_PATH = RAW_DIR / "offers.parquet"


def load_params(path=PARAMS_PATH):
    with open(path) as f:
        return yaml.safe_load(f)
