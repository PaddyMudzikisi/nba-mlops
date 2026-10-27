from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
PARAMS_PATH = REPO / "params.yaml"


def load_params(path=PARAMS_PATH):
    with open(path) as f:
        return yaml.safe_load(f)
