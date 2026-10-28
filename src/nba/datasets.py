"""Dataset registry: each dataset is data/datasets/<name>/ with a schema.yaml
(the column mapping and feature lists of params.yaml's dataset section) and its
CSV, versioned with DVC."""
import copy
import re
import subprocess
from datetime import datetime, timezone

import pandas as pd
import yaml

from nba.config import REPO, load_params

DATASETS = REPO / "data" / "datasets"
ROLES = ["customer_key", "offer_date", "action", "accepted", "value", "propensity"]
REQUIRED_ROLES = ["customer_key", "offer_date", "action", "accepted"]
NAME = re.compile(r"^[a-z][a-z0-9_]{1,39}$")


def list_datasets():
    out = []
    for path in sorted(DATASETS.glob("*/schema.yaml")):
        schema = yaml.safe_load(path.read_text())
        out.append(schema)
    return out


def load_schema(name):
    return yaml.safe_load((DATASETS / name / "schema.yaml").read_text())


def params_for(name, base=None):
    """The default params with this dataset's schema in place of params.yaml's."""
    params = copy.deepcopy(base or load_params())
    params["dataset"] = load_schema(name)
    return params


def guess_schema(frame):
    """Suggested role and feature type for each column, by name and content."""
    cols = list(frame.columns)

    text_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(frame[c])]

    def find(*patterns, among=None):
        for p in patterns:
            for c in cols if among is None else among:
                if re.search(p, c, re.IGNORECASE):
                    return c
        return None

    roles = {
        "customer_key": find(r"customer.*(id|key|ref)", r"(user|client|cust|member).*(id|key|ref|no)",
                             r"^(client|customer|member)", r"^id$"),
        "offer_date": find(r"date", r"time"),
        "action": find(r"offer.*(name|type)", r"^offer$", r"product", r"action", r"campaign",
                       among=text_cols),  # an offer is a label, never a count
        "accepted": find(r"accept", r"target", r"label", r"convert", r"response", r"outcome", r"took_?up",
                         r"uptake", r"purchas", r"bought", r"success"),
        "value": find(r"value", r"revenue", r"amount"),
        "propensity": find(r"propensity", r"prob"),
    }
    if not roles["accepted"]:  # any yes/no column
        from nba.features import FALSE_VALUES, TRUE_VALUES
        for c in cols:
            values = set(frame[c].dropna().astype(str).str.strip().str.lower())
            if values and values <= TRUE_VALUES | FALSE_VALUES and c not in roles.values():
                roles["accepted"] = c
                break
    used = {v for v in roles.values() if v}
    categorical, numeric = [], []
    for c in cols:
        if c in used:
            continue
        s = frame[c]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 2 or set(s.dropna().unique()) <= {0, 1}:
            numeric.append(c)
        elif s.nunique() <= 50:
            categorical.append(c)
    protected = [c for c in categorical if re.search(r"gender|sex|race|ethnic|religion", c, re.IGNORECASE)]
    categorical = [c for c in categorical if c not in protected]
    eligibility = next((c for c in numeric if re.search(r"consent|opt.?in", c, re.IGNORECASE)), None)
    numeric = [c for c in numeric if c != eligibility]
    return {"columns": roles, "categorical": categorical, "numeric": numeric, "protected": protected,
            "eligibility_column": eligibility}


def validate(frame, schema):
    """Problems that would stop the pipeline, as readable messages."""
    problems = []
    roles = schema["columns"]
    for role in REQUIRED_ROLES:
        if not roles.get(role):
            problems.append(f"choose a column for {role.replace('_', ' ')}")
    mapped = [c for c in roles.values() if c]
    if len(mapped) != len(set(mapped)):
        problems.append("the same column is used for two roles")
    overlap = set(mapped) & set(schema["categorical"] + schema["numeric"])
    if overlap:
        problems.append(f"columns used as roles cannot also be features: {sorted(overlap)}")
    if set(schema.get("protected", [])) & set(schema["categorical"] + schema["numeric"]):
        problems.append("protected attributes cannot be model features")
    if not schema["categorical"] + schema["numeric"]:
        problems.append("choose at least one feature")
    if problems:
        return problems
    try:
        dates = pd.to_datetime(frame[roles["offer_date"]], format=schema.get("date_format"))
        if dates.isna().any():
            problems.append("some offer dates could not be read")
    except (ValueError, TypeError) as e:
        problems.append(f"offer dates could not be read: {e}")
    from nba.features import to_binary
    try:
        y = to_binary(frame[roles["accepted"]])
        counts = frame.assign(_y=y).groupby(roles["action"])._y.agg(["sum", "size"])
        thin = counts[(counts["sum"] < 10) | (counts["size"] - counts["sum"] < 10)].index.tolist()
        if thin:
            problems.append(f"offers with fewer than 10 acceptances or 10 rejections cannot be modelled: {thin}")
    except ValueError as e:
        problems.append(str(e))
    for role in ["value", "propensity"]:
        c = roles.get(role)
        if c and not pd.api.types.is_numeric_dtype(frame[c]):
            problems.append(f"{role} column must be numeric")
    c = roles.get("propensity")
    if c and pd.api.types.is_numeric_dtype(frame[c]) and not frame[c].between(0, 1, inclusive="right").all():
        problems.append("propensity must be in (0, 1]")
    return problems


def save(name, csv_bytes, schema, description=""):
    """Write the dataset and its schema, then version the CSV with DVC."""
    if not NAME.match(name):
        raise ValueError("name: lower case letters, digits and _, starting with a letter")
    folder = DATASETS / name
    if folder.exists():
        raise ValueError(f"dataset {name!r} already exists")
    folder.mkdir(parents=True)
    (folder / "data.csv").write_bytes(csv_bytes)
    schema = {"name": name, "description": description, "path": f"data/datasets/{name}/data.csv",
              "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), **schema}
    (folder / "schema.yaml").write_text(yaml.safe_dump(schema, sort_keys=False))
    log = []
    for cmd in (["dvc", "add", f"data/datasets/{name}/data.csv"], ["dvc", "push"]):
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        log.append(f"$ {' '.join(cmd)}\n{r.stdout}{r.stderr}")
    return schema, "\n".join(log)
