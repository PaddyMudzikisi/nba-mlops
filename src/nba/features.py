import hashlib

import numpy as np
import pandas as pd

from nba.config import REPO

TRUE_VALUES = {"1", "true", "yes", "y", "accepted", "t"}
FALSE_VALUES = {"0", "false", "no", "n", "rejected", "declined", "f"}


def features(params):
    ds = params["dataset"]
    return ds["categorical"] + ds["numeric"]


def to_binary(series):
    """0/1 outcome from numbers, booleans or yes/no style text."""
    if pd.api.types.is_numeric_dtype(series) and set(series.dropna().unique()) <= {0, 1}:
        return series.astype(int)
    text = series.astype(str).str.strip().str.lower()
    unknown = sorted(set(text) - TRUE_VALUES - FALSE_VALUES)
    if unknown:
        raise ValueError(f"outcome column has values that are not yes/no: {unknown[:5]}")
    return text.isin(TRUE_VALUES).astype(int)


def load_data(params):
    """Read the offers CSV and rename its columns to the pipeline's roles.
    Offer value and logged propensity are optional: without a value every
    acceptance is worth 1 (the policy then ranks by probability), and without a
    propensity off-policy estimates that need it are skipped."""
    ds = params["dataset"]
    frame = pd.read_csv(REPO / ds["path"])
    frame = frame.rename(columns={col: role for role, col in ds["columns"].items() if col})
    frame["offer_date"] = pd.to_datetime(frame.offer_date, format=ds.get("date_format"))
    frame["customer_key"] = frame.customer_key.astype(str)
    frame["action"] = frame.action.astype(str)
    frame["accepted"] = to_binary(frame.accepted)
    if "value" not in frame:
        frame["value"] = 1.0
    if "propensity" not in frame:
        frame["propensity"] = np.nan
    key = frame.customer_key + "|" + frame.offer_date.dt.strftime("%Y-%m-%d") + "|" + frame.action
    # Customer, date and action are normally unique; repeats get a counter in
    # file order, which the dataset hash pins down.
    repeat = key.groupby(key).cumcount()
    frame["offer_id"] = key.where(repeat == 0, key + "|" + repeat.astype(str))
    for col in ds["categorical"]:
        # Categories come from the full data, not an action subset, so every
        # model sees the same encoding.
        values = frame[col].astype("string").fillna("missing")
        frame[col] = pd.Categorical(values, categories=sorted(values.unique()))
    for col in ds["numeric"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def build_action_frame(frame, action):
    """Labelled rows for one action: only customers who were actually offered it."""
    return frame[frame.action == action].sort_values("offer_id").reset_index(drop=True)


def dataset_sha256(*paths):
    h = hashlib.sha256()
    for path in sorted(paths):
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()
