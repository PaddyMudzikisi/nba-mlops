import hashlib

import pandas as pd

from nba.config import REPO


def features(params):
    ds = params["dataset"]
    return ds["categorical"] + ds["numeric"]


def load_data(params):
    """Read the offers CSV and rename its columns to the pipeline's roles."""
    ds = params["dataset"]
    frame = pd.read_csv(REPO / ds["path"])
    frame = frame.rename(columns={col: role for role, col in ds["columns"].items()})
    frame["offer_date"] = pd.to_datetime(frame.offer_date)
    # The CSV has no offer id; customer, date and action are unique together.
    frame["offer_id"] = frame.customer_key + "|" + frame.offer_date.dt.strftime("%Y-%m-%d") + "|" + frame.action
    if frame.offer_id.duplicated().any():
        raise ValueError("customer_key + offer_date + action is not unique")
    for col in ds["categorical"]:
        # Categories come from the full data, not an action subset, so every
        # model sees the same encoding.
        frame[col] = pd.Categorical(frame[col], categories=sorted(frame[col].unique()))
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
