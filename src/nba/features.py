import hashlib

import pandas as pd

CATEGORICAL = ["age_band", "region", "segment", "income_band", "channel"]
NUMERIC = [
    "tenure_months", "salary_credit", "has_savings", "has_loan", "has_card", "has_fd",
    "n_products", "digital_active", "txn_count_30d", "txn_value_30d", "avg_balance_90d",
    "days_since_last_txn", "arrears_flag",
]
FEATURES = CATEGORICAL + NUMERIC


def build_action_frame(customers, offers, action):
    """Labelled rows for one action: only customers who were actually offered it."""
    frame = offers[offers.action == action].merge(customers, on="customer_key", how="left", validate="m:1")
    for col in CATEGORICAL:
        # Categories come from the full data, not the action subset, so every
        # model sees the same encoding.
        source = offers if col in offers else customers
        frame[col] = pd.Categorical(frame[col], categories=sorted(source[col].unique()))
    return frame.sort_values("offer_id").reset_index(drop=True)


def dataset_sha256(*paths):
    h = hashlib.sha256()
    for path in sorted(paths):
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()
