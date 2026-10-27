"""Generate a fully synthetic customer and offer dataset.

Mirrors the schema expected from the anonymised Bank extract so the whole
pipeline can be built and tested without any real customer data. Only offered
(exposed) customer-action pairs carry a label, matching the labelling rule in
section 2.5 of the proposal.
"""
import numpy as np
import pandas as pd

from nba.config import CUSTOMERS_PATH, OFFERS_PATH, RAW_DIR, load_params

AGE_BANDS = ["18-24", "25-34", "35-44", "45-54", "55+"]
INCOME_BANDS = ["low", "mid", "high"]
CHANNELS = ["sms", "app", "branch", "call"]


def make_customers(rng, n):
    age = rng.choice(AGE_BANDS, n, p=[0.15, 0.30, 0.25, 0.18, 0.12])
    income = rng.choice(INCOME_BANDS, n, p=[0.50, 0.38, 0.12])
    income_idx = pd.Series(income).map({b: i for i, b in enumerate(INCOME_BANDS)}).to_numpy()
    young = np.isin(age, ["18-24", "25-34"])

    salary_credit = rng.random(n) < 0.45
    has_savings = rng.random(n) < 0.55
    has_loan = rng.random(n) < 0.20
    has_card = rng.random(n) < 0.15
    has_fd = rng.random(n) < 0.08
    digital_active = rng.random(n) < 0.40 + 0.25 * young
    txn_count_30d = rng.poisson(6 + 10 * digital_active + 5 * salary_credit)

    return pd.DataFrame({
        "customer_key": [f"C{i:07d}" for i in rng.permutation(n)],
        "age_band": age,
        "region": rng.choice([f"R{i:02d}" for i in range(1, 11)], n),
        "segment": rng.choice(["mass", "emerging", "premium", "sme"], n, p=[0.60, 0.25, 0.08, 0.07]),
        "income_band": income,
        "tenure_months": rng.integers(1, 240, n),
        "salary_credit": salary_credit.astype(int),
        "has_savings": has_savings.astype(int),
        "has_loan": has_loan.astype(int),
        "has_card": has_card.astype(int),
        "has_fd": has_fd.astype(int),
        "n_products": 1 + has_savings + has_loan + has_card + has_fd,
        "digital_active": digital_active.astype(int),
        "txn_count_30d": txn_count_30d,
        "txn_value_30d": np.round(txn_count_30d * rng.lognormal(3 + 0.5 * income_idx, 0.7), 2),
        "avg_balance_90d": np.round(rng.lognormal(5 + 0.8 * income_idx, 1.0), 2),
        "days_since_last_txn": np.clip(rng.exponential(12 / (1 + digital_active)), 0, 180).astype(int),
        "arrears_flag": (has_loan & (rng.random(n) < 0.10)).astype(int),
    })


def eligible(df, action):
    return {
        "savings_upgrade": df.has_savings == 1,
        "personal_loan": df.arrears_flag == 0,
        "credit_card": df.has_card == 0,
        "fixed_deposit": df.has_fd == 0,
        "digital_enrol": df.digital_active == 0,
    }[action]


def acceptance_logit(df, months_in):
    z_bal = (np.log(df.avg_balance_90d) - 5.6) / 1.1
    log_txn = np.log1p(df.txn_count_30d)
    mid_high = df.income_band.isin(["mid", "high"])
    channel = df.channel.map({"sms": 0.0, "app": 0.3, "branch": 0.2, "call": 0.1})
    logits = {
        "savings_upgrade": -2.0 + 0.6 * z_bal + 0.4 * df.salary_credit + 0.2 * df.tenure_months / 120,
        "personal_loan": -3.0 + 0.8 * df.salary_credit + 0.5 * mid_high - 1.5 * df.has_loan
                         + 0.3 * df.age_band.isin(["25-34", "35-44"])
                         - 0.4 * (months_in >= 9),  # built-in drift for monitoring tests
        "credit_card": -2.8 + 0.6 * df.digital_active + 0.7 * (df.income_band == "high")
                       + 0.4 * df.salary_credit + 0.3 * log_txn,
        "fixed_deposit": -3.2 + 1.2 * z_bal + 0.5 * df.age_band.isin(["45-54", "55+"]),
        "digital_enrol": -1.8 - 0.6 * (df.age_band == "55+") + 0.5 * df.age_band.isin(["18-24", "25-34"])
                         + 0.4 * log_txn - 0.02 * df.days_since_last_txn,
    }
    out = np.zeros(len(df))
    for action, logit in logits.items():
        mask = (df.action == action).to_numpy()
        out[mask] = logit.to_numpy()[mask]
    return out + channel.to_numpy()


def make_offers(rng, customers, actions, start, months):
    n_off = rng.poisson(3, len(customers))
    offers = customers.iloc[np.repeat(np.arange(len(customers)), n_off)].reset_index(drop=True)
    offers["action"] = rng.choice(actions, len(offers))
    keep = np.zeros(len(offers), dtype=bool)
    for action in actions:
        keep |= (offers.action == action) & eligible(offers, action)
    offers = offers[keep].reset_index(drop=True)

    start = pd.Timestamp(start)
    span_days = (start + pd.DateOffset(months=months) - start).days
    offers["offer_date"] = start + pd.to_timedelta(rng.integers(0, span_days, len(offers)), unit="D")
    offers["channel"] = np.where(offers.digital_active == 1,
                                 rng.choice(CHANNELS, len(offers)),
                                 rng.choice(["sms", "branch", "call"], len(offers)))
    months_in = ((offers.offer_date.dt.year - start.year) * 12 + offers.offer_date.dt.month - start.month).to_numpy()
    p = 1 / (1 + np.exp(-acceptance_logit(offers, months_in)))
    offers["accepted"] = (rng.random(len(offers)) < p).astype(int)

    offers = offers.sort_values(["offer_date", "customer_key", "action"]).reset_index(drop=True)
    offers.insert(0, "offer_id", [f"O{i:08d}" for i in range(len(offers))])
    return offers[["offer_id", "customer_key", "action", "offer_date", "channel", "accepted"]]


def main():
    params = load_params()
    rng = np.random.default_rng(params["seed"])
    customers = make_customers(rng, params["data"]["n_customers"])
    offers = make_offers(rng, customers, list(params["actions"]),
                         params["data"]["start"], params["data"]["months"])

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    customers.to_parquet(CUSTOMERS_PATH, index=False)
    offers.to_parquet(OFFERS_PATH, index=False)
    print(f"customers: {len(customers):,}  offers: {len(offers):,}")
    print(offers.groupby("action").accepted.agg(offers="size", acceptance_rate="mean").round(3))


if __name__ == "__main__":
    main()
