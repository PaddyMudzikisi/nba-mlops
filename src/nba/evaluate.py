"""Offline evaluation of the next-best-action policy on the out-of-time test period.

Each test row is an offer the bank actually made, with the logging policy's
propensity p for that offer. The value a target policy pi would have earned is
estimated from those logs alone (reward r = accepted x offer value, USD):

  IPS    mean( w r ),  w = pi(a_logged | x) / p
  SNIPS  sum( w r ) / sum( w )
  DM     mean( sum_a pi(a | x) q(x, a) ),  q = model expected value
  DR     DM + mean( w (r - q(x, a_logged)) )

Without logged propensities only the direct method (DM) can be estimated.
Also reports probability calibration per action and a fairness check on each
protected attribute of the dataset (protected attributes are not features).
"""
import argparse
import json
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from nba.config import REPO, load_params
from nba.features import load_data
from nba.policy import Policy
from nba.train import provenance, test_rows

REPORTS = REPO / "reports"


def one_hot(choice, actions):
    """Deterministic policy as a rows x actions matrix of pi(a | x); NaN choice = no offer."""
    return pd.DataFrame({a: (choice == a).astype(float) for a in actions}, index=choice.index)


def ope_contributions(target, logged_action, reward, propensity, q):
    """Per-row importance weights and per-row contributions to IPS, DM and DR."""
    col = {a: i for i, a in enumerate(q.columns)}
    rows = np.arange(len(logged_action))
    idx = logged_action.map(col).to_numpy()
    pi, qv = target[q.columns].to_numpy(), q.to_numpy()
    w = pi[rows, idx] / np.asarray(propensity)
    r = np.asarray(reward)
    dm = (pi * qv).sum(axis=1)
    return w, r, {"IPS": w * r, "DM": dm, "DR": dm + w * (r - qv[rows, idx])}


def estimate(w, r, contrib, n_boot, seed):
    """Point estimates with percentile bootstrap 95% intervals (rows resampled)."""
    rng = np.random.default_rng(seed)
    boot = rng.integers(0, len(w), size=(n_boot, len(w)))
    out = {}
    for name, c in contrib.items():
        dist = c[boot].mean(axis=1)
        out[name] = (c.mean(), *np.percentile(dist, [2.5, 97.5]))
    dist = (w * r)[boot].sum(axis=1) / w[boot].sum(axis=1)
    out["SNIPS"] = ((w * r).sum() / w.sum(), *np.percentile(dist, [2.5, 97.5]))
    return {k: {"value": float(v), "ci_low": float(lo), "ci_high": float(hi)} for k, (v, lo, hi) in out.items()}


def fairness(test, recommended, proba, attribute="gender"):
    """Recommendation mix, acceptance and model discrimination per group of a protected attribute."""
    rows = np.arange(len(test))
    logged_p = proba.to_numpy()[rows, test.action.map({a: i for i, a in enumerate(proba.columns)}).to_numpy()]
    out = {}
    for group, idx in test.groupby(test[attribute].astype(str), observed=True).indices.items():
        g = test.iloc[idx]
        out[group] = {
            "n": int(len(idx)),
            "acceptance_rate": float(g.accepted.mean()),
            "auc_logged_action": (float(roc_auc_score(g.accepted, logged_p[idx]))
                                  if g.accepted.nunique() == 2 else float("nan")),
            "recommendation_share": recommended.iloc[idx].value_counts(normalize=True).round(4).to_dict(),
        }
    return out


def protected(params, frame):
    """Protected attributes to audit; runs recorded before the setting existed audited gender."""
    attrs = params["dataset"].get("protected", ["gender"])
    return [a for a in attrs if a in frame]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="parent training run to evaluate")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--out", default=str(REPORTS), help="folder for evaluation.json")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        params = load_params(mlflow.artifacts.download_artifacts(
            run_id=args.run_id, artifact_path="params.yaml", dst_path=tmp))
    data = load_data(params)
    train, test = test_rows(data, params)
    policy = Policy.from_run(args.run_id)
    actions = policy.actions

    proba, q = policy.score(test)
    recommended = policy.recommend(test)
    reward = test.accepted * test.value
    train_reward = (train.accepted * train.value).groupby(train.action).mean()
    targets = {
        "NBA (expected value)": one_hot(recommended, actions),
        "Best single offer": one_hot(pd.Series(train_reward.idxmax(), index=test.index), actions),
        "Most offered": one_hot(pd.Series(train.action.mode()[0], index=test.index), actions),
        "Random": pd.DataFrame(1 / len(actions), index=test.index, columns=actions),
    }

    # If the logged propensities are proper probabilities over these actions,
    # E[1 / p] = number of actions, so the mean weight under a uniform policy is 1.
    has_propensity = bool(test.propensity.notna().all())
    propensity_check = float((1 / len(actions) / test.propensity).mean()) if has_propensity else None
    # Without propensities only the direct method applies: weights of 1 are
    # placeholders, and only the DM estimate is kept.
    propensity = test.propensity if has_propensity else pd.Series(1.0, index=test.index)

    results, acceptance = {}, {}
    for name, target in targets.items():
        w, r, contrib = ope_contributions(target, test.action, reward, propensity, q)
        est = estimate(w, r, contrib, args.n_boot, params["seed"])
        results[name] = ({**est, "matched_offers": int((w > 0).sum()),
                          "effective_sample_size": float(w.sum() ** 2 / (w ** 2).sum())}
                         if has_propensity else {"DM": est["DM"]})
        # Same policies, judged on acceptance rate instead of value.
        w, r, contrib = ope_contributions(target, test.action, test.accepted, propensity, proba)
        est = estimate(w, r, contrib, args.n_boot, params["seed"])
        acceptance[name] = est if has_propensity else {"DM": est["DM"]}
    logged = np.random.default_rng(params["seed"]).integers(0, len(test), size=(args.n_boot, len(test)))
    logged_dist = reward.to_numpy()[logged].mean(axis=1)
    results["Logged (bank's actual offers)"] = {"observed": float(reward.mean()),
                                                "ci_low": float(np.percentile(logged_dist, 2.5)),
                                                "ci_high": float(np.percentile(logged_dist, 97.5))}

    calibration = {}
    for a in actions:
        rows = test[test.action == a]
        raw = policy.models[a].predict_proba(policy.encode(rows))[:, 1]
        calibration[a] = {"observed_rate": float(rows.accepted.mean()),
                          "mean_raw_proba": float(raw.mean()),
                          "mean_calibrated_proba": float(proba.loc[rows.index, a].mean()),
                          "brier": float(brier_score_loss(rows.accepted, proba.loc[rows.index, a]))}

    report = {
        "train_run_id": args.run_id,
        "test_period": [str(test.offer_date.min().date()), str(test.offer_date.max().date())],
        "n_test_offers": int(len(test)),
        "policy_value_per_offer_usd": results,
        "policy_acceptance_rate": acceptance,
        "logged_acceptance_rate": float(test.accepted.mean()),
        "recommendation_share": {
            "NBA": recommended.value_counts(normalize=True).round(4).to_dict(),
            "Logged": test.action.value_counts(normalize=True).round(4).to_dict(),
        },
        "calibration": calibration,
        "propensity_check_mean_uniform_weight": propensity_check,
        "headline_estimator": "DR" if has_propensity else "DM",
        "fairness": {attr: fairness(test, recommended, proba, attr) for attr in protected(params, test)},
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps(report, indent=2))

    mlflow.set_experiment(params["mlflow"]["experiment"])
    with mlflow.start_run(run_name="evaluate"):
        mlflow.set_tags({"evaluates": args.run_id, **provenance(params)})
        for name, res in results.items():
            key = name.split(" (")[0].lower().replace(" ", "_")
            mlflow.log_metric(f"value_{key}", res.get(report["headline_estimator"], {}).get("value", res.get("observed")))
        mlflow.log_dict(report, "evaluation.json")

    print(f"Policy value per offer, USD, test {report['test_period'][0]} to {report['test_period'][1]} "
          f"(n={len(test):,}):")
    for name, res in results.items():
        if "observed" in res:
            print(f"  {name:30s} observed {res['observed']:6.2f}  [{res['ci_low']:.2f}, {res['ci_high']:.2f}]")
            continue
        cells = "  ".join(f"{k} {res[k]['value']:6.2f} [{res[k]['ci_low']:.2f}, {res[k]['ci_high']:.2f}]"
                          for k in ["IPS", "SNIPS", "DR", "DM"] if k in res)
        print(f"  {name:30s} {cells}  matched={res.get('matched_offers', '-')}")
    h = report["headline_estimator"]
    print(f"\nAcceptance rate, {h} (logged observed {test.accepted.mean():.3f}):")
    for name, res in acceptance.items():
        print(f"  {name:30s} {res[h]['value']:.3f} [{res[h]['ci_low']:.3f}, {res[h]['ci_high']:.3f}]")
    print("\nCalibration (observed vs mean predicted, raw -> calibrated):")
    for a, c in calibration.items():
        print(f"  {a:26s} {c['observed_rate']:.3f}  {c['mean_raw_proba']:.3f} -> {c['mean_calibrated_proba']:.3f}"
              f"  Brier {c['brier']:.4f}")
    if has_propensity:
        print(f"\nPropensity check: mean weight under uniform policy = {propensity_check:.3f} (1.0 if consistent)")
    else:
        print("\nNo logged propensities: only direct-method (model-based) estimates are reported.")
    print("\nRecommendation share, NBA:", report["recommendation_share"]["NBA"])
    for attr, groups in report["fairness"].items():
        print(f"\nBy {attr}:")
        for g, f in groups.items():
            print(f"  {g:18s} n={f['n']:5d} accept={f['acceptance_rate']:.3f} AUC={f['auc_logged_action']:.3f} "
                  f"mix={f['recommendation_share']}")


if __name__ == "__main__":
    main()
