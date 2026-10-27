"""Validate a training run: reproducibility test, offline policy evaluation with
fairness, and drift monitoring. Each runs as a background job; the latest
results recorded in MLflow are shown."""
import pandas as pd
import streamlit as st

from nba import jobs
from nba.portal import common
from nba.report import label

common.page("5 · Validate")
name = common.active_dataset("validate")
runs = common.train_runs(name)
run_id = common.pick_run(runs, "validate-run")
if run_id is None:
    st.stop()

c1, c2, c3 = st.columns(3)
if c1.button("Run reproducibility test", width="stretch"):
    jobs.start("reproduce", ["--run-id", run_id], label=f"reproduce {run_id[:8]}")
if c2.button("Evaluate the policy", width="stretch"):
    jobs.start("evaluate", ["--run-id", run_id, "--out", f"logs/portal/{run_id}"], label=f"evaluate {run_id[:8]}")
if c3.button("Check for drift", width="stretch"):
    jobs.start("monitor", ["--run-id", run_id, "--out", f"logs/portal/{run_id}"], label=f"monitor {run_id[:8]}")
common.job_panel(match=run_id, title="Jobs for this run")

st.subheader("Reproducibility")
repro = common.latest_related("reproduces", run_id)
if repro is None:
    st.info("Not tested yet. The test retrains every model from the recorded parameters and data and passes "
            "only if every test prediction matches within the tolerance.")
else:
    passed = repro.get("tags.passed") == "True"
    (st.success if passed else st.error)(
        f"{'PASSED' if passed else 'FAILED'} · tested {repro.start_time:%Y-%m-%d %H:%M} UTC")
    diffs = {c.removeprefix("metrics.max_abs_diff_"): repro[c] for c in repro.index
             if c.startswith("metrics.max_abs_diff_")}
    st.dataframe(pd.DataFrame({"offer": [label(a) for a in diffs], "max |difference|": list(diffs.values())}),
                 hide_index=True)

st.subheader("Policy evaluation")
ev_run = common.latest_related("evaluates", run_id)
ev = common.load_artifact_json(ev_run.run_id, "evaluation.json") if ev_run is not None else None
if ev is None:
    st.info("Not evaluated yet. Estimates the value and acceptance rate the recommendation policy would have "
            "achieved on the held-out offers, against simple baselines, plus a fairness check.")
else:
    h = ev.get("headline_estimator", "DR")
    rows = []
    for policy, res in ev["policy_value_per_offer_usd"].items():
        acc = ev["policy_acceptance_rate"].get(policy)
        if "observed" in res:
            rows.append({"policy": policy, "value per offer": res["observed"], "95% CI": f"{res['ci_low']:.2f}–{res['ci_high']:.2f}",
                         "acceptance rate": ev["logged_acceptance_rate"]})
        else:
            rows.append({"policy": policy, "value per offer": res[h]["value"],
                         "95% CI": f"{res[h]['ci_low']:.2f}–{res[h]['ci_high']:.2f}",
                         "acceptance rate": acc[h]["value"] if acc else None})
    st.caption(f"{ev['n_test_offers']:,} held-out offers · estimator: "
               + ("doubly robust (uses logged propensities)" if h == "DR" else
                  "direct method (model-based; the dataset has no logged propensities)"))
    st.dataframe(pd.DataFrame(rows).style.format({"value per offer": "{:.2f}", "acceptance rate": "{:.1%}"}),
                 hide_index=True, width="stretch")
    if ev.get("propensity_check_mean_uniform_weight") is not None:
        st.caption(f"Propensity check: mean weight under a uniform policy = "
                   f"{ev['propensity_check_mean_uniform_weight']:.3f} (1.0 if the logged propensities are consistent).")
    fairness = ev.get("fairness") or {"gender": ev.get("fairness_by_gender", {})}
    for attr, groups in fairness.items():
        st.markdown(f"**Fairness by {attr}** ({attr} is not a model input)")
        rows = [{"group": g, "n": f["n"], "acceptance": f["acceptance_rate"], "AUC": f["auc_logged_action"],
                 "most recommended": max(f["recommendation_share"], key=f["recommendation_share"].get, default="-"),
                 "its share": max(f["recommendation_share"].values(), default=None)}
                for g, f in groups.items()]
        st.dataframe(pd.DataFrame(rows).style.format({"acceptance": "{:.1%}", "AUC": "{:.3f}", "its share": "{:.1%}"},
                                                     na_rep="-"), hide_index=True)

st.subheader("Drift monitoring")
mon_run = common.latest_related("monitors", run_id)
drift = common.load_artifact_json(mon_run.run_id, "drift.json") if mon_run is not None else None
if drift is None:
    st.info("Not checked yet. Compares the data after the training window with the training data (PSI) and "
            "observed with predicted acceptance per offer.")
else:
    if drift["retrain_recommended"]:
        st.warning(f"Retrain recommended: {', '.join(drift['retrain_reasons'])}. Start a new run on the Train page "
                   "with a later cut-off date.")
    else:
        st.success("No retraining needed.")
    st.caption(f"Reference window ends {drift.get('reference_end', '?')}.")
    c1, c2 = st.columns(2)
    psi = pd.DataFrame([{"signal": k, "PSI": v["value"], "status": v["status"]} for k, v in drift["psi"].items()])
    c1.dataframe(psi.style.format({"PSI": "{:.4f}"}), hide_index=True, width="stretch")
    out = pd.DataFrame([{"offer": label(a), "observed": v["observed_rate"], "predicted": v["mean_predicted"],
                         "gap": v["relative_gap"], "status": v["status"]} for a, v in drift["outcome_drift"].items()])
    c2.dataframe(out.style.format({"observed": "{:.1%}", "predicted": "{:.1%}", "gap": "{:+.0%}"}),
                 hide_index=True, width="stretch")
