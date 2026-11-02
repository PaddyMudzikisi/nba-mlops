"""Deploy a validated run behind the scoring API, try a recommendation, and
score a whole customer file in batch."""
import json
import os
from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st

from nba import datasets, deploy
from nba.config import REPO
from nba.explore import load
from nba.portal import common
from nba.privacy import scan
from nba.report import label

API = os.environ.get("API_URL", "http://api:8000")

common.page("7 · Deploy")
name = common.active_dataset("deploy")

current = deploy.current()
if current:
    st.success(f"Deployed: run `{current['run_id']}` since {current['deployed_at']} UTC")
else:
    st.info("Nothing deployed yet.")
try:
    health = requests.get(f"{API}/health", timeout=3).json()
    st.caption(f"Scoring API at {API}: {health.get('status')}"
               + (f" · serving run {health['train_run_id'][:8]}" if "train_run_id" in health else ""))
except requests.RequestException:
    health = None
    st.warning("The scoring API is not reachable; start everything with `make portal`.")

st.subheader("Deployment gate")
runs = common.train_runs(name)
run_id = common.pick_run(runs, "deploy-run")
if run_id is None:
    st.stop()
checks = deploy.checks(run_id)
for check, ok, detail in checks:
    optional = check.endswith("(recommended)")
    mark = "Passed" if ok else ("Recommended" if optional else "Not met")
    st.markdown(f"- **{mark}** · {check}: {detail}")
blocked = [c for c, ok, _ in checks if not ok and not c.endswith("(recommended)")]
if blocked:
    st.caption("Run the missing steps on the Validate page. Runs started from uncommitted code cannot be deployed: "
               "commit first so the served model is tied to an exact version of the code.")
if st.button("Deploy this run", type="primary", disabled=bool(blocked)):
    record = deploy.deploy(run_id)
    try:
        requests.post(f"{API}/reload", timeout=60).raise_for_status()
        st.success(f"Deployed and loaded by the API. Registry alias `production` now points to: "
                   + ", ".join(record["models"].values()))
    except requests.RequestException as e:
        st.warning(f"Deployed, but the API did not reload ({e}); it will load the run when restarted.")
    st.rerun()

st.subheader("Try a recommendation")


def try_recommendation():
    if not health or health.get("status") != "ok":
        st.info("Deploy a run to try it.")
        return
    schema = requests.get(f"{API}/schema", timeout=5).json()
    _, data = load(name)
    example = data.iloc[0]
    with st.form("try"):
        values = {}
        cols = st.columns(3)
        for i, f in enumerate(schema["features"]):
            box = cols[i % 3]
            if f in schema["categories"]:
                options = schema["categories"][f]
                default = str(example[f]) if f in example and str(example[f]) in options else options[0]
                values[f] = box.selectbox(f, options, index=options.index(default))
            else:
                default = float(example[f]) if f in example and pd.notna(example[f]) else 0.0
                values[f] = box.number_input(f, value=default)
        eligible = st.checkbox(f"Eligible ({schema['eligibility_column'] or 'no eligibility rule'} = 1)", value=True)
        submitted = st.form_submit_button("Recommend")
    if submitted:
        r = requests.post(f"{API}/recommend", timeout=10,
                          json={"customer_key": "PORTAL_TEST", "consent_marketing": int(eligible), "features": values})
        if r.ok:
            d = r.json()
            st.markdown(f"**Next best action: {label(d['action']) if d['action'] else 'none'}** ({d['reason']})")
            st.dataframe(pd.DataFrame(d["ranking"]).assign(action=lambda t: t.action.map(label))
                         .style.format({"probability": "{:.1%}", "expected_value": "{:.2f}"}), hide_index=True)
        else:
            st.error(r.json().get("detail", r.text))


try_recommendation()

st.subheader("Decision log")
log = common.decisions()
if log:
    st.dataframe(pd.DataFrame([{"time": d["timestamp"][:19], "customer": d["customer_key"],
                                "action": d["action"], "run": d["train_run_id"][:8]} for d in log]),
                 hide_index=True, width="stretch")
else:
    st.caption("No decisions logged yet.")

st.subheader("Batch scoring")
st.markdown("Upload a CSV with one row per customer and the model's input columns, and download a recommendation "
            "for every row, scored by the deployed run. **Only anonymised, masked or synthetic data.**")
if not current:
    st.info("Deploy a run first.")
    st.stop()


@st.cache_resource(show_spinner="Loading the deployed models...")
def deployed_policy(run_id):
    from nba.policy import Policy
    return Policy.from_run(run_id)


file = st.file_uploader("Customer file (CSV)", type="csv", key="batch-file")
if file is None:
    st.stop()
frame = pd.read_csv(file)
found = scan(frame)
if not found["clean"]:
    st.error("Blocked: values that look like personal data were found. Remove or mask them and upload again.")
    st.stop()
policy = deployed_policy(current["run_id"])
inputs = policy.meta.get("inputs", policy.meta["features"])
missing = [c for c in inputs if c not in frame]
if missing:
    st.error(f"The file is missing these input columns: {', '.join(missing)}")
    st.stop()
eligibility = policy.meta.get("eligibility_column")
if eligibility and eligibility not in frame:
    st.warning(f"No {eligibility} column: every customer is treated as eligible.")
    frame[eligibility] = 1
bad = pd.Series(False, index=frame.index)
for col, categories in policy.meta["categories"].items():
    if col in inputs:
        bad |= ~frame[col].astype("string").fillna("missing").isin(categories)
if bad.any():
    st.warning(f"{int(bad.sum()):,} rows have categories the models never saw in training and are not scored.")
ok = frame[~bad]
keys = [c for c in frame.columns if c not in inputs]
key = st.selectbox("Customer key column to carry into the results", ["(row number)"] + keys,
                   index=1 + keys.index(schema_key) if (schema_key := datasets.load_schema(name)["columns"].get(
                       "customer_key")) in keys else 0)
with st.spinner(f"Scoring {len(ok):,} customers..."):
    proba, ev = policy.score(ok)
    action = policy.recommend(ok)
out = pd.DataFrame({"customer": ok[key] if key != "(row number)" else ok.index + 1,
                    "recommended_action": action, "expected_value": ev.max(axis=1).where(action.notna())})
out = out.join(proba.add_prefix("p_"))
out["train_run_id"] = current["run_id"]
c1, c2 = st.columns([1, 2])
c1.dataframe(out.recommended_action.fillna("none (not eligible)").map(lambda a: label(a) if a in policy.actions else a)
             .value_counts().rename("customers"), width="stretch")
c2.dataframe(out.head(200), hide_index=True, width="stretch")
st.download_button("Download recommendations (CSV)", out.to_csv(index=False).encode(), "recommendations.csv",
                   "text/csv", type="primary")
with open(REPO / "logs" / "batch_scoring.jsonl", "a") as f:
    f.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "train_run_id": current["run_id"], "rows": len(frame), "scored": len(ok)}) + "\n")
