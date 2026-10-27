"""Deploy a validated run behind the scoring API, then try a recommendation."""
import os

import pandas as pd
import requests
import streamlit as st

from nba import deploy
from nba.explore import load
from nba.portal import common
from nba.report import label

API = os.environ.get("API_URL", "http://api:8000")

common.page("6 · Deploy")
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
if not health or health.get("status") != "ok":
    st.info("Deploy a run to try it.")
    st.stop()
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

st.subheader("Decision log")
log = common.decisions()
if log:
    st.dataframe(pd.DataFrame([{"time": d["timestamp"][:19], "customer": d["customer_key"],
                                "action": d["action"], "run": d["train_run_id"][:8]} for d in log]),
                 hide_index=True, width="stretch")
else:
    st.caption("No decisions logged yet.")
