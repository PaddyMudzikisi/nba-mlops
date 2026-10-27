"""NBA MLOps portal: upload data, explore, train, validate, deploy and monitor
from the browser, with the nba package as the backend.

  make portal   ->   http://localhost:8501
"""
import streamlit as st

from nba import datasets, deploy
from nba.portal import common

common.page("NBA MLOps portal")
name = common.active_dataset("home")

st.markdown(
    "Every step below runs the same tested pipeline code as the command line. Each training run records "
    "its parameters, code commit, data fingerprint and Docker image in MLflow, so any run started here "
    "can be regenerated exactly. Data stays on this machine."
)

runs = common.train_runs()
current = deploy.current()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Datasets", len(datasets.list_datasets()))
c2.metric("Training runs", len(runs))
c3.metric("Deployed run", current["run_id"][:8] if current else "none")
drift = common.latest_related("monitors", current["run_id"]) if current else None
c4.metric("Retrain recommended", "-" if drift is None else drift.get("tags.retrain_recommended", "-"))

st.subheader("Pipeline")
steps = [
    ("pages/1_Data.py", "1 · Data", "upload a CSV, privacy check, map columns, version with DVC"),
    ("pages/2_Explore.py", "2 · Explore", "interactive EDA and the standard EDA report"),
    ("pages/3_Train.py", "3 · Train", "choose split, features, model, tuning; train in the background"),
    ("pages/4_Results.py", "4 · Results", "train vs test metrics, overfitting, feature importance, compare runs"),
    ("pages/5_Validate.py", "5 · Validate", "reproducibility test, policy evaluation and fairness, drift monitoring"),
    ("pages/6_Deploy.py", "6 · Deploy", "deployment gate, serve a run through the API, try a recommendation"),
]
for path, title, text in steps:
    c1, c2 = st.columns([1, 4])
    with c1:
        st.page_link(path, label=title)
    c2.write(text)

if current:
    st.subheader("Currently deployed")
    st.write(f"Run `{current['run_id']}` deployed {current['deployed_at']} UTC")
    st.table({"offer": list(current["models"]), "model": list(current["models"].values())})

common.job_panel(title="Recent jobs")
