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
st.caption("The steps follow the machine learning life cycle: data collection, pre-processing (cleaning and "
           "feature engineering), training and validation, testing, deployment, and maintenance.")
steps = [
    ("pages/1_Data.py", "1 · Data", "Data collection and cleaning",
     "upload a CSV, privacy check, map columns, clean (missing values, outliers, duplicates), version with DVC"),
    ("pages/2_Explore.py", "2 · Explore", "Exploratory data analysis",
     "bar and pie charts, histograms, box plots, time plots, summary statistics, contingency tables, "
     "binomial and normal models, statistical tests"),
    ("pages/3_Features.py", "3 · Features", "Feature engineering",
     "feature creation, log transform, binning, binarisation, date split; feature selection by relevance"),
    ("pages/4_Train.py", "4 · Train", "Model building and training",
     "split, imputation, scaling, one-hot encoding, PCA, model family, hyperparameter tuning, model selection"),
    ("pages/5_Results.py", "5 · Results", "Model testing",
     "accuracy, confusion matrix, precision, recall, F1, ROC-AUC; overfitting and underfitting; importance"),
    ("pages/6_Validate.py", "6 · Validate", "Model validation and maintenance",
     "reproducibility test, policy evaluation and fairness, drift monitoring and retraining"),
    ("pages/7_Deploy.py", "7 · Deploy", "Deployment and prediction",
     "deployment gate, scoring API, single recommendation, batch scoring of a customer file"),
]
for path, title, stage, text in steps:
    c1, c2, c3 = st.columns([1, 1.4, 4])
    with c1:
        st.page_link(path, label=title)
    c2.write(f"**{stage}**")
    c3.write(text)

if current:
    st.subheader("Currently deployed")
    st.write(f"Run `{current['run_id']}` deployed {current['deployed_at']} UTC")
    st.table({"offer": list(current["models"]), "model": list(current["models"].values())})

common.job_panel(title="Recent jobs")
