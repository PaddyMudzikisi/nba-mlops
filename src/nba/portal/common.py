"""Shared pieces of the portal pages: page setup, dataset selection, MLflow
lookups and the live job panel."""
import json

import matplotlib

matplotlib.use("Agg")
import mlflow
import pandas as pd
import streamlit as st

from nba import datasets, jobs
from nba.config import load_params
from nba.report import label  # noqa: F401  (also applies the house chart style)

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STATE_TEXT = {"running": "running", "succeeded": "succeeded", "failed": "FAILED", "lost": "lost (portal restarted)"}


def page(title, show_title=True):
    st.set_page_config(page_title=f"{title} · NBA portal", layout="wide")
    if show_title:
        st.title(title)


def experiment():
    return load_params()["mlflow"]["experiment"]


def active_dataset(page_key):
    """Dataset chosen in the sidebar, remembered across pages."""
    names = [d["name"] for d in datasets.list_datasets()]
    if not names:
        st.sidebar.warning("No datasets registered yet: upload one on the Data page.")
        return None
    current = st.session_state.get("active_dataset", "mydata" if "mydata" in names else names[0])
    if current not in names:
        current = names[0]
    name = st.sidebar.selectbox("Dataset", names, index=names.index(current), key=f"dataset-{page_key}")
    st.session_state["active_dataset"] = name
    return name


@st.cache_data(ttl=60)
def dataset_hashes():
    """Data fingerprint -> dataset name, to place runs recorded before runs carried a dataset tag."""
    from nba.config import REPO
    from nba.features import dataset_sha256
    return {dataset_sha256(REPO / d["path"]): d["name"] for d in datasets.list_datasets() if (REPO / d["path"]).exists()}


@st.cache_data(ttl=5)
def train_runs(dataset=None):
    """Parent training runs, newest first, with mean test metrics over their offers."""
    parents = mlflow.search_runs(experiment_names=[experiment()], filter_string="attributes.run_name = 'train'",
                                 order_by=["attributes.start_time DESC"], max_results=300)
    if parents.empty:
        return parents
    children = mlflow.search_runs(experiment_names=[experiment()], filter_string="tags.action LIKE '%'",
                                  max_results=5000)
    metric_cols = [c for c in ["metrics.roc_auc", "metrics.pr_auc", "metrics.f1", "metrics.train_roc_auc"]
                   if c in children]
    means = children.groupby("tags.mlflow.parentRunId")[metric_cols].mean() if not children.empty else None
    out = pd.DataFrame({
        "run_id": parents.run_id,
        "started": parents.start_time.dt.tz_convert(None).dt.strftime("%Y-%m-%d %H:%M"),
        "dataset": parents.get("tags.dataset", pd.Series(index=parents.index, dtype=object)).fillna(
            parents.get("tags.data_sha256", pd.Series(index=parents.index, dtype=object)).map(dataset_hashes())
        ).fillna("unregistered data"),
        "model": parents.get("tags.model_type", pd.Series(index=parents.index, dtype=object)).fillna("xgboost"),
        "split": parents.get("tags.split_method", pd.Series(index=parents.index, dtype=object)).fillna("time"),
        "committed": parents.get("tags.git_dirty", pd.Series(index=parents.index, dtype=object)).map(
            {"False": "yes", "True": "no"}).fillna("?"),
        "status": parents.status,
    })
    if means is not None:
        out = out.join(means.rename(columns=lambda c: "mean_" + c.split(".")[1]), on="run_id")
    if dataset:
        out = out[out.dataset == dataset]
    return out.reset_index(drop=True)


@st.cache_data(ttl=5)
def child_runs(run_id):
    return mlflow.search_runs(experiment_names=[experiment()],
                              filter_string=f"tags.mlflow.parentRunId = '{run_id}'")


@st.cache_data(ttl=5)
def latest_related(step_tag, run_id):
    """Latest reproduce / evaluate / monitor run for a training run (tag reproduces/evaluates/monitors)."""
    runs = mlflow.search_runs(experiment_names=[experiment()], filter_string=f"tags.{step_tag} = '{run_id}'",
                              order_by=["attributes.start_time DESC"], max_results=1)
    return None if runs.empty else runs.iloc[0]


def load_artifact_json(run_id, path):
    try:
        return mlflow.artifacts.load_dict(f"runs:/{run_id}/{path}")
    except Exception:  # noqa: BLE001  (missing artifact)
        return None


def load_artifact_yaml(run_id, path):
    import tempfile

    import yaml
    try:
        with tempfile.TemporaryDirectory() as tmp:
            local = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=path, dst_path=tmp)
            return yaml.safe_load(open(local))
    except Exception:  # noqa: BLE001  (missing artifact)
        return None


def run_label(row):
    return f"{row['started']} · {row['model']} · {row['split']} split · {row['run_id'][:8]}"


def pick_run(runs, key, help_text=None):
    if runs.empty:
        st.info("No training runs for this dataset yet: start one on the Train page.")
        return None
    options = runs.run_id.tolist()
    labels = {row["run_id"]: run_label(row) for row in runs.to_dict("records")}
    return st.selectbox("Training run", options, format_func=lambda i: labels[i], key=key, help=help_text)


@st.fragment(run_every=3)
def job_panel(match=None, title="Jobs"):
    """Live status and log of recent pipeline jobs; `match` filters by step or argument."""
    recent = [j for j in jobs.recent(n=30)
              if match is None or match == j["step"] or match in j["args"]][:6]
    if not recent:
        return
    st.subheader(title)
    for j in recent:
        heading = f"{j['step']} · {j['label'] or j['id']} · {STATE_TEXT[j['state']]}"
        with st.expander(heading, expanded=j["state"] == "running"):
            st.caption(f"started {j['started']} UTC · job {j['id']}"
                       + (f" · run {j['run_id']}" if j.get("run_id") else ""))
            st.code(j["log_tail"] or "(starting...)", language=None)


def decisions(n=20, path="logs/decisions.jsonl"):
    from nba.config import REPO
    f = REPO / path
    if not f.exists():
        return []
    lines = f.read_text().splitlines()[-n:]
    return [json.loads(line) for line in reversed(lines)]
