"""Interactive EDA on the active dataset, plus the standard EDA report as a job."""
import streamlit as st

from nba import jobs
from nba.explore import explorer, load
from nba.portal import common

common.page("2 · Explore", show_title=False)
name = common.active_dataset("explore")
if name is None:
    st.stop()

with st.sidebar:
    st.divider()
    if st.button("Generate the standard EDA report"):
        out = "reports/eda" if name == "mydata" else f"logs/portal/eda/{name}"
        jobs.start("eda", ["--dataset", name, "--out", out], label=f"EDA report · {name}")
        st.success(f"Started: figures and tables go to {out}/ and to MLflow.")

params, data = load(name)
explorer(params, data, title=f"2 · Explore · {name}")
common.job_panel(match="eda", title="EDA report jobs")
