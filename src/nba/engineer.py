"""Feature engineering as a recorded recipe: feature creation (combining
features), transformations (log), binning, binarisation and splitting the offer
date. The recipe is saved with the dataset and with every training run, and the
same function applies it in training, reproduction, evaluation and the scoring
API, so a model always receives features computed the same way.

Steps that must be fitted to the data (imputation, scaling, one-hot encoding,
PCA) are not here: they live in the model pipeline, fitted on training data
only (see models.build)."""
import re

import numpy as np
import pandas as pd

OPS = {
    "combine": "Feature creation: combine two numbers",
    "log": "Log transform: log(1 + x), for skewed amounts",
    "bin": "Binning: cut a number into ranges",
    "binarize": "Binarisation: 1 above a threshold, else 0",
    "date_part": "Feature split: a part of the offer date",
}
HOW = {"add": "+", "subtract": "−", "multiply": "×", "divide": "÷"}
DATE_PARTS = ["month", "quarter", "day_of_week", "weekend"]
NAME = re.compile(r"^[a-z][a-z0-9_]{1,39}$")


def kind(step):
    return "categorical" if step["op"] == "bin" else "numeric"


def sources(step):
    if step["op"] == "combine":
        return [step["a"], step["b"]]
    if step["op"] == "date_part":
        return ["offer_date"]
    return [step["column"]]


def describe(step):
    op = step["op"]
    if op == "combine":
        return f"{step['a']} {HOW[step['how']]} {step['b']}"
    if op == "log":
        return f"log(1 + {step['column']})"
    if op == "bin":
        return f"{step['column']} in ranges split at {', '.join(f'{e:g}' for e in step['edges'])}"
    if op == "binarize":
        return f"1 if {step['column']} > {step['threshold']:g} else 0"
    return f"{step['part'].replace('_', ' ')} of the offer date"


def bin_labels(edges):
    return ([f"< {edges[0]:g}"] + [f"{a:g} to < {b:g}" for a, b in zip(edges, edges[1:])]
            + [f">= {edges[-1]:g}"])


def apply(frame, recipe):
    """The frame with each engineered feature (re)computed, in recipe order."""
    if not recipe:
        return frame
    frame = frame.copy()
    for s in recipe:
        op = s["op"]
        if op == "date_part":
            d = pd.to_datetime(frame["offer_date"])
            v = {"month": d.dt.month, "quarter": d.dt.quarter, "day_of_week": d.dt.dayofweek,
                 "weekend": (d.dt.dayofweek >= 5).astype(int)}[s["part"]].astype(float)
        elif op == "combine":
            a = pd.to_numeric(frame[s["a"]], errors="coerce")
            b = pd.to_numeric(frame[s["b"]], errors="coerce")
            v = {"add": a + b, "subtract": a - b, "multiply": a * b,
                 "divide": a / b.where(b != 0)}[s["how"]]  # division by zero -> missing
        else:
            x = pd.to_numeric(frame[s["column"]], errors="coerce")
            if op == "log":
                v = np.log1p(x.where(x > -1))  # undefined at or below -1 -> missing
            elif op == "binarize":
                v = (x > s["threshold"]).astype(float).where(x.notna())
            else:
                edges = list(s["edges"])
                v = pd.cut(x, [-np.inf] + edges + [np.inf], right=False, labels=bin_labels(edges)).astype(object)
        frame[s["name"]] = v
    return frame


def raw_inputs(features, recipe):
    """The columns a caller must supply to score: the model's features, with each
    engineered one replaced by the columns it is computed from (the offer date
    defaults to the day of scoring)."""
    by_name = {s["name"]: s for s in recipe or []}
    out = []

    def add(col):
        if col in by_name:
            for src in sources(by_name[col]):
                add(src)
        elif col != "offer_date" and col not in out:
            out.append(col)

    for f in features:
        add(f)
    return out


def quantile_edges(series, n):
    """n roughly equal-sized bins: the inner cut points, rounded and de-duplicated."""
    qs = pd.to_numeric(series, errors="coerce").quantile(np.linspace(0, 1, n + 1)[1:-1]).to_numpy()
    edges = sorted({float(f"{q:.4g}") for q in qs if np.isfinite(q)})
    return edges


def check(step, frame, recipe):
    """Problems with a new step, as readable messages."""
    problems = []
    existing = set(frame.columns) | {s["name"] for s in recipe}
    if not NAME.match(step.get("name", "")):
        problems.append("name: lower case letters, digits and _, starting with a letter (2-40 characters)")
    elif step["name"] in existing:
        problems.append(f"a column called {step['name']!r} already exists")
    for col in sources(step):
        if col == "offer_date":
            continue
        if col not in frame:
            problems.append(f"no column {col!r}")
        elif not pd.api.types.is_numeric_dtype(frame[col]):
            problems.append(f"{col} is not numeric")
    if step["op"] == "bin" and not step.get("edges"):
        problems.append("give at least one cut point")
    return problems


def dependents(name, recipe):
    return [s["name"] for s in recipe if name in sources(s)]
