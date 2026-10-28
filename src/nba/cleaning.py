"""Data cleaning as a recorded recipe: the portal saves the recipe and the log
of what each step removed or changed with the dataset, so the cleaned data can
be explained and regenerated from the original file."""
import pandas as pd

from nba.features import FALSE_VALUES, TRUE_VALUES

DEFAULT = {
    "trim_text": True,           # strip spaces, blanks -> missing, one spelling per case variant
    "drop_incomplete": True,     # rows without a readable key, date, offer or outcome
    "date_from": None,           # keep offers on or after this date (YYYY-MM-DD)
    "date_to": None,             # ... and on or before this date
    "drop_duplicates": True,     # exact duplicate rows
    "max_missing_share": None,   # drop features missing in more than this share of rows
    "fill_numeric": "model",     # model (median, fitted on training data) | median | drop
    "fill_categorical": "model",  # model ("missing" category) | mode | drop
    "outliers": "none",          # none | cap | drop
    "outlier_columns": [],
    "outlier_quantiles": [0.01, 0.99],
    "rare_category_min": 0,      # categories with fewer rows become "other"
    "drop_thin_offers": False,   # offers with fewer than 10 acceptances or 10 rejections
}
FILL_NUMERIC = ["model", "median", "drop"]
FILL_CATEGORICAL = ["model", "mode", "drop"]
OUTLIERS = ["none", "cap", "drop"]


def quality(frame, schema):
    """Per-column profile to decide what to clean: missing share, distinct
    values and text spelled several ways."""
    rows = []
    for c in frame.columns:
        s = frame[c]
        text = s.dropna().astype(str)
        variants = 0
        if not pd.api.types.is_numeric_dtype(s):
            norm = text.str.strip().str.lower()
            variants = int(text.nunique() - norm.nunique())
        rows.append({"column": c, "role": role_of(c, schema), "type": "number" if pd.api.types.is_numeric_dtype(s) else "text",
                     "missing": float(s.isna().mean()), "distinct": int(s.nunique()), "extra spellings": variants})
    return pd.DataFrame(rows), int(frame.duplicated().sum())


def role_of(column, schema):
    for role, c in schema["columns"].items():
        if c == column:
            return role.replace("_", " ")
    if column in schema["categorical"]:
        return "categorical feature"
    if column in schema["numeric"]:
        return "numeric feature"
    if column in schema.get("protected", []):
        return "protected"
    if column == schema.get("eligibility_column"):
        return "eligibility"
    return "not used"


def _outcome_readable(s):
    if pd.api.types.is_numeric_dtype(s):
        return s.isin([0, 1])
    return s.astype(str).str.strip().str.lower().isin(TRUE_VALUES | FALSE_VALUES)


def clean(frame, schema, recipe):
    """Apply the recipe. Returns the cleaned frame, the schema (features dropped
    for missingness are removed) and a log of each step's effect."""
    recipe = {**DEFAULT, **recipe}
    frame = frame.copy()
    schema = {**schema, "categorical": list(schema["categorical"]), "numeric": list(schema["numeric"])}
    roles = schema["columns"]
    log = []

    def step(name, before, detail=""):
        log.append({"step": name, "rows_before": before, "rows_after": len(frame), "detail": detail})

    if recipe["trim_text"]:
        changed = 0
        for c in frame.columns:
            if pd.api.types.is_numeric_dtype(frame[c]) or c == roles.get("customer_key"):
                continue
            s = frame[c].astype("string").str.strip().replace("", pd.NA)
            # Case variants ("Mobile", "mobile ") take the most common spelling.
            counts = s.value_counts().rename_axis("value").reset_index()
            counts["key"] = counts.value.str.lower()
            canon = counts.drop_duplicates("key").set_index("key").value  # value_counts sorts by count
            s = s.str.lower().map(canon).astype(object).where(s.notna(), None)
            changed += int((s.fillna("\0") != frame[c].astype(object).fillna("\0")).sum())
            frame[c] = s
        step("trim and standardise text", len(frame), f"{changed:,} values changed")

    if recipe["drop_incomplete"]:
        before = len(frame)
        keep = pd.Series(True, index=frame.index)
        for role in ["customer_key", "offer_date", "action", "accepted"]:
            if roles.get(role):
                keep &= frame[roles[role]].notna()
        if roles.get("offer_date"):
            dates = pd.to_datetime(frame[roles["offer_date"]], format=schema.get("date_format"), errors="coerce")
            keep &= dates.notna()
        if roles.get("accepted"):
            keep &= _outcome_readable(frame[roles["accepted"]])
        frame = frame[keep]
        step("drop incomplete rows", before, "missing or unreadable key, date, offer or outcome")

    if (recipe["date_from"] or recipe["date_to"]) and roles.get("offer_date"):
        before = len(frame)
        dates = pd.to_datetime(frame[roles["offer_date"]], format=schema.get("date_format"), errors="coerce")
        keep = pd.Series(True, index=frame.index)
        if recipe["date_from"]:
            keep &= dates >= pd.Timestamp(recipe["date_from"])
        if recipe["date_to"]:
            keep &= dates <= pd.Timestamp(recipe["date_to"])
        frame = frame[keep]
        step("date range", before, f"{recipe['date_from'] or 'start'} to {recipe['date_to'] or 'end'}")

    if recipe["drop_duplicates"]:
        before = len(frame)
        frame = frame.drop_duplicates()
        step("drop duplicate rows", before)

    if recipe["max_missing_share"] is not None:
        share = frame[schema["categorical"] + schema["numeric"]].isna().mean()
        sparse = share[share > recipe["max_missing_share"]].index.tolist()
        schema["categorical"] = [c for c in schema["categorical"] if c not in sparse]
        schema["numeric"] = [c for c in schema["numeric"] if c not in sparse]
        frame = frame.drop(columns=sparse)
        step("drop sparse features", len(frame),
             f"more than {recipe['max_missing_share']:.0%} missing: {', '.join(sparse) or 'none'}")

    for kind, cols in [("numeric", schema["numeric"]), ("categorical", schema["categorical"])]:
        how = recipe[f"fill_{kind}"]
        if how == "model":
            continue
        before = len(frame)
        if how == "drop":
            frame = frame.dropna(subset=cols)
            step(f"drop rows with missing {kind} features", before)
        else:
            filled = int(frame[cols].isna().sum().sum())
            for c in cols:
                value = frame[c].median() if how == "median" else frame[c].mode().iloc[0]
                frame[c] = frame[c].fillna(value)
            step(f"fill missing {kind} features ({how})", before, f"{filled:,} values filled")

    cols = [c for c in recipe["outlier_columns"] if c in frame and pd.api.types.is_numeric_dtype(frame[c])]
    if recipe["outliers"] != "none" and cols:
        before = len(frame)
        lo_q, hi_q = recipe["outlier_quantiles"]
        bounds = {c: (frame[c].quantile(lo_q), frame[c].quantile(hi_q)) for c in cols}
        if recipe["outliers"] == "cap":
            capped = sum(int(((frame[c] < lo) | (frame[c] > hi)).sum()) for c, (lo, hi) in bounds.items())
            for c, (lo, hi) in bounds.items():
                frame[c] = frame[c].clip(lo, hi)
            step("cap outliers", before, f"{capped:,} values capped to the {lo_q:.0%}-{hi_q:.0%} quantiles")
        else:
            keep = pd.Series(True, index=frame.index)
            for c, (lo, hi) in bounds.items():
                keep &= frame[c].isna() | frame[c].between(lo, hi)
            frame = frame[keep]
            step("drop outlier rows", before, f"outside the {lo_q:.0%}-{hi_q:.0%} quantiles")

    if recipe["rare_category_min"]:
        grouped = []
        for c in schema["categorical"]:
            counts = frame[c].value_counts()
            rare = counts[counts < recipe["rare_category_min"]].index
            if len(rare):
                frame[c] = frame[c].where(~frame[c].isin(rare), "other")
                grouped.append(f"{c} ({len(rare)})")
        step("group rare categories", len(frame),
             f"fewer than {recipe['rare_category_min']} rows -> other: {', '.join(grouped) or 'none'}")

    if recipe["drop_thin_offers"] and roles.get("action") and roles.get("accepted"):
        from nba.features import to_binary
        before = len(frame)
        try:
            y = to_binary(frame[roles["accepted"]])
            counts = y.groupby(frame[roles["action"]]).agg(["sum", "size"])
            thin = counts[(counts["sum"] < 10) | (counts["size"] - counts["sum"] < 10)].index.tolist()
            frame = frame[~frame[roles["action"]].isin(thin)]
            step("drop thin offers", before, ", ".join(map(str, thin)) or "none")
        except ValueError as e:
            step("drop thin offers", before, f"skipped: {e}")

    return frame.reset_index(drop=True), schema, log
