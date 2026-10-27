"""Privacy gate for datasets: only anonymised, masked or synthetic data may be
used. Values that look like personal data block a dataset; column names that
suggest personal data must be confirmed by the user before it is accepted."""
import re

PII_PATTERNS = {
    "email": r"[\w.+-]+@[\w-]+\.[\w.]+",
    "phone": r"(?:\+?263|\b0)7\d{8}\b",
    "national_id": r"\b\d{2}-?\d{6,7}-?[A-Z]-?\d{2}\b",
    "long_digit_run": r"\b\d{10,}\b",
}
SENSITIVE_NAME = re.compile(
    r"(^|_)(name|surname|firstname|first_name|lastname|last_name|fullname|full_name|email|e_mail|phone|"
    r"phone_number|mobile_number|msisdn|cell|national_id|id_number|idnumber|passport|account|account_number|"
    r"account_no|acc_no|iban|address|dob|date_of_birth)($|_)", re.IGNORECASE)


def scan(frame):
    """Counts of personal-data patterns per column, and columns whose names suggest personal data."""
    text = frame.select_dtypes(exclude="number").astype(str)
    matches = {}
    for col in text:
        hits = {name: int(text[col].str.contains(p, regex=True).sum()) for name, p in PII_PATTERNS.items()}
        hits = {k: v for k, v in hits.items() if v}
        if hits:
            matches[col] = hits
    sensitive = [c for c in frame.columns if SENSITIVE_NAME.search(str(c))]
    return {"pattern_matches": matches, "sensitive_columns": sensitive, "clean": not matches}


def pattern_totals(result):
    totals = dict.fromkeys(PII_PATTERNS, 0)
    for hits in result["pattern_matches"].values():
        for k, v in hits.items():
            totals[k] += v
    return totals
