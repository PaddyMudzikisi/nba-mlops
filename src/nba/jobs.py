"""Background pipeline jobs for the portal: each job is `python -m nba.<step>`
run detached, with its log, exit code and metadata under logs/jobs/."""
import json
import os
import re
import secrets
import shlex
import subprocess
from datetime import datetime, timezone

from nba.config import REPO

JOBS = REPO / "logs" / "jobs"
STEPS = {"train", "reproduce", "evaluate", "monitor", "eda"}


def _paths(job_id):
    return {k: JOBS / f"{job_id}.{k}" for k in ("json", "log", "exit")}


def start(step, args, label=""):
    if step not in STEPS:
        raise ValueError(f"unknown step {step!r}")
    JOBS.mkdir(parents=True, exist_ok=True)
    job_id = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{step}-{secrets.token_hex(2)}"
    p = _paths(job_id)
    cmd = ["python", "-m", f"nba.{step}", *map(str, args)]
    shell = f"{shlex.join(cmd)} > {shlex.quote(str(p['log']))} 2>&1; echo $? > {shlex.quote(str(p['exit']))}"
    proc = subprocess.Popen(["sh", "-c", shell], cwd=REPO, env=os.environ.copy(), start_new_session=True)
    meta = {"id": job_id, "step": step, "args": list(map(str, args)), "label": label, "pid": proc.pid,
            "started": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    p["json"].write_text(json.dumps(meta))
    return job_id


def status(job_id):
    p = _paths(job_id)
    meta = json.loads(p["json"].read_text())
    if p["exit"].exists():
        code = int(p["exit"].read_text().strip() or 1)
        meta.update(state="succeeded" if code == 0 else "failed", exit_code=code)
    else:
        try:
            os.kill(meta["pid"], 0)
            meta["state"] = "running"
        except OSError:  # the portal restarted while the job ran
            meta["state"] = "lost"
    log = p["log"].read_text() if p["log"].exists() else ""
    meta["log_tail"] = "\n".join(log.splitlines()[-60:])
    m = re.search(r"Parent run id: (\w+)", log)
    meta["run_id"] = m.group(1) if m else None
    return meta


def recent(step=None, n=10):
    if not JOBS.exists():
        return []
    ids = sorted((f.stem for f in JOBS.glob("*.json")), reverse=True)
    ids = [i for i in ids if step is None or f"-{step}-" in i]
    return [status(i) for i in ids[:n]]
