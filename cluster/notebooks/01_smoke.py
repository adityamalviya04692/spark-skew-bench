# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Smoke test
# MAGIC Three cells at one tenth scale. Its job is not to produce science; it is
# MAGIC to prove the generator, the arms, the event-log parser and the writer all
# MAGIC work on this cluster, and to measure a per-cell time we can extrapolate
# MAGIC from. Ten minutes here has repeatedly saved three hours later.

# COMMAND ----------

# MAGIC %pip install PyYAML
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

import time

from skewbench.cli import load_config, specs_from_config
from skewbench.runner import run_grid

specs = specs_from_config(load_config(str(REPO / "config" / "cluster_smoke.yaml")))
print(f"{len(specs)} smoke cells")

# Recorded before the grid so the wait below can tell this run's event log
# apart from every previous run's, which stay in the destination forever.
RUN_STARTED = time.time()
started = RUN_STARTED
rows = run_grid(
    specs,
    data_root=DATA_ROOT,
    out_path=f"{RESULTS}/smoke.jsonl",
    progress=True,
)
elapsed_min = (time.time() - started) / 60
print(f"\nsmoke rows: {len(rows)}  in {elapsed_min:.1f} min")

# COMMAND ----------

# MAGIC %md
# MAGIC ## The only check that matters
# MAGIC On a managed platform the runner deliberately does NOT attach metrics
# MAGIC inline: Databricks delivers event logs asynchronously, roughly every
# MAGIC five minutes, so nothing is readable at the moment the grid finishes.
# MAGIC `metrics_found` being absent right now is expected and means nothing.
# MAGIC
# MAGIC What has to be proved is the next link in the chain, and it is the one
# MAGIC still unproven: preflight showed a job tag gets **set**, but not that it
# MAGIC survives into the **event log**, which is what the parser reads. The
# MAGIC cell below waits for delivery, runs the reparse, and checks. If tags do
# MAGIC not reach the log, this is where it shows up -- not in the grid run.

# COMMAND ----------

import glob
import os
import subprocess
import time

# Where the platform delivers this cluster's logs. Derived, not pasted: the
# cluster id is the folder name and getting it wrong looks exactly like
# "delivery never happened".
cluster_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterId", "")
LOG_ROOT = f"{VOLUME}/logs/{cluster_id}"
print(f"cluster id   {cluster_id}")
print(f"log root     {LOG_ROOT}")


def event_log_files(root: str, newer_than: float = 0.0):
    """Delivered event logs, optionally only ones written after a cutoff.

    The cutoff is what makes this correct. Previous sessions leave their event
    logs in the same destination forever, so a bare existence check returns
    immediately with stale files -- and the reparse then attributes nothing,
    because those jobs ran days before this run's windows. That failure looks
    identical to attribution being broken, and cost a full smoke cycle to tell
    apart.
    """
    out = []
    for f in glob.glob(f"{root}/**/*", recursive=True):
        if not (os.path.isfile(f) and "eventlog" in f):
            continue
        if os.path.basename(f).startswith("."):
            continue
        if os.path.getmtime(f) < newer_than:
            continue
        out.append(f)
    return out


stale = event_log_files(LOG_ROOT)
print(f"event log files already present (from earlier runs): {len(stale)}")

# Poll rather than sleep a flat five minutes: delivery is usually faster, and
# when it is not, a fixed sleep just hides the wait.
deadline = time.time() + 600
found = []
while time.time() < deadline:
    found = event_log_files(LOG_ROOT, newer_than=RUN_STARTED)
    if found:
        break
    print(f"  waiting for THIS run's event log "
          f"({int(deadline - time.time())}s left)...", flush=True)
    time.sleep(30)

print(f"event log files delivered SINCE THIS RUN STARTED: {len(found)}")
if not found:
    raise SystemExit(
        "No event logs delivered after 7 minutes. Check that cluster log "
        "delivery is configured (Compute -> Edit -> Advanced -> Logging) and "
        "that it was set BEFORE the cluster started -- it cannot be added "
        "retroactively to a running cluster."
    )

# COMMAND ----------

SMOKE = f"{RESULTS}/smoke.jsonl"
out = subprocess.run(
    [sys.executable, str(REPO / "analysis" / "reparse.py"), SMOKE, LOG_ROOT],
    capture_output=True, text=True)
print(out.stdout or "")
print(out.stderr or "")

# COMMAND ----------

import json

rows = [json.loads(l) for l in open(SMOKE) if l.strip()]
missing = [r["arm_label"] for r in rows if not r.get("metrics_found")]
print("cells without metrics:", missing or "none")
for r in rows:
    print(f"  {r['arm_label']:24s} median {r['wall_median_s']:7.2f}s  "
          f"straggler {r.get('join_straggler_index')}  "
          f"tasks {r.get('n_tasks')}  metrics {r.get('metrics_found')}")
if missing:
    raise SystemExit(
        "Event-log metrics are STILL missing after reparse. The job tags are "
        "not reaching the event log, so the grid would produce timings with no "
        "tasks attached. Do not start the full run."
    )
print("\nJob tags reach the event log. Metrics attach correctly.")

# COMMAND ----------

# MAGIC ## Projection
# MAGIC Feed the smoke timings to the projector. Read the total before you start
# MAGIC the real grid, not after.

# COMMAND ----------

# MAGIC %sh
# MAGIC # gen-minutes is the data-generation time the cell above printed under
# MAGIC # "[gen]"; pass it so the projection includes generation, not just joins.
# MAGIC echo "run the next Python cell instead -- it knows the paths"

# COMMAND ----------

import subprocess

print(subprocess.run(
    [sys.executable, str(REPO / "cluster" / "project_runtime.py"),
     "--smoke", f"{RESULTS}/smoke.jsonl", "--repo", str(REPO),
     "--gen-minutes", "2"],
    capture_output=True, text=True).stdout)
