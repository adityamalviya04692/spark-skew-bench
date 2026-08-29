# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Collect and verify before you shut down
# MAGIC Run this while the cluster is still alive. It packages everything needed
# MAGIC to re-analyse the run without re-running it, and it checks the four things
# MAGIC that have gone wrong before.

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

# MAGIC %md
# MAGIC ## Attach metrics first
# MAGIC Neither grid notebook attaches metrics, and it is not their fault: on a
# MAGIC managed platform the event logs do not exist yet when a grid finishes.
# MAGIC Databricks delivers them asynchronously. So the reparse belongs here, at
# MAGIC the end, once. Without it every `with_metrics` check below fails on a run
# MAGIC that was in fact perfectly good.

# COMMAND ----------

import glob
import json
import os
import subprocess
import sys
import time

cluster_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterId", "")
LOG_ROOT = f"{VOLUME}/logs/{cluster_id}"
print(f"log root  {LOG_ROOT}")

# Only logs newer than the newest result row count. Previous sessions leave
# their event logs in this destination forever, so a bare existence check
# returns instantly with stale files and the reparse then attributes nothing --
# a failure that looks exactly like broken attribution and is not.
newest_run_ms = 0
for name in ("cluster_salt", "cluster_aqe"):
    target = f"{RESULTS}/{name}.jsonl"
    if not os.path.exists(target):
        continue
    for line in open(target):
        if not line.strip():
            continue
        row = json.loads(line)
        for span in row.get("rep_windows", []):
            newest_run_ms = max(newest_run_ms, int(span[1]))
cutoff = newest_run_ms / 1000 if newest_run_ms else 0


def delivered(newer_than=0.0):
    return [f for f in glob.glob(f"{LOG_ROOT}/**/*", recursive=True)
            if os.path.isfile(f) and "eventlog" in f
            and not os.path.basename(f).startswith(".")
            and os.path.getmtime(f) >= newer_than]


print(f"logs already present from earlier runs: {len(delivered())}")
deadline = time.time() + 600
found = []
while time.time() < deadline:
    found = delivered(cutoff)
    if found:
        break
    print(f"  waiting for THIS run's event log "
          f"({int(deadline - time.time())}s left)...", flush=True)
    time.sleep(30)
print(f"event log files covering this run: {len(found)}")

for name in ("cluster_salt", "cluster_aqe"):
    target = f"{RESULTS}/{name}.jsonl"
    if not os.path.exists(target):
        print(f"{name}: results file absent, skipping reparse")
        continue
    print(f"\n--- reparse {name} ---")
    out = subprocess.run(
        [sys.executable, str(REPO / "analysis" / "reparse.py"), target, LOG_ROOT],
        capture_output=True, text=True)
    print(out.stdout or "")
    print(out.stderr or "")

# COMMAND ----------

import json
import shutil
from pathlib import Path

REPORT = {}
for name in ("cluster_salt", "cluster_aqe"):
    p = Path(f"{RESULTS}/{name}.jsonl")
    if not p.exists():
        REPORT[name] = "MISSING"
        continue
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    REPORT[name] = {
        "rows": len(rows),
        "with_metrics": sum(1 for r in rows if r.get("metrics_found")),
        "sessions": sorted({r.get("session_id") for r in rows}),
        "fingerprints": sorted({r.get("code_fingerprint") for r in rows}),
        "arms": sorted({r["arm_label"] for r in rows}),
    }
print(json.dumps(REPORT, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read these four lines, in this order
# MAGIC 1. **`rows` == cells expected** (44 and 10). Fewer means cells were skipped.
# MAGIC 2. **`with_metrics` == `rows`.** Anything less and part of the paper's
# MAGIC    task-level evidence does not exist for those cells.
# MAGIC 3. **`sessions` has exactly one entry per file.** More than one means the
# MAGIC    file was assembled from separate Spark sessions, and wall-clock
# MAGIC    comparisons across it are not valid.
# MAGIC 4. **`fingerprints` has exactly one entry.** More than one means the code
# MAGIC    changed mid-run and the rows are not pooled results of one program.

# COMMAND ----------

for name, info in REPORT.items():
    if info == "MISSING":
        print(f"{name}: MISSING")
        continue
    problems = []
    if info["with_metrics"] != info["rows"]:
        problems.append("some cells have no metrics")
    if len(info["sessions"]) != 1:
        problems.append("multiple Spark sessions in one file")
    if len(info["fingerprints"]) != 1:
        problems.append("multiple code fingerprints in one file")
    print(f"{name}: {'OK' if not problems else '; '.join(problems)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bundle everything for download
# MAGIC The event logs matter as much as the JSONL. With them the run can be
# MAGIC re-parsed for a metric we did not think to record; without them a new
# MAGIC question means a new three hours of cluster time.

# COMMAND ----------

bundle = f"{RESULTS}/cluster_bundle"
shutil.rmtree(bundle, ignore_errors=True)
Path(bundle).mkdir(parents=True, exist_ok=True)

for pattern in ("cluster_salt.jsonl", "cluster_aqe.jsonl", "smoke.jsonl",
                "cluster_salt.control.json", "cluster_aqe.control.json",
                "cluster_salt.summary.json", "cluster_aqe.summary.json"):
    src = Path(RESULTS) / pattern
    if src.exists():
        shutil.copy2(src, Path(bundle) / pattern)
        print("bundled", pattern)

for logs in Path(RESULTS).glob("eventlogs*"):
    shutil.copytree(logs, Path(bundle) / logs.name, dirs_exist_ok=True)
    print("bundled", logs.name)

shutil.make_archive(f"{RESULTS}/cluster_bundle", "zip", bundle)
size_mb = Path(f"{RESULTS}/cluster_bundle.zip").stat().st_size / 1e6
print(f"\n{RESULTS}/cluster_bundle.zip   {size_mb:.1f} MB")
print("Download this from Catalog -> your volume -> right-click -> Download.")
