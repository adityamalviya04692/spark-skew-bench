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

started = time.time()
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
# MAGIC `metrics_found` false on any row means the event-log parser found nothing.
# MAGIC The timings would still be written, and the paper's task-level claims --
# MAGIC straggler index, shuffle read skew, whether AQE actually fired -- would
# MAGIC all be missing. A run in that state looks complete and is not.

# COMMAND ----------

missing = [r["arm_label"] for r in rows if not r.get("metrics_found")]
print("cells without metrics:", missing or "none")
for r in rows:
    print(f"  {r['arm_label']:24s} median {r['wall_median_s']:7.2f}s  "
          f"straggler {r.get('straggler_index')}  metrics {r.get('metrics_found')}")
if missing:
    raise SystemExit("Event-log metrics are missing. Do not start the full run.")

# COMMAND ----------

# MAGIC %md
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
