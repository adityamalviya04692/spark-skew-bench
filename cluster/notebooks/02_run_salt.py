# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Salt grid (44 cells)
# MAGIC The file that tests the paper's one unconfirmed prediction: that the
# MAGIC optimal salt cardinality falls as 1/sqrt(P), and is smaller on a cluster
# MAGIC than on one node.
# MAGIC
# MAGIC **If this dies partway, just run it again.** Results are appended and
# MAGIC fsynced after every cell, and completed cells are skipped by `run_id`.
# MAGIC You lose at most one cell.
# MAGIC
# MAGIC **Do not run 03 until this finishes.** Two grids writing the same volume
# MAGIC from two sessions would let a wall-clock from one session be compared
# MAGIC against a wall-clock from another, which is exactly the error that once
# MAGIC inflated our noise floor from 2% to 29%.

# COMMAND ----------

# MAGIC %pip install PyYAML
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

import time

from skewbench.cli import load_config, specs_from_config
from skewbench.runner import run_grid

specs = specs_from_config(load_config(str(REPO / "config" / "cluster_salt.yaml")))
print(f"{len(specs)} cells")

started = time.time()
rows = run_grid(
    specs,
    data_root=DATA_ROOT,
    out_path=f"{RESULTS}/cluster_salt.jsonl",
    progress=True,
)
print(f"\n{len(rows)} rows in {(time.time()-started)/60:.1f} min")

# COMMAND ----------

missing = [r["arm_label"] for r in rows if not r.get("metrics_found")]
print("cells without metrics:", missing or "none")
print("distinct sessions:", {r.get("session_id") for r in rows})
print("distinct fingerprints:", {r.get("code_fingerprint") for r in rows})
