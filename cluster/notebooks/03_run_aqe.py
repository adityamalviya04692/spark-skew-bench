# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — AQE grid (10 cells)
# MAGIC Does the single-node AQE null survive at cluster scale? At 60M rows the
# MAGIC hot key carries 22.23% of the data, so the hot partition is roughly
# MAGIC 1.3-2.4 GB. That clears all three of `OptimizeSkewedJoin`'s conjunctive
# MAGIC preconditions by orders of magnitude -- the 256MB threshold, the 5x
# MAGIC median factor, and the 64MB advisory size that blocked the rule on the
# MAGIC laptop. If AQE is ever going to help, it helps here.
# MAGIC
# MAGIC This grid carries its own `baseline` arm so every AQE ratio is measured
# MAGIC against a baseline from the same Spark session.

# COMMAND ----------

# MAGIC %pip install PyYAML
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

import time

from skewbench.cli import load_config, specs_from_config
from skewbench.runner import run_grid

specs = specs_from_config(load_config(str(REPO / "config" / "cluster_aqe.yaml")))
print(f"{len(specs)} cells")

started = time.time()
rows = run_grid(
    specs,
    data_root=DATA_ROOT,
    out_path=f"{RESULTS}/cluster_aqe.jsonl",
    progress=True,
)
print(f"\n{len(rows)} rows in {(time.time()-started)/60:.1f} min")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Did the skew rule actually fire?
# MAGIC This is the question the whole grid exists to answer. A speedup with no
# MAGIC `AQEShuffleRead ... skewed` in the plan is coalescing, not skew handling,
# MAGIC and must not be reported as the latter.

# COMMAND ----------

for r in rows:
    if r["arm_name"] in ("aqe", "aqe_salt"):
        plan = r.get("physical_plan", "")
        print(f"{r['arm_label']:16s} coalesce={r['sp_aqe_coalesce_enabled']}  "
              f"m={r['wl_hot_dim_multiplier']:3}  median={r['wall_median_s']:7.2f}s  "
              f"skew_in_plan={'skew' in plan.lower()}")
