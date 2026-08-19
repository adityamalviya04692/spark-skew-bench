# Databricks notebook source
# MAGIC %md
# MAGIC # 00 — Preflight
# MAGIC Six checks, about ninety seconds. Every one of them has cost us a run at
# MAGIC some point. Run this after every `Pull` in the Git folder; it is the
# MAGIC cheapest way to find out that the pull did not take.

# COMMAND ----------

# MAGIC %pip install PyYAML
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

failures = []


def check(label, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {label}   {detail}")
    if not ok:
        failures.append(label)


# 1. Platform detection. If this is False the harness will try to set
#    spark.eventLog.* itself, fight Databricks' own logging, and collect no
#    metrics at all -- the run completes and the science is missing.
from skewbench.runner import on_managed_platform, code_fingerprint
check("Databricks detected", on_managed_platform(), str(on_managed_platform()))

# 2. The Spark data generator exists. This module is the one that was rewritten
#    to avoid Arrow; if the Git pull did not take, the import fails here rather
#    than forty minutes into generation.
try:
    from skewbench import datagen_spark
    ok_gen = hasattr(datagen_spark, "generate")
except Exception as exc:  # noqa: BLE001
    ok_gen = False
    datagen_spark = None
    print(exc)
check("datagen_spark importable", ok_gen)

# 3. Code fingerprint. Record it. Any results file whose fingerprint differs
#    from this one was produced by different code and cannot be pooled with it.
fp = code_fingerprint()
check("code fingerprint", bool(fp), fp)

# 4. The configs this run needs are present.
for name in ("cluster_smoke", "cluster_salt", "cluster_aqe"):
    p = REPO / "config" / f"{name}.yaml"
    check(f"config/{name}.yaml", p.exists())

# 5. The volume is writable. A read-only volume fails at the END of a three
#    hour run, which is the worst possible time to find out.
probe = f"{RESULTS}/.preflight_probe"
try:
    with open(probe, "w") as fh:
        fh.write("ok")
    os.remove(probe)
    ok_vol = True
except Exception as exc:  # noqa: BLE001
    ok_vol = False
    print(exc)
check("volume writable", ok_vol, RESULTS)

# 6. Cluster shape. The projection and the cost estimate both assume this.
cores = sc.defaultParallelism
execs = len([e for e in sc._jsc.sc().statusTracker().getExecutorInfos()]) - 1
check("executors online", execs >= 1, f"{execs} executors, {cores} default parallelism")

# COMMAND ----------

print()
if failures:
    raise SystemExit(f"PREFLIGHT FAILED: {failures} -- fix these before running anything")
print("Preflight clean. Proceed to 01_smoke.")
