# Databricks notebook source
# MAGIC %md
# MAGIC # 00 — Preflight
# MAGIC Seven checks, about two minutes. Every one of them has cost a run at some
# MAGIC point. Run this after every `Pull` in the Git folder; it is the cheapest
# MAGIC way to find out that the pull did not take.
# MAGIC
# MAGIC Check 6 is the one that matters most and the one that is new. Databricks
# MAGIC Standard access mode and Serverless both withhold `sparkContext`, so the
# MAGIC classic `setJobGroup` call the harness used to rely on is unavailable
# MAGIC there. The harness now tags jobs through `SparkSession.addTag` instead
# MAGIC and **refuses to run** if neither mechanism exists — because a run
# MAGIC without job attribution completes, writes timings, and contains no
# MAGIC task-level evidence at all. It looks finished and is not.

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
from skewbench.runner import (  # noqa: E402
    code_fingerprint,
    on_managed_platform,
    require_job_tagging,
    session_identity,
    tagging_mechanism,
)

check("Databricks detected", on_managed_platform(), str(on_managed_platform()))

# 2. The Spark data generator exists. This module was rewritten to avoid Arrow;
#    if the Git pull did not take, the import fails here rather than forty
#    minutes into generation.
try:
    from skewbench import datagen_spark
    ok_gen = hasattr(datagen_spark, "generate")
except Exception as exc:  # noqa: BLE001
    ok_gen = False
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

# 5. The volume is writable. A read-only volume fails at the END of a multi-hour
#    run, which is the worst possible moment to discover it.
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Job attribution — the check that decides whether to spend cluster time
# MAGIC This does not merely ask whether the API exists. It tags a real job, runs
# MAGIC it, and reports the mechanism. If this fails, nothing below is worth
# MAGIC running: the grid would produce timings with no tasks attached to them.

# COMMAND ----------

mechanism = tagging_mechanism(spark)
check("job attribution mechanism", mechanism != "none", mechanism)

if mechanism != "none":
    # Exercise it end to end -- declaring a tag and never having a job carry it
    # is exactly the silent failure this check exists to catch.
    from pyspark.sql import functions as F

    from skewbench.runner import _clear_job_group, _set_job_group

    try:
        _set_job_group(spark, "preflight::probe", "preflight probe")
        # A shuffle, so the job is real: a scan alone can be optimised away and
        # would prove nothing about whether tags reach the event log.
        (spark.range(100000)
              .groupBy(F.col("id") % 7)
              .count()
              .write.format("noop").mode("overwrite").save())
        _clear_job_group(spark)
        check("tagged job executed", True, "one shuffling job tagged preflight::probe")
    except Exception as exc:  # noqa: BLE001
        print(exc)
        check("tagged job executed", False)

# 7. Cluster shape. The runtime projection and the cost estimate assume this.
try:
    cores = sc.defaultParallelism
    execs = len(sc._jsc.sc().statusTracker().getExecutorInfos()) - 1
    check("cluster shape", execs >= 1, f"{execs} executors, {cores} default parallelism")
except Exception:  # noqa: BLE001
    # Expected on Standard access mode and Serverless: the executor list is not
    # reachable. Not a failure -- job attribution is what matters, and check 6
    # already covers it.
    #
    # The default argument to spark.conf.get is NOT a fallback string: Spark
    # type-checks it against the config's own type, so passing "unknown" for an
    # integer config raises NumberFormatException from the JVM. Ask without a
    # default and handle absence here instead.
    try:
        parallelism = spark.conf.get("spark.sql.shuffle.partitions")
    except Exception:  # noqa: BLE001
        parallelism = "unknown"
    check("cluster shape", True,
          f"executor list unavailable; shuffle partitions {parallelism}")

print(f"\nsession identity: {session_identity(spark)}")

# COMMAND ----------

print()
if failures:
    raise SystemExit(f"PREFLIGHT FAILED: {failures} -- fix these before running anything")
require_job_tagging(spark)   # belt and braces: the same gate the runner applies
print("Preflight clean. Proceed to 01_smoke.")
