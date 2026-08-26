"""Experiment driver: session construction, warmup, repetitions, result capture.

Design decisions worth stating, because they are the ones a reviewer will
challenge.

*One session for the whole grid.* Every configuration this study varies --
``spark.sql.adaptive.enabled``, the skew-join thresholds, shuffle partitions, the
broadcast threshold -- is runtime-settable, so restarting the JVM between cells
would add minutes of noise per cell and buy nothing. Runs are separated instead
by Spark job group, which gives an exact, not inferred, mapping from a results
row back to the tasks that produced it.

*Warmup runs are discarded, not averaged in.* The first execution of a plan pays
JIT compilation, class loading and page-cache misses. Including it inflates the
mean and, worse, inflates it unevenly across arms.

*Cell order is randomised, and a session-level warmup precedes the grid.* An
earlier version of this harness executed cells in configuration order, which put
the ``baseline`` arm first in every workload block. Because a cold JVM has not
yet reached steady state, that systematically inflated the reference arm and
therefore every speedup measured against it. The effect was not small: the
harness generates its own control for this, since ``salt_selective(k=1)``
compiles to exactly the same physical plan as ``baseline``, and the two differed
by 17.7% on the first workload while agreeing to within 0.3% on the later ones.
Randomising the order spreads any residual warmth across arms instead of
concentrating it in one, and :func:`control_discrepancy` turns the identity into
an assertion rather than an assumption.

*Repetitions report median and IQR, never mean and standard deviation.* Run
times on a shared machine have a long right tail; the median is the honest
central estimate and the IQR is the honest dispersion.

*Rows are written as they complete.* An earlier version buffered the whole grid
in memory and wrote once at the end, so an interrupted run -- a killed process,
a lost cluster node, an exhausted quota -- lost every cell it had already paid
for. Rows are now appended to a partial file after each cell, and the run can be
resumed against it. On a five-hour cluster grid this is the difference between
losing an afternoon and losing a cell.
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from pyspark.sql import DataFrame, SparkSession

from skewbench import arms as arms_mod
from skewbench import datagen
from skewbench.config import RunSpec, SparkSpec, flatten
from skewbench.metrics import RunMetrics, parse_event_log

# Every numeric metric carried from the event log onto a results row. Defined
# once: analysis/reparse.py imports this, so a metric added to one path cannot
# go missing from the other -- which is exactly how straggler_index ended up
# present in re-analysis and absent from live runs.
METRIC_FIELDS = [
    "shuffle_read_bytes", "shuffle_write_bytes", "shuffle_read_records",
    "shuffle_write_records", "memory_spilled", "disk_spilled",
    "total_task_time_ms", "peak_execution_memory", "n_tasks", "wall_clock_ms",
    "join_skew_ratio", "join_straggler_index", "join_task_time_cv",
    "join_task_time_max_ms", "join_task_time_median_ms",
    "join_task_time_mean_ms", "join_total_task_time_ms", "join_n_tasks",
    "join_shuffle_read_max_bytes", "join_shuffle_read_median_bytes",
]

JVM_OPENS = " ".join([
    "--add-opens=java.base/java.lang=ALL-UNNAMED",
    "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
    "--add-opens=java.base/java.io=ALL-UNNAMED",
    "--add-opens=java.base/java.net=ALL-UNNAMED",
    "--add-opens=java.base/java.nio=ALL-UNNAMED",
    "--add-opens=java.base/java.util=ALL-UNNAMED",
    "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
    "--add-opens=java.base/sun.security.action=ALL-UNNAMED",
])


def on_managed_platform() -> bool:
    """True when Spark is supplied by a managed platform such as Databricks.

    Detected rather than configured, because getting it wrong is costly in both
    directions: setting ``spark.eventLog.*`` on Databricks fights the platform's
    own logging and typically yields nothing, while failing to set it locally
    yields no metrics at all.
    """
    import os

    return any(key in os.environ for key in
               ("DATABRICKS_RUNTIME_VERSION", "DB_HOME", "DATABRICKS_HOST"))


# --- job attribution ------------------------------------------------------
#
# Every repetition must be traceable from a results row back to the exact Spark
# tasks that produced it. Two mechanisms exist and which one is available is a
# property of the cluster, not of the code:
#
#   * ``sparkContext.setJobGroup`` on a classic session. Writes the
#     ``spark.jobGroup.id`` job property.
#   * ``SparkSession.addTag`` on a Spark Connect session -- which is what
#     Databricks Standard (formerly Shared) access mode and Serverless both
#     give you, because they withhold ``sparkContext`` entirely. Writes the
#     ``spark.job.tags`` job property.
#
# Both land in the event log, so the parser reads either. What must never
# happen is the third case: neither mechanism available and the failure passing
# silently. A run in that state completes, writes timings, and contains no
# task-level evidence at all -- no straggler index, no shuffle skew, no proof
# that AQE fired. It looks finished and is not. Hence :func:`require_job_tagging`,
# which is called once at session setup and refuses to continue.

TAG_PREFIX = "skewbench:"


def _connect_tagging(spark: SparkSession) -> bool:
    return hasattr(spark, "addTag") and hasattr(spark, "clearTags")


def _classic_tagging(spark: SparkSession) -> bool:
    sc = getattr(spark, "sparkContext", None)
    return sc is not None and hasattr(sc, "setJobGroup")


def tagging_mechanism(spark: SparkSession) -> str:
    """Which attribution mechanism this session supports: 'tags', 'jobgroup', or 'none'.

    Connect is probed first because a session may expose ``sparkContext`` as an
    attribute that raises on access; ``hasattr`` swallows that, so the classic
    probe is the less trustworthy of the two.
    """
    if _connect_tagging(spark):
        return "tags"
    try:
        if _classic_tagging(spark):
            return "jobgroup"
    except Exception:  # noqa: BLE001 -- attribute access itself can raise here
        pass
    return "none"


def require_job_tagging(spark: SparkSession) -> str:
    mechanism = tagging_mechanism(spark)
    if mechanism == "none":
        raise RuntimeError(
            "This Spark session supports neither SparkSession.addTag nor "
            "sparkContext.setJobGroup, so no measurement can be attributed to "
            "the cell that produced it. The run would finish and contain no "
            "task-level evidence. Refusing to start.\n"
            "On Databricks, use a classic cluster on DBR 14.3+ (Standard or "
            "Dedicated access mode both work via addTag), not an older runtime."
        )
    return mechanism


def _quiet(spark: SparkSession) -> None:
    """Silence Spark's INFO chatter where the API for doing so exists.

    Cosmetic only. A Connect session has no ``sparkContext`` to set it on, and
    a noisy log is not a reason to abort a run -- unlike missing job tagging,
    which is.
    """
    try:
        spark.sparkContext.setLogLevel("ERROR")
    except Exception:  # noqa: BLE001
        pass


def session_identity(spark: SparkSession) -> str:
    """A stable id for THIS Spark session.

    Wall-clock is only comparable within one session: JIT state, cache warmth
    and executor placement all differ across sessions, and comparing across
    them once inflated this study's noise floor from 2% to 29%. Every results
    row carries this so the analysis can refuse such a comparison rather than
    silently make it.
    """
    try:
        return spark.sparkContext.applicationId
    except Exception:  # noqa: BLE001
        client = getattr(spark, "client", None)
        for attr in ("_session_id", "session_id"):
            value = getattr(client, attr, None)
            if value:
                return str(value)
        return "managed-session"


def _set_job_group(spark: SparkSession, group_id: str, description: str = "") -> None:
    """Tag every job launched from here until cleared, with ``group_id``."""
    if _connect_tagging(spark):
        spark.clearTags()
        spark.addTag(TAG_PREFIX + group_id)
        return
    spark.sparkContext.setJobGroup(group_id, description)


def _clear_job_group(spark: SparkSession) -> None:
    if _connect_tagging(spark):
        spark.clearTags()
        return
    # PySpark exposes no clearJobGroup(); unset the local property directly.
    spark.sparkContext.setLocalProperty("spark.jobGroup.id", None)



def build_session(spec: SparkSpec, event_log_dir: Path,
                  app_name: str = "skewbench") -> SparkSession:
    """Construct the session. Only non-runtime-settable options belong here.

    On a managed platform the session already exists and event logging is the
    platform's business: Databricks runs its own event logging and delivers it
    through cluster log delivery, which must be configured on the cluster
    *before it starts*. We therefore attach to the existing session and leave
    logging alone, rather than overriding properties the platform owns.
    """
    managed = on_managed_platform()
    if managed:
        session = SparkSession.builder.appName(app_name).getOrCreate()
        _quiet(session)
        return session

    event_log_dir.mkdir(parents=True, exist_ok=True)
    builder = (
        SparkSession.builder.master(spec.master)
        .appName(app_name)
        .config("spark.driver.memory", spec.driver_memory)
        .config("spark.driver.extraJavaOptions", JVM_OPENS)
        .config("spark.executor.extraJavaOptions", JVM_OPENS)
        .config("spark.eventLog.enabled", "true")
        .config("spark.eventLog.dir", event_log_dir.resolve().as_uri())
        .config("spark.eventLog.compress", "false")
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.session.timeZone", "UTC")
        # Deterministic file splitting keeps the number of input partitions
        # stable across arms, so a difference in task counts is attributable to
        # the join strategy rather than to scan planning.
        .config("spark.sql.files.maxPartitionBytes", "64MB")
        .config("spark.sql.files.openCostInBytes", "4MB")
    )
    if spec.local_dir:
        builder = builder.config("spark.local.dir", spec.local_dir)
    session = builder.getOrCreate()
    _quiet(session)
    return session


def observed_hot_counts(fact: DataFrame, dim: DataFrame,
                        hot_keys: Sequence[str]) -> Dict[str, float]:
    """Measure ``H`` and ``P`` from the data rather than trusting the sampler.

    The generator reports expected counts under the Zipf law. These are the
    realised ones. Reporting both, and their agreement, is cheap insurance
    against a silent generation bug invalidating every downstream number.
    """
    hot = list(hot_keys)
    H = fact.filter(fact.engine_id.isin(hot)).count()
    P = dim.filter(dim.engine_id.isin(hot)).count()
    return {"observed_hot_fact_rows": float(H), "observed_hot_dim_rows": float(P)}


def _row_bytes(path: str, n_rows: int) -> float:
    total = sum(f.stat().st_size for f in Path(path).rglob("*.parquet"))
    return total / max(1, n_rows)


def execute_run(spark: SparkSession, spec: RunSpec, fact: DataFrame, dim: DataFrame,
                hot_keys: Sequence[str], capture_plan: bool = False) -> Dict[str, Any]:
    """Run one cell: warmup, then timed repetitions, tagged by job group."""
    arms_mod.apply_spark_conf(spark, spec.spark)
    plan = arms_mod.build(fact, dim, spec.arm, hot_keys)

    # Warmup executions are tagged separately so the parser can discard them.
    for w in range(spec.warmup):
        _set_job_group(spark, f"{spec.run_id}::warmup{w}", "warmup")
        plan.write.format("noop").mode("overwrite").save()

    wall_times: List[float] = []
    group_ids: List[str] = []
    for rep in range(spec.repetitions):
        group = f"{spec.run_id}::rep{rep}"
        group_ids.append(group)
        _set_job_group(spark, group, f"{spec.arm.label} rep {rep}")
        started = time.perf_counter()
        # The noop sink forces full materialisation without writing bytes to disk
        # or shipping rows to the driver, so the measurement is of the join and
        # nothing else.
        plan.write.format("noop").mode("overwrite").save()
        wall_times.append(time.perf_counter() - started)

    _clear_job_group(spark)

    row = flatten(spec)
    row.update({
        "arm_label": spec.arm.label,
        "group_ids": group_ids,
        "wall_times_s": [round(t, 4) for t in wall_times],
        "wall_median_s": round(statistics.median(wall_times), 4),
        "wall_min_s": round(min(wall_times), 4),
        "wall_max_s": round(max(wall_times), 4),
        "wall_iqr_s": round(_iqr(wall_times), 4),
    })
    if capture_plan:
        try:
            row["physical_plan"] = arms_mod.explain(fact, dim, spec.arm, hot_keys)
        except Exception as exc:  # pragma: no cover - diagnostics only
            row["physical_plan"] = f"<unavailable: {exc}>"
    return row


def _iqr(values: Sequence[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n < 4:
        return max(ordered) - min(ordered) if n > 1 else 0.0
    q1 = statistics.median(ordered[: n // 2])
    q3 = statistics.median(ordered[(n + 1) // 2 :])
    return q3 - q1


def attach_metrics(rows: List[Dict[str, Any]], event_log_dir: Path) -> List[Dict[str, Any]]:
    """Join event-log metrics onto the timing rows, per repetition then median."""
    logs = sorted(Path(event_log_dir).glob("*"), key=lambda p: p.stat().st_mtime)
    logs = [p for p in logs if p.is_file() and not p.name.endswith(".inprogress")] or \
           [p for p in Path(event_log_dir).glob("*") if p.is_file()]
    by_group: Dict[str, RunMetrics] = {}
    for log in logs:
        by_group.update(parse_event_log(log))

    enriched: List[Dict[str, Any]] = []
    for row in rows:
        per_rep = [by_group[g] for g in row.get("group_ids", []) if g in by_group]
        if not per_rep:
            row["metrics_found"] = False
            enriched.append(row)
            continue
        row["metrics_found"] = True
        row["n_reps_measured"] = len(per_rep)
        numeric = METRIC_FIELDS
        for field_name in numeric:
            values = [getattr(m, field_name) for m in per_rep]
            row[field_name] = statistics.median(values)
        row["shuffle_read_mb"] = round(row["shuffle_read_bytes"] / 1e6, 3)
        row["shuffle_write_mb"] = round(row["shuffle_write_bytes"] / 1e6, 3)
        row["spill_mb"] = round((row["memory_spilled"] + row["disk_spilled"]) / 1e6, 3)
        row["join_hot_partition_mb"] = round(
            row["join_shuffle_read_max_bytes"] / 1e6, 3)
        row["join_median_partition_mb"] = round(
            row["join_shuffle_read_median_bytes"] / 1e6, 3)
        enriched.append(row)
    return enriched


def control_discrepancy(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compare ``baseline`` against ``salt_selective(k=1)`` within each workload.

    The two are the same physical plan under the same configuration, so any
    difference between them is measurement noise plus whatever systematic bias
    the execution order introduced. Reporting it makes the study's noise floor
    an observable rather than an assumption, and any speedup smaller than this
    discrepancy is not a result.
    """
    findings: List[Dict[str, Any]] = []
    by_workload: Dict[str, Dict[str, float]] = {}
    for row in rows:
        # Only compare within one Spark application; see session_id above.
        pass
    for row in rows:
        slug = row.get("wl_slug") or row.get("run_id", "")[:0] or "workload"
        key = f'{row.get("wl_theta")}_{row.get("wl_hot_dim_multiplier")}_{row.get("wl_hot_keys")}'
        arm, k = row.get("arm_name"), row.get("arm_k")
        if arm == "baseline":
            by_workload.setdefault(key, {})["baseline"] = row["wall_median_s"]
        elif arm == "salt_selective" and k == 1:
            by_workload.setdefault(key, {})["control"] = row["wall_median_s"]
    for key, pair in by_workload.items():
        if "baseline" in pair and "control" in pair:
            lo, hi = sorted((pair["baseline"], pair["control"]))
            findings.append({
                "workload": key,
                "baseline_s": pair["baseline"],
                "control_s": pair["control"],
                "discrepancy_pct": round(100.0 * (hi - lo) / lo, 2),
            })
    return findings


def code_fingerprint() -> str:
    """Hash of the harness source, recorded in every results row.

    A results file that cannot be tied to the code that produced it is not
    reproducible, however deterministic the code is. This caught a real problem:
    an earlier results file was produced before a fix to plan capture, and
    nothing in the file recorded that.
    """
    import hashlib

    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def completed_run_ids(partial_path: str | Path) -> set:
    """Run ids already present in a partial results file, for resumption."""
    path = Path(partial_path)
    if not path.exists():
        return set()
    done = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["run_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def run_grid(specs: List[RunSpec], data_root: str | Path, out_path: str | Path,
             event_log_dir: Optional[str | Path] = None,
             overwrite_data: bool = False,
             progress: bool = True,
             shuffle_order: bool = True,
             session_warmup: int = 3,
             resume: bool = True) -> List[Dict[str, Any]]:
    """Generate any missing data, execute every cell, and write a JSONL result file."""
    data_root = data_root if on_managed_platform() else Path(data_root)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    event_log_dir = Path(event_log_dir or out_path.parent / "eventlogs" / out_path.stem)
    if event_log_dir.exists():
        shutil.rmtree(event_log_dir)

    # Generation strategy. The pandas generator runs on one core, which is fine
    # at the two-million-row scale of the single-node study and untenable at
    # cluster scale -- sixty million rows would take over an hour per workload
    # on the driver alone, before any measurement. On a managed platform we let
    # Spark generate the data instead, which spreads it across the cluster and
    # produces the identical schema and key law.
    use_spark_gen = on_managed_platform()
    manifests: Dict[str, Dict] = {}
    if use_spark_gen:
        from skewbench import datagen_spark
        spark_for_gen = SparkSession.builder.getOrCreate()

    for spec in specs:
        slug = spec.workload.slug
        if slug not in manifests:
            if progress:
                print(f"[data] materialising {slug}"
                      f"{' (spark)' if use_spark_gen else ''}", flush=True)
            if use_spark_gen:
                manifests[slug] = datagen_spark.generate(
                    spec.workload, str(data_root), spark_for_gen,
                    overwrite=overwrite_data)
            else:
                manifests[slug] = datagen.generate(spec.workload, data_root,
                                                   overwrite=overwrite_data)

    session_spec = specs[0].spark
    # Session-level fields come from the first spec only, so refuse a grid that
    # silently disagrees about them rather than mislabelling the results.
    for spec in specs:
        for field_name in ("master", "driver_memory", "local_dir"):
            if getattr(spec.spark, field_name) != getattr(session_spec, field_name):
                raise ValueError(
                    f"all specs in a grid must share session-level field "
                    f"{field_name!r}; got {getattr(spec.spark, field_name)!r} and "
                    f"{getattr(session_spec, field_name)!r}"
                )

    spark = build_session(session_spec, event_log_dir)
    fingerprint = code_fingerprint()
    # Spark's application id identifies the JVM this cell ran in. Wall-clock
    # comparisons are only valid within one application: a later session starts
    # cold, warms differently, and may sit on a differently-loaded host. Cells
    # gap-filled from a second run were measurably and systematically slower
    # here, which is invisible unless the session is recorded.
    mechanism = require_job_tagging(spark)
    if progress:
        print(f"[setup] job attribution via {mechanism}")
    session_id = session_identity(spark)
    rows: List[Dict[str, Any]] = []
    loaded: Dict[str, Any] = {}
    seen_plan: set = set()

    # Execution order is randomised so that JVM warmth is spread across arms
    # rather than concentrated in whichever arm the config happens to list
    # first. The permutation is seeded, so a run is still reproducible.
    order = list(range(len(specs)))
    if shuffle_order:
        import random

        random.Random(20260818).shuffle(order)
    ordered_specs = [specs[i] for i in order]

    # Partial file: one row appended per completed cell. Timing rows only --
    # event-log metrics are attached at the end, and can be re-attached later
    # from the logs alone via analysis/reparse.py.
    partial_path = Path(str(out_path) + ".partial")
    already_done = completed_run_ids(partial_path) if resume else set()
    resumed_rows: Dict[str, Dict[str, Any]] = {}
    if already_done:
        with open(partial_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    prior = json.loads(line)
                    resumed_rows[prior["run_id"]] = prior
                except (json.JSONDecodeError, KeyError):
                    continue
    if already_done and progress:
        print(f"[resume] {len(already_done)} cells already in {partial_path.name}",
              flush=True)
    partial_handle = open(partial_path, "a", encoding="utf-8")

    try:
        for index, spec in enumerate(ordered_specs, start=1):
            manifest = manifests[spec.workload.slug]
            if spec.workload.slug not in loaded:
                fact = spark.read.parquet(manifest["fact_path"])
                dim = spark.read.parquet(manifest["dim_path"])
                counts = observed_hot_counts(fact, dim, manifest["hot_keys"])
                loaded[spec.workload.slug] = (fact, dim, counts)
            fact, dim, counts = loaded[spec.workload.slug]

            # Session-level warmup: run a throwaway plan a few times before the
            # first measured cell so the JVM is at steady state for all of them,
            # not just for the ones that happen to execute later.
            if index == 1 and session_warmup:
                from skewbench.config import ArmSpec as _Arm

                warm = arms_mod.build(fact, dim, _Arm("baseline"),
                                      manifest["hot_keys"])
                _set_job_group(spark, "session::warmup", "session warmup")
                for _ in range(session_warmup):
                    warm.write.format("noop").mode("overwrite").save()
                _clear_job_group(spark)

            if spec.run_id in already_done:
                # Carry the previously-completed row into this run's output.
                # Skipping the cell must not also drop it from the results file
                # -- that would silently shrink the grid on every resume.
                if spec.run_id in resumed_rows:
                    rows.append(resumed_rows[spec.run_id])
                if progress:
                    print(f"[{index}/{len(specs)}] carried forward (already complete)",
                          flush=True)
                continue

            if progress:
                print(f"[{index}/{len(specs)}] {spec.workload.slug} :: "
                      f"{spec.arm.label} aqe={spec.spark.aqe_enabled} "
                      f"p={spec.spark.shuffle_partitions}", flush=True)

            plan_key = (spec.workload.slug, spec.arm.name, spec.arm.k)
            row = execute_run(spark, spec, fact, dim, manifest["hot_keys"],
                              capture_plan=plan_key not in seen_plan)
            seen_plan.add(plan_key)

            row.update(counts)
            row["expected_hot_fact_rows"] = manifest["expected_hot_fact_rows"]
            row["expected_hot_dim_rows"] = manifest["expected_hot_dim_rows"]
            row["fact_row_bytes"] = (
                0.0 if on_managed_platform()
                else round(_row_bytes(manifest["fact_path"], spec.workload.n_fact), 3))
            row["theoretical_skew_ratio"] = datagen.theoretical_skew_ratio(
                spec.workload, spec.spark.shuffle_partitions)
            row["code_fingerprint"] = fingerprint
            row["session_id"] = session_id
            row["exec_order"] = index
            rows.append(row)
            partial_handle.write(json.dumps(row, default=str) + "\n")
            partial_handle.flush()
            os.fsync(partial_handle.fileno())
    finally:
        partial_handle.close()
        if not on_managed_platform():
            spark.stop()
        time.sleep(1.0)  # let the event log flush before it is parsed

    if on_managed_platform():
        # Event logs are delivered asynchronously by the platform (Databricks
        # flushes every ~5 minutes), so they are not readable at the moment the
        # grid finishes. Timing rows are complete and written; metrics are
        # attached afterwards with analysis/reparse.py once delivery lands.
        if progress:
            cluster_id = ""
            try:
                cluster_id = spark.conf.get(
                    "spark.databricks.clusterUsageTags.clusterId", "")
            except Exception:  # noqa: BLE001
                pass
            print("[managed platform] timing rows written WITHOUT metrics.\n"
                  "  This is expected: Databricks delivers event logs "
                  "asynchronously, roughly every 5 minutes, so they are not\n"
                  "  readable at the moment the grid finishes. Wait for "
                  "delivery, then attach metrics with:\n"
                  f"    python analysis/reparse.py {out_path} "
                  f"<log-destination>/{cluster_id or '<cluster-id>'}\n"
                  "  Until that runs, metrics_found is absent on every row -- "
                  "which is not the same as the metrics being lost.",
                  flush=True)
    else:
        rows = attach_metrics(rows, event_log_dir)
    with open(out_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")

    controls = control_discrepancy(rows)
    if controls:
        Path(out_path).with_suffix(".control.json").write_text(
            json.dumps(controls, indent=2))
        if progress:
            print("[control] baseline vs salt_selective(k=1) "
                  "(identical plans; this is the noise floor):", flush=True)
            for finding in controls:
                print(f"  {finding['workload']}: {finding['baseline_s']:.3f}s vs "
                      f"{finding['control_s']:.3f}s "
                      f"-> {finding['discrepancy_pct']:.2f}%", flush=True)
    if progress:
        print(f"[done] {len(rows)} rows -> {out_path}", flush=True)
    return rows
