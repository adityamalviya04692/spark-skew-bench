"""Why does OptimizeSkewedJoin not fire on an obviously skewed join?

Sweeping the two documented trigger conditions (absolute threshold and median
factor) to their most permissive settings did not make the rule fire. This probe
tests a third, far less discussed condition: a skewed partition is split into
pieces of roughly `advisoryPartitionSizeInBytes`, so if the hot partition is
already *smaller* than that advisory size, there is no split to make and the
optimisation is a no-op no matter how skewed the partition is relative to its
peers.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skewbench import arms as arms_mod                     # noqa: E402
from skewbench import datagen                              # noqa: E402
from skewbench.config import ArmSpec, SparkSpec, WorkloadSpec  # noqa: E402
from skewbench.metrics import parse_event_log              # noqa: E402
from skewbench.runner import build_session                 # noqa: E402

# (shuffle partitions, advisory size, threshold, factor)
SETTINGS = [
    (16,  "16MB",  "8MB",   5),   # the main grid's configuration
    (16,  "1MB",   "1MB",   5),   # advisory below the hot partition
    (64,  "1MB",   "1MB",   5),   # more partitions, smaller advisory
    (200, "256KB", "256KB", 5),   # production-like partition count
    (200, "256KB", "256KB", 2),
]
REPS = 3


def main(data_root: str = "data") -> None:
    workload = WorkloadSpec(n_fact=2_000_000, n_dim=20_000, n_engines=2_000,
                            theta=1.2, hot_dim_multiplier=1, hot_keys=1,
                            n_sensors=12, seed=20260818)
    manifest = datagen.generate(workload, data_root)
    log_dir = Path("results/eventlogs/advisory")
    spark = build_session(
        SparkSpec(master="local[2]", driver_memory="4g", aqe_enabled=True),
        log_dir, "skewbench-advisory")

    rows = []
    try:
        fact = spark.read.parquet(manifest["fact_path"])
        dim = spark.read.parquet(manifest["dim_path"])
        for parts, advisory, threshold, factor in SETTINGS:
            spec = SparkSpec(master="local[2]", driver_memory="4g",
                             shuffle_partitions=parts, aqe_enabled=True,
                             autobroadcast_threshold=-1,
                             advisory_partition_size=advisory,
                             skewed_partition_threshold=threshold,
                             skewed_partition_factor=factor)
            arms_mod.apply_spark_conf(spark, spec)
            arm = ArmSpec("aqe")
            # Clear the job group before the plan capture. explain() executes
            # an action, and without this it inherits the PREVIOUS iteration's
            # rep tag -- so that group ends up containing two runs under two
            # different Spark configurations, and the metrics attributed to it
            # are a union of both.
            spark.sparkContext.setLocalProperty("spark.jobGroup.id", None)
            plan = arms_mod.explain(fact, dim, arm, manifest["hot_keys"])
            evidence = arms_mod.aqe_evidence(plan)

            group = f"adv-p{parts}-{advisory}-{threshold}-f{factor}"
            built = arms_mod.build(fact, dim, arm, manifest["hot_keys"])
            spark.sparkContext.setJobGroup(f"{group}::warm", "warmup")
            built.write.format("noop").mode("overwrite").save()
            times = []
            for rep in range(REPS):
                spark.sparkContext.setJobGroup(f"{group}::rep{rep}", "timed")
                started = time.perf_counter()
                built.write.format("noop").mode("overwrite").save()
                times.append(time.perf_counter() - started)

            rows.append({
                "partitions": parts, "advisory": advisory,
                "threshold": threshold, "factor": factor,
                "skew_join_applied": evidence["skew_join_applied"],
                "aqe_read_modes": sorted(set(
                    line.strip() for line in plan.splitlines()
                    if "AQEShuffleRead" in line)),
                "wall_median_s": round(statistics.median(times), 3),
                "group": group,
            })
            print(json.dumps({k: v for k, v in rows[-1].items()
                              if k != "aqe_read_modes"}), flush=True)
            Path(f"results/analysis/plan_{group}.txt").write_text(plan)
    finally:
        spark.stop()
        time.sleep(1)

    # Sorted by modification time: the directory can hold several logs with
    # overlapping group names, and an unsorted glob makes the result depend on
    # filesystem enumeration order.
    by_group = {}
    for log in sorted((p for p in log_dir.glob("*") if p.is_file()),
                      key=lambda p: p.stat().st_mtime):
        by_group.update(parse_event_log(log))
    for row in rows:
        reps = [m for g, m in by_group.items() if g.startswith(row["group"] + "::rep")]
        if reps:
            row["join_skew_ratio"] = round(
                statistics.median(m.join_skew_ratio for m in reps), 2)
            row["join_n_tasks"] = statistics.median(m.join_n_tasks for m in reps)

    Path("results/analysis/advisory_probe.json").write_text(json.dumps(rows, indent=2))
    print("\n=== advisory-size probe ===")
    print(f"{'parts':>6} {'advisory':>9} {'thresh':>8} {'f':>2}  "
          f"{'skewJoin':>8} {'wall_s':>7} {'skew_ratio':>10} {'tasks':>6}")
    for row in rows:
        print(f"{row['partitions']:>6} {row['advisory']:>9} {row['threshold']:>8} "
              f"{row['factor']:>2}  {str(row['skew_join_applied']):>8} "
              f"{row['wall_median_s']:>7.2f} "
              f"{row.get('join_skew_ratio', float('nan')):>10.2f} "
              f"{row.get('join_n_tasks', '?'):>6}")


if __name__ == "__main__":
    main(*sys.argv[1:])
