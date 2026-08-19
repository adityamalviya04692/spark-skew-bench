"""Does AQE's skew-join rule fire, and does it matter when it does?

The main grid found the rule never firing, because the hot shuffle partition sat
below the absolute byte threshold. That is a claim about a *non-event*, which is
exactly the kind of claim that deserves a positive control: if we lower the
threshold until the rule does fire, the plan should say so and the straggler
should shrink. If nothing changes, our explanation was wrong.
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
from skewbench.config import ArmSpec, RunSpec, SparkSpec, WorkloadSpec  # noqa: E402
from skewbench.metrics import parse_event_log              # noqa: E402
from skewbench.runner import build_session                 # noqa: E402

# Spark's trigger is conjunctive: a partition is skewed only if its size
# exceeds BOTH the absolute threshold AND `skewedPartitionFactor` times the
# median partition size. Sweeping the threshold alone leaves the second
# condition binding, so we sweep the pair.
SETTINGS = [
    ("64MB", 5), ("8MB", 5), ("1MB", 5),
    ("1MB", 3), ("1MB", 2), ("256KB", 2), ("256KB", 1),
]
REPS = 3


def main(data_root: str = "data") -> None:
    workload = WorkloadSpec(n_fact=2_000_000, n_dim=20_000, n_engines=2_000,
                            theta=1.2, hot_dim_multiplier=32, hot_keys=1,
                            n_sensors=12, seed=20260818)
    manifest = datagen.generate(workload, data_root)
    log_dir = Path("results/eventlogs/threshold")
    base = SparkSpec(master="local[2]", driver_memory="4g",
                     shuffle_partitions=16, aqe_enabled=True,
                     autobroadcast_threshold=-1)
    spark = build_session(base, log_dir, "skewbench-threshold")

    rows = []
    try:
        fact = spark.read.parquet(manifest["fact_path"])
        dim = spark.read.parquet(manifest["dim_path"])
        for threshold, factor in SETTINGS:
            spec = SparkSpec(master="local[2]", driver_memory="4g",
                             shuffle_partitions=16, aqe_enabled=True,
                             autobroadcast_threshold=-1,
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

            group = f"thr-{threshold}-f{factor}"
            built = arms_mod.build(fact, dim, arm, manifest["hot_keys"])
            times = []
            spark.sparkContext.setJobGroup(f"{group}::warm", "warmup")
            built.write.format("noop").mode("overwrite").save()
            for rep in range(REPS):
                spark.sparkContext.setJobGroup(f"{group}::rep{rep}", "timed")
                started = time.perf_counter()
                built.write.format("noop").mode("overwrite").save()
                times.append(time.perf_counter() - started)

            rows.append({"threshold": threshold, "factor": factor,
                         "skew_join_applied": evidence["skew_join_applied"],
                         "coalesced": evidence["coalesced"],
                         "is_final_plan": evidence["is_final_plan"],
                         "wall_median_s": round(statistics.median(times), 3),
                         "group": group})
            print(json.dumps(rows[-1]), flush=True)
            Path(f"results/analysis/plan_threshold_{threshold}_f{factor}.txt").write_text(plan)
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
                statistics.median(m.join_skew_ratio for m in reps), 3)
            row["join_n_tasks"] = statistics.median(m.join_n_tasks for m in reps)
            row["shuffle_read_mb"] = round(statistics.median(
                m.shuffle_read_bytes for m in reps) / 1e6, 2)

    Path("results/analysis/threshold_sweep.json").write_text(json.dumps(rows, indent=2))
    print("\n=== threshold sweep ===")
    for row in rows:
        print(f"{row['threshold']:>7} x{row['factor']}  skewJoin={str(row['skew_join_applied']):<5} "
              f"wall={row['wall_median_s']:>6.2f}s  "
              f"skew_ratio={row.get('join_skew_ratio', float('nan')):>7.2f}  "
              f"tasks={row.get('join_n_tasks', '?')}")


if __name__ == "__main__":
    main(*sys.argv[1:])
