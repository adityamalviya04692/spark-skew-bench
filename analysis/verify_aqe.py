"""Verify what AQE actually did, per workload.

The paper's central claim is that AQE's benefit decays as the probe side
thickens because split-and-replicate degenerates. That claim is only meaningful
if we know whether the skew-join rule fired at all -- an AQE speedup that comes
entirely from partition coalescing is a different phenomenon wearing the same
name. This script executes the AQE arm on each workload and reports the final
adaptive plan's evidence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skewbench import arms as arms_mod          # noqa: E402
from skewbench.cli import load_config, specs_from_config  # noqa: E402
from skewbench.runner import build_session      # noqa: E402
from skewbench import datagen                   # noqa: E402


def main(config: str = "config/pilot.yaml", data_root: str = "data") -> None:
    specs = specs_from_config(load_config(config))
    aqe_specs, seen = [], set()
    for spec in specs:
        key = spec.workload.slug
        if spec.arm.name == "aqe" and key not in seen:
            seen.add(key)
            aqe_specs.append(spec)

    spark = build_session(aqe_specs[0].spark,
                          Path("results/eventlogs/verify"), "skewbench-verify")
    findings = []
    try:
        for spec in aqe_specs:
            manifest = datagen.generate(spec.workload, data_root)
            fact = spark.read.parquet(manifest["fact_path"])
            dim = spark.read.parquet(manifest["dim_path"])
            arms_mod.apply_spark_conf(spark, spec.spark)
            plan = arms_mod.explain(fact, dim, spec.arm, manifest["hot_keys"])
            evidence = arms_mod.aqe_evidence(plan)
            evidence["workload"] = spec.workload.slug
            evidence["hot_dim_multiplier"] = spec.workload.hot_dim_multiplier
            findings.append(evidence)
            print(json.dumps(evidence, indent=2), flush=True)
            Path(f"results/analysis/plan_aqe_m"
                 f"{spec.workload.hot_dim_multiplier:g}.txt").write_text(plan)
    finally:
        spark.stop()

    Path("results/analysis/aqe_evidence.json").write_text(
        json.dumps(findings, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
