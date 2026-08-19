"""One-command cluster runner for skewbench.

Runs the same harness, unchanged, against a cluster Spark session. The only
differences from a local run are that the session is attached to an existing
cluster rather than built with a local master, and that data and event logs live
on distributed storage.

Databricks:
    Upload the repo, attach a notebook or job to a cluster, then:
        %pip install -r requirements.txt
        %run ./cluster/run_cluster.py --config config/cluster_salt.yaml \
             --data-root dbfs:/tmp/skewbench/data \
             --out /dbfs/tmp/skewbench/cluster_salt.jsonl

EMR / self-managed YARN:
    spark-submit --deploy-mode client \
        --conf spark.eventLog.enabled=true \
        --conf spark.eventLog.dir=hdfs:///var/log/spark/skewbench \
        cluster/run_cluster.py --config config/cluster_salt.yaml \
        --data-root hdfs:///tmp/skewbench/data --out /tmp/cluster_salt.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skewbench.cli import load_config, specs_from_config  # noqa: E402
from skewbench.runner import run_grid                     # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/cluster_salt.yaml")
    parser.add_argument("--data-root", required=True,
                        help="distributed path, e.g. dbfs:/tmp/skewbench/data")
    parser.add_argument("--out", required=True,
                        help="LOCAL driver path for the results JSONL")
    parser.add_argument("--event-log-dir", default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    specs = specs_from_config(load_config(args.config))
    if args.limit:
        specs = specs[: args.limit]

    print(f"skewbench cluster run: {len(specs)} cells")
    print(f"  config     {args.config}")
    print(f"  data root  {args.data_root}")
    print(f"  results    {args.out}")
    print("Reminder: this run is only comparable to the single-node results if")
    print("the arms and k values match. Do not edit the arm list.")

    rows = run_grid(specs, args.data_root, args.out,
                    event_log_dir=args.event_log_dir, progress=True)

    Path(args.out).with_suffix(".summary.json").write_text(json.dumps({
        "n_cells": len(rows),
        "config": args.config,
        "code_fingerprint": rows[0].get("code_fingerprint") if rows else None,
        "cells_with_metrics": sum(1 for r in rows if r.get("metrics_found")),
    }, indent=2))
    print(f"\nDone. Send back:\n  {args.out}\n  {args.out}.control.json\n"
          f"  the event log directory (for re-analysis without re-running)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
