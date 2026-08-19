"""Re-derive metrics from saved event logs, without re-running any experiment.

This is the payoff of storing raw Spark event logs rather than only derived
numbers. When the metrics layer gains a field -- as it did when the analytical
partition-size estimates turned out to be wrong and the measured ones were
needed -- the existing runs can be re-analysed rather than repeated. Nothing
about the experiment changes; only what we compute from it.

Usage:
    python analysis/reparse.py results/v3.jsonl results/eventlogs/v3
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skewbench.metrics import RunMetrics, parse_event_log  # noqa: E402
from skewbench.runner import METRIC_FIELDS as NUMERIC_FIELDS  # noqa: E402


def main(results_path: str, log_dir: str, out_path: str | None = None) -> None:
    rows: List[Dict[str, Any]] = [
        json.loads(line) for line in open(results_path, encoding="utf-8")
        if line.strip()
    ]

    by_group: Dict[str, RunMetrics] = {}
    logs = sorted((p for p in Path(log_dir).glob("*") if p.is_file()),
                  key=lambda p: p.stat().st_mtime)
    for log in logs:
        by_group.update(parse_event_log(log))
    print(f"parsed {len(logs)} event log(s), {len(by_group)} job groups")

    missing = 0
    for row in rows:
        reps = [by_group[g] for g in row.get("group_ids", []) if g in by_group]
        if not reps:
            missing += 1
            row["metrics_found"] = False
            continue
        row["metrics_found"] = True
        row["n_reps_measured"] = len(reps)
        for field in NUMERIC_FIELDS:
            row[field] = statistics.median(getattr(m, field) for m in reps)
        row["shuffle_read_mb"] = round(row["shuffle_read_bytes"] / 1e6, 3)
        row["shuffle_write_mb"] = round(row["shuffle_write_bytes"] / 1e6, 3)
        row["spill_mb"] = round(
            (row["memory_spilled"] + row["disk_spilled"]) / 1e6, 3)
        row["join_hot_partition_mb"] = round(
            row["join_shuffle_read_max_bytes"] / 1e6, 3)
        row["join_median_partition_mb"] = round(
            row["join_shuffle_read_median_bytes"] / 1e6, 3)

    target = Path(out_path or results_path)
    with open(target, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")
    print(f"rewrote {len(rows)} rows -> {target}"
          + (f" ({missing} without metrics)" if missing else ""))


if __name__ == "__main__":
    main(*sys.argv[1:])
