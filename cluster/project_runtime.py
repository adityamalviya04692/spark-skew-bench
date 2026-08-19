"""Project full-grid wall-clock from a completed smoke run.

The smoke config is the real grid shrunk on one axis only -- fact rows -- so a
per-cell time measured there scales to the real grid by the row ratio, plus a
one-off data-generation charge that the smoke run also measures. Everything
else (arms, partition count, cluster shape) is identical, which is what makes
the extrapolation worth anything.

The projection is deliberately pessimistic in one respect: shuffle cost grows
slightly faster than linearly in rows once spill starts, so the printed figure
is a floor, not a forecast. Treat a projection above four hours as a signal to
cut the grid, not as a schedule to keep.

    python cluster/project_runtime.py --smoke /Volumes/.../smoke.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skewbench.cli import load_config, specs_from_config  # noqa: E402

SMOKE_ROWS = 6_000_000
FULL_ROWS = 60_000_000
GRIDS = ("config/cluster_salt.yaml", "config/cluster_aqe.yaml")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", required=True, help="smoke.jsonl written by the smoke run")
    ap.add_argument("--repo", default=".", help="repo root, for reading the grid configs")
    ap.add_argument("--gen-minutes", type=float, default=0.0,
                    help="observed data-generation minutes in the smoke run, if known")
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.smoke).read_text().splitlines() if line.strip()]
    if not rows:
        print("smoke file is empty -- the smoke run did not complete")
        return 1

    # Median across cells of each cell's median repetition. The median twice is
    # not a typo: within a cell it rejects a slow first rep, across cells it
    # rejects an arm that happens to be unusually cheap or dear.
    per_cell = sorted(r["wall_median_s"] for r in rows)
    smoke_cell_s = per_cell[len(per_cell) // 2]
    scale = FULL_ROWS / SMOKE_ROWS
    full_cell_s = smoke_cell_s * scale

    reps = rows[0].get("repetitions", 5) + rows[0].get("warmup", 1)

    print(f"smoke cells measured        {len(rows)}")
    print(f"median cell wall time       {smoke_cell_s:.1f} s  ({SMOKE_ROWS/1e6:.0f}M rows)")
    print(f"row scale factor            {scale:.0f}x")
    print(f"projected cell wall time    {full_cell_s:.1f} s  ({FULL_ROWS/1e6:.0f}M rows)")
    print(f"repetitions charged per cell {reps} (incl. warmup)")
    print()

    total_h = 0.0
    # Generation is charged once per DISTINCT workload across both grids, not
    # once per grid. The two grids share two workloads, and the second grid to
    # run finds that data already materialised on the volume.
    seen_workloads: set = set()
    for grid in GRIDS:
        path = Path(args.repo) / grid
        if not path.exists():
            print(f"{grid}: MISSING")
            continue
        specs = specs_from_config(load_config(path))
        slugs = {s.workload.slug for s in specs}
        fresh = slugs - seen_workloads
        seen_workloads |= slugs
        hours = len(specs) * full_cell_s / 3600.0
        gen_h = args.gen_minutes * scale * len(fresh) / 60.0
        total_h += hours + gen_h
        print(f"{grid}")
        print(f"   cells {len(specs):3d}   workloads {len(slugs)} ({len(fresh)} new)")
        print(f"   measurement {hours:.1f} h   data generation {gen_h:.1f} h")

    print()
    print(f"PROJECTED TOTAL             {total_h:.1f} hours  (floor, not a forecast)")
    if total_h > 4:
        print()
        print("Above four hours. Cut cost before running, in this order:")
        print("  1. repetitions: 5 -> 3 in both configs      (saves ~33%)")
        print("  2. drop k: 64 from salt_selective           (saves ~9% of the salt grid)")
        print("  3. n_fact: 60000000 -> 30000000 everywhere  (saves ~50%, weakens the")
        print("     network-cost argument but does not invalidate it)")
        print("Do NOT cut the baseline arm or the aqe_coalesce_enabled: false profile.")
        print("Those two are controls; without them the run answers nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
