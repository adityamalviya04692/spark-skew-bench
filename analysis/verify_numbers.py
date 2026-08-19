"""Recompute the paper's headline numbers from raw results and compare.

`analysis/analyze.py` generates `paper/numbers.tex`, and the manuscript cites
only those macros -- so in principle the paper cannot disagree with the data.
In practice that argument is circular: it assumes the generator is correct. This
script closes the loop by recomputing each headline figure from the results
file by an independent path, and failing if any disagrees.

It is deliberately written not to import the analysis code. Sharing a helper
with the thing under test would let one bug produce two matching wrong answers.

Usage:
    python analysis/verify_numbers.py results/v3_single.jsonl paper/numbers.tex
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

TAGS = {"Thin": 1, "Mid": 10, "Thick": 32}


def load_rows(path: str | Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def load_macros(path: str | Path) -> Dict[str, str]:
    text = Path(path).read_text()
    return dict(re.findall(r"\\newcommand\{\\([A-Za-z0-9]+)\}\{([^}]*)\}", text))


def pick(rows, mult, arm, k=None, coalesce=None) -> Optional[Dict[str, Any]]:
    hits = [
        r for r in rows
        if r["wl_hot_dim_multiplier"] == mult and r["arm_name"] == arm
        and (k is None or r["arm_k"] == k)
        and (coalesce is None or r["sp_aqe_coalesce_enabled"] == coalesce)
    ]
    return hits[0] if hits else None


def build_checks(rows: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    checks: List[Tuple[str, str]] = []
    for tag, mult in TAGS.items():
        base = pick(rows, mult, "baseline")
        aqe = pick(rows, mult, "aqe", 1, True)
        if not (base and aqe):
            continue
        checks += [
            (f"AqeSpeedup{tag}",
             f"{base['wall_median_s'] / aqe['wall_median_s']:.2f}"),
            (f"BaseMaxTask{tag}", f"{base['join_task_time_max_ms'] / 1000:.1f}"),
            (f"AqeCoalMaxTask{tag}", f"{aqe['join_task_time_max_ms'] / 1000:.1f}"),
        ]
        salts = [r for r in rows
                 if r["wl_hot_dim_multiplier"] == mult
                 and r["arm_name"] == "salt_selective" and r["arm_k"] > 1]
        if salts:
            best = min(salts, key=lambda r: r["wall_median_s"])
            checks += [
                (f"SaltSpeedup{tag}",
                 f"{base['wall_median_s'] / best['wall_median_s']:.2f}"),
                (f"SaltBestK{tag}", str(best["arm_k"])),
            ]

    thin = pick(rows, 1, "baseline")
    if thin:
        checks += [
            ("MeasHotMB", f"{thin['join_shuffle_read_max_bytes'] / 1e6:.2f}"),
            ("MeasMedMB", f"{thin['join_shuffle_read_median_bytes'] / 1e6:.2f}"),
        ]

    def slope(arm: str, min_k: int) -> Optional[float]:
        block = sorted((r for r in rows
                        if r["wl_hot_dim_multiplier"] == 1
                        and r["arm_name"] == arm and r["arm_k"] >= min_k),
                       key=lambda r: r["arm_k"])
        if len(block) < 2:
            return None
        return float(np.polyfit([r["arm_k"] for r in block],
                                [r["shuffle_read_records"] for r in block], 1)[0])

    sel, uni = slope("salt_selective", 2), slope("salt_uniform", 2)
    if sel and uni:
        checks += [
            ("SelRecSlope", f"{sel:.0f}"),
            ("UnifRecSlope", f"{uni:.0f}"),
            ("RecSlopeRatio", f"{uni / sel:.0f}"),
        ]
    return checks


def main(results: str = "results/v3_single.jsonl",
         numbers: str = "paper/numbers.tex") -> int:
    rows = load_rows(results)
    macros = load_macros(numbers)
    checks = build_checks(rows)
    if not checks:
        print("FAIL: no checks could be built -- wrong results file?")
        return 1

    failures = 0
    for name, expected in checks:
        got = macros.get(name)
        ok = got == expected
        failures += 0 if ok else 1
        status = "OK  " if ok else "MISMATCH"
        print(f"{status} {name:<24} paper={got!r:<16} recomputed={expected!r}")

    print(f"\n{len(checks) - failures}/{len(checks)} headline numbers verified "
          f"independently against {results}")
    if failures:
        print(f"\nFAIL: {failures} number(s) in the manuscript do not match the "
              "data. Regenerate with `make analyze`, or find the bug.")
    return 1 if failures else 0


if __name__ == "__main__":
    args = sys.argv[1:]
    raise SystemExit(main(*args) if args else main())
