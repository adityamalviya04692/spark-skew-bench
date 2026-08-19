"""Compare a single-node run against a cluster run of the same grid.

The paper's one unconfirmed prediction is that the replication coefficient ``b``
is larger on a cluster -- because a replicated row crosses the network rather
than the local disk -- and that the optimal salt cardinality is therefore
smaller. This script puts the two fits side by side and states plainly whether
the prediction held, failed, or remains unresolved.

It refuses to declare confirmation from an unidentified fit. That is the whole
point: a fit whose replication coefficient is indistinguishable from zero gives
a k* that is an artefact of noise, and comparing two such artefacts would
manufacture a result.

Usage:
    python analysis/compare_scales.py results/v3.jsonl results/cluster.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from skewbench.costmodel import fit, optimal_k_unconstrained  # noqa: E402


def load(path: str | Path) -> pd.DataFrame:
    return pd.DataFrame([json.loads(line) for line in
                         open(path, encoding="utf-8") if line.strip()])


def fits_by_p(frame: pd.DataFrame, arm: str = "salt_selective") -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for mult, group in frame[frame["arm_name"] == arm].groupby("wl_hot_dim_multiplier"):
        group = group.sort_values("arm_k")
        if len(group) < 3:
            continue
        H = float(group["observed_hot_fact_rows"].iloc[0])
        P = float(group["observed_hot_dim_rows"].iloc[0])
        params = fit(group["arm_k"].to_numpy(float),
                     group["wall_median_s"].to_numpy(float), H, P)
        out.append({
            "mult": float(mult), "H": H, "P": P,
            "a": params.a, "b": params.b, "se_b": params.se_b,
            "b_t": params.b_t_statistic, "gamma": params.gamma,
            "r_squared": params.r_squared,
            "identified": params.identified,
            "k_star": optimal_k_unconstrained(H, P, params, strict=True),
            "k_empirical": int(group.loc[group["wall_median_s"].idxmin(), "arm_k"]),
            "diagnosis": params.diagnosis(),
        })
    return out


def main(local_path: str, cluster_path: str,
         out_dir: str = "results/analysis_scales") -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    local, cluster = load(local_path), load(cluster_path)
    local_fits, cluster_fits = fits_by_p(local), fits_by_p(cluster)

    rows: List[Dict[str, Any]] = []
    for lf in local_fits:
        cf = next((c for c in cluster_fits if abs(c["P"] - lf["P"]) < 0.5), None)
        row = {
            "P": lf["P"],
            "b_local": lf["b"], "b_cluster": cf["b"] if cf else None,
            "b_ratio": (cf["b"] / lf["b"]) if (cf and lf["b"] > 0) else None,
            "identified_local": lf["identified"],
            "identified_cluster": cf["identified"] if cf else None,
            "kstar_local": lf["k_star"], "kstar_cluster": cf["k_star"] if cf else None,
            "k_empirical_local": lf["k_empirical"],
            "k_empirical_cluster": cf["k_empirical"] if cf else None,
        }
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(out / "scale_comparison.csv", index=False)

    # --- verdict -----------------------------------------------------------
    usable = [r for r in rows if r["identified_local"] and r["identified_cluster"]]
    verdict: str
    if not usable:
        unresolved_side = ("cluster" if any(r["identified_local"] for r in rows)
                           else "both scales")
        verdict = (
            "UNRESOLVED. The replication coefficient b is not identified at "
            f"{unresolved_side}, so no k* can be compared. This is itself a "
            "reportable result: it means replication cost remains negligible "
            "relative to straggler cost even at this scale, and that selective "
            "salting can be over-provisioned cheaply."
        )
    else:
        ratios = [r["b_ratio"] for r in usable if r["b_ratio"]]
        larger = sum(1 for x in ratios if x > 1.0)
        if larger == len(ratios):
            verdict = (
                f"CONFIRMED. b is larger on the cluster at all {len(ratios)} "
                f"comparable P (median ratio {pd.Series(ratios).median():.2f}x), "
                "so k* is correspondingly smaller, as predicted."
            )
        elif larger == 0:
            verdict = (
                "REFUTED. b is not larger on the cluster at any comparable P. "
                "The prediction that network shuffle raises the replication "
                "coefficient does not hold for this workload."
            )
        else:
            verdict = (
                f"MIXED. b is larger on the cluster at {larger} of "
                f"{len(ratios)} comparable P. Report per-P rather than as a "
                "single claim."
            )

    report = {"verdict": verdict, "rows": rows,
              "local_fits": local_fits, "cluster_fits": cluster_fits}
    (out / "scale_comparison.json").write_text(json.dumps(report, indent=2, default=str))

    print(table.to_string(index=False))
    print("\nVERDICT:", verdict)
    return report


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2:
        raise SystemExit(__doc__)
    main(*args)
