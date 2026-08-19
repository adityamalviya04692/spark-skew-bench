"""Turn a results file into the tables and figures the paper reports.

Every number that appears in the paper is produced here, written to
``results/analysis/`` as both CSV and LaTeX, and referenced by filename in the
manuscript. Nothing is transcribed by hand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from skewbench.costmodel import (CostParams, fit, optimal_k_unconstrained,  # noqa: E402
                                 predict_time, recommend_k, saturation_k)
import plots  # noqa: E402

pd.set_option("display.width", 200)


def load(path: str | Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    frame = pd.DataFrame(rows)
    if "physical_plan" in frame.columns:
        frame = frame.drop(columns=["physical_plan"])
    return frame


def _tex(frame: pd.DataFrame, path: Path, caption: str, label: str,
         float_format: str = "%.2f") -> None:
    body = frame.to_latex(index=False, escape=False, float_format=float_format,
                          column_format="l" + "r" * (frame.shape[1] - 1))
    path.write_text(
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        "\\footnotesize\n" + body + "\\end{table}\n"
    )


def summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (workload, arm, k) with the metrics the paper quotes."""
    cols = {
        "wl_hot_dim_multiplier": "mult", "wl_theta": "theta",
        "wl_hot_keys": "hot_keys", "wl_source": "source",
        "sp_aqe_coalesce_enabled": "coalesce",
        "arm_name": "arm", "arm_k": "k",
        "wall_median_s": "wall_s", "wall_iqr_s": "iqr_s",
        # max/median is retained for continuity but is NOT comparable across
        # arms with different task counts -- AQE's coalescing moves the
        # denominator. straggler_index (max/mean) and the absolute max task
        # duration are the metrics that survive that confound.
        "join_skew_ratio": "skew_ratio",
        "join_straggler_index": "straggler",
        "join_task_time_max_ms": "max_task_ms",
        "join_total_task_time_ms": "total_task_ms",
        "join_n_tasks": "join_tasks",
        "join_hot_partition_mb": "hot_part_mb",
        "join_median_partition_mb": "med_part_mb",
        "shuffle_read_mb": "shuffle_mb",
        "shuffle_read_records": "shuffle_recs",
        "spill_mb": "spill_mb",
    }
    present = {k: v for k, v in cols.items() if k in frame.columns}
    out = frame[list(present)].rename(columns=present)
    return out.sort_values(["mult", "arm", "k"]).reset_index(drop=True)


def headline_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Best configuration per arm per workload, with speedup over baseline."""
    summary = summary_table(frame)
    rows = []
    for mult, group in summary.groupby("mult"):
        base = group[group["arm"] == "baseline"]["wall_s"]
        base_time = float(base.iloc[0]) if len(base) else np.nan
        for arm, arm_group in group.groupby("arm"):
            best = arm_group.loc[arm_group["wall_s"].idxmin()]
            rows.append({
                "mult": float(mult),
                "arm": arm,
                "best_k": int(best["k"]),
                "wall_s": float(best["wall_s"]),
                "speedup_vs_baseline": (base_time / float(best["wall_s"])
                                        if base_time == base_time else np.nan),
                "skew_ratio": float(best["skew_ratio"]),
                "shuffle_mb": float(best["shuffle_mb"]),
            })
    return pd.DataFrame(rows).sort_values(["mult", "wall_s"]).reset_index(drop=True)


def fit_cost_model(frame: pd.DataFrame, mult: float,
                   arm: str = "salt_selective") -> Optional[Dict[str, Any]]:
    """Fit T(k) on one workload's salting sweep and report the derived optimum."""
    sel = frame[(frame["wl_hot_dim_multiplier"] == mult) & (frame["arm_name"] == arm)]
    sel = sel.sort_values("arm_k")
    if len(sel) < 3:
        return None

    H = float(sel["observed_hot_fact_rows"].iloc[0])
    P = float(sel["observed_hot_dim_rows"].iloc[0])
    ks = sel["arm_k"].to_numpy(dtype=float)
    times = sel["wall_median_s"].to_numpy(dtype=float)

    params = fit(ks, times, H, P)
    n_fact = int(sel["wl_n_fact"].iloc[0])
    partitions = int(sel["sp_shuffle_partitions"].iloc[0])
    cores = 2
    master = str(sel["sp_master"].iloc[0])
    if master.startswith("local[") and master[6:-1].isdigit():
        cores = int(master[6:-1])

    rec = recommend_k(H, P, n_fact, partitions, cores, params)
    return {
        "mult": float(mult), "arm": arm, "H": H, "P": P,
        "a": params.a, "b": params.b, "c": params.c,
        "se_b": params.se_b, "b_t_statistic": params.b_t_statistic,
        "identified": params.identified, "diagnosis": params.diagnosis(),
        "gamma": params.gamma, "r_squared": params.r_squared,
        "n_points": params.n_points,
        "k_star_unconstrained": optimal_k_unconstrained(H, P, params, strict=True),
        "k_star_unguarded": rec.k_star_unconstrained,
        "rho_saturation": rec.k_saturation,
        "cores": cores,
        "k_recommended": rec.k_recommended,
        "binding_constraint": rec.binding_constraint,
        "k_empirical_best": int(ks[int(np.argmin(times))]),
        "ks": ks.tolist(), "times": times.tolist(),
        "iqr": sel["wall_iqr_s"].to_numpy(dtype=float).tolist(),
        "shuffle_mb": sel["shuffle_read_mb"].to_numpy(dtype=float).tolist()
        if "shuffle_read_mb" in sel else [],
    }


def main(results_path: str, outdir: str = "results/analysis") -> Dict[str, Any]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    figdir = out / "figures"

    frame = load(results_path)
    frame.to_csv(out / "raw_results.csv", index=False)

    summary = summary_table(frame)
    summary.to_csv(out / "table_summary.csv", index=False)
    _tex(summary, out / "table_summary.tex",
         "Per-configuration results. $k$ is the salt cardinality; skew ratio is "
         "maximum over median task duration within the join stage.",
         "tab:summary")

    headline = headline_table(frame)
    headline.to_csv(out / "table_headline.csv", index=False)
    # The LaTeX version is narrowed to fit an IEEE column: the full frame,
    # including shuffle volume, stays in the CSV.
    narrow = headline.rename(columns={
        "mult": "$\\mu$", "arm": "strategy", "best_k": "$k$",
        "wall_s": "wall (s)", "speedup_vs_baseline": "speedup",
        "skew_ratio": "skew"})[
        ["$\\mu$", "strategy", "$k$", "wall (s)", "speedup", "skew"]]
    narrow["strategy"] = narrow["strategy"].str.replace("_", "\\_", regex=False)
    _tex(narrow, out / "table_headline.tex",
         "Best configuration per strategy, with speedup over the unmitigated "
         "baseline. $\\mu$ is the probe-side hot-key multiplier; skew is the "
         "join-stage maximum-to-median task duration ratio.", "tab:headline")

    report: Dict[str, Any] = {"summary_rows": len(summary), "fits": []}

    for mult in sorted(frame["wl_hot_dim_multiplier"].unique()):
        fitted = fit_cost_model(frame, float(mult))
        if not fitted:
            continue
        report["fits"].append({k: v for k, v in fitted.items()
                               if k not in ("ks", "times", "iqr", "shuffle_mb")})

        ks = np.array(fitted["ks"])
        times = np.array(fitted["times"])
        params = CostParams(fitted["a"], fitted["b"], fitted["c"])
        smooth = np.linspace(max(1.0, ks.min()), ks.max(), 200)
        modelled = predict_time(smooth, fitted["H"], fitted["P"], params)

        tag = f"mult{mult:g}"
        plots.fig_u_curve(
            ks, times, figdir, name=f"fig_u_curve_{tag}",
            model=(smooth, modelled), k_star=fitted["k_star_unconstrained"],
            iqr=np.array(fitted["iqr"]),
            title=f"Salt cardinality vs. runtime, $P={fitted['P']:.0f}$",
        )
        if fitted["shuffle_mb"]:
            plots.fig_shuffle_vs_k(ks, np.array(fitted["shuffle_mb"]), figdir,
                                   name=f"fig_shuffle_vs_k_{tag}")

        cores_axis = [1, 2, 4, 8, 16, 32, 64]
        recommended = [
            recommend_k(fitted["H"], fitted["P"], int(frame["wl_n_fact"].iloc[0]),
                        int(frame["sp_shuffle_partitions"].iloc[0]), c,
                        params).k_recommended
            for c in cores_axis
        ]
        plots.fig_model_projection(cores_axis, recommended,
                                   fitted["k_star_unconstrained"],
                                   fitted["rho_saturation"], figdir,
                                   name=f"fig_model_projection_{tag}")

    # Residual skew across strategies, at each strategy's own best k.
    # The straggler figure, at the thickest probe side where the effect is
    # largest. It plots the ABSOLUTE longest task rather than a max/median
    # ratio: coalescing changes the task count and therefore the median, so the
    # ratio is not comparable across arms and in our data ranked them backwards.
    thickest = summary["mult"].max()
    block = summary[summary["mult"] == thickest]
    picks = []
    for arm in ("baseline", "aqe", "salt_selective", "aqe_salt", "broadcast"):
        hit = block[block["arm"] == arm]
        if arm in ("aqe", "aqe_salt") and "coalesce" in hit:
            hit = hit[hit["coalesce"] == True]  # noqa: E712
        if len(hit):
            picks.append(hit.loc[hit["wall_s"].idxmin()])
    if picks:
        plots.fig_straggler(
            [p_["arm"] for p_ in picks],
            [float(p_["max_task_ms"]) for p_ in picks],
            [float(p_["straggler"]) for p_ in picks],
            [int(p_["join_tasks"]) for p_ in picks],
            figdir, name="fig_straggler",
            title=f"Longest join-stage task ($P={thickest*10:.0f}$)")

    # One-sided against two-sided at matched k: the decision rule's hinge.
    # The optimum against probe-side hot-key count: the model says k* falls as
    # 1/sqrt(P), and this is the figure that tests it.
    if len(fits_frame := pd.DataFrame(report["fits"])) >= 2:
        plots.fig_kstar_vs_p(
            fits_frame["P"].tolist(),
            fits_frame["k_star_unconstrained"].tolist(),
            fits_frame["k_empirical_best"].tolist(),
            figdir)

    sel = frame[frame["arm_name"] == "salt_selective"].sort_values("arm_k")
    mults = sorted(sel["wl_hot_dim_multiplier"].unique())
    if len(mults) >= 2:
        shared = sorted(set.intersection(*[
            set(sel[sel["wl_hot_dim_multiplier"] == m]["arm_k"]) for m in mults]))
        if len(shared) >= 3:
            series = {
                f"$P={m*10:.0f}$": [
                    float(sel[(sel["wl_hot_dim_multiplier"] == m)
                              & (sel["arm_k"] == k)]["wall_median_s"].iloc[0])
                    for k in shared]
                for m in mults}
            plots.fig_curves_by_p(shared, series, figdir)

    fits_frame = pd.DataFrame(report["fits"])
    if len(fits_frame):
        fits_frame.to_csv(out / "table_costmodel.csv", index=False)
        _tex(fits_frame[["mult", "H", "P", "gamma", "r_squared",
                         "k_star_unconstrained", "rho_saturation", "cores",
                         "k_recommended", "binding_constraint", "k_empirical_best"]],
             out / "table_costmodel.tex",
             "Fitted cost-model coefficients and the resulting recommendation. "
             "$\\gamma = a/b$ is the engine constant; $k^{*}$ is the unconstrained "
             "optimum; $\\rho$ is the saturation bound.",
             "tab:costmodel", float_format="%.3g")

    write_numbers_tex(frame, report,
                      Path(__file__).resolve().parents[1] / "paper" / "numbers.tex",
                      results_path_hint=str(results_path))

    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str)[:4000])
    return report




# ---------------------------------------------------------------------------
# Paper macros
# ---------------------------------------------------------------------------

# Replication cost per unit k.
#
# An earlier version measured this in shuffle *megabytes* and reported a
# ~7x gap between the salting variants. That was wrong. For selective
# salting the replicated rows number only P(k-1) -- a few hundred rows over
# the whole sweep -- while the measured megabyte growth was dominated by the
# `_salt` column added to every one of two million fact rows. Over 99% of
# the "replication cost" was the extra column, not replication.
#
# Shuffle *records* have no such confound: they are exactly the row count
# the model predicts, they are deterministic, and they are what the model's
# bP(k-1) term actually refers to. We fit a least-squares slope over a
# matched k range rather than taking a two-point secant, because the secant
# was sensitive to the endpoint each arm happened to start from.
def record_slope(summary: pd.DataFrame, arm: str,
                 mult: Optional[float] = None) -> Dict[str, float]:
    target = summary["mult"].min() if mult is None else mult
    block = summary[(summary["arm"] == arm) & (summary["mult"] == target)]
    block = block.sort_values("k")
    # Restrict to k >= 2: at k = 1 the salting arms short-circuit to a plain
    # join with no salt column at all, so k = 1 is a structurally different
    # plan and a high-leverage outlier in any fit.
    block = block[block["k"] >= 2]
    if len(block) < 2 or "shuffle_recs" not in block:
        return {"slope": float("nan"), "r2": float("nan"), "n": 0}
    x = block["k"].to_numpy(float)
    y = block["shuffle_recs"].to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"slope": float(slope), "n": int(len(block)),
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")}


def _fmt_int(value: float) -> str:
    """Group digits with LaTeX thin spaces, as IEEEtran expects."""
    return "\\,".join(f"{int(round(value)):,}".split(","))


def write_numbers_tex(frame: pd.DataFrame, report: Dict[str, Any],
                      path: Path, platform: Optional[Dict[str, str]] = None,
                      results_path_hint: Optional[str] = None) -> None:
    """Emit every number the manuscript quotes as a LaTeX macro.

    The manuscript contains no hand-typed measurement. If a number appears in
    the paper it was defined here, from the results file, which is the only way
    to keep a paper and its data in agreement across revisions.
    """
    from skewbench.costmodel import CostParams, decide

    ordered = sorted(report.get("fits", []), key=lambda f: f["mult"])
    one = ordered[0] if ordered else None
    two = ordered[-1] if len(ordered) > 1 else None

    row = frame.iloc[0]
    n_fact = int(row["wl_n_fact"])
    n_dim = int(row["wl_n_dim"])
    partitions = int(row["sp_shuffle_partitions"])
    # Row width must be measured in *shuffle* bytes, not compressed Parquet
    # bytes: AQE's trigger compares shuffle-partition sizes, and Parquet on disk
    # is columnar and compressed, understating the shuffle width several-fold.
    # Taken from the baseline arm, whose plan adds no replication.
    base_rows = frame[frame["arm_name"] == "baseline"]
    if len(base_rows) and "shuffle_write_bytes" in base_rows:
        row_bytes = float(base_rows["shuffle_write_bytes"].median()) / n_fact
    else:
        row_bytes = float(row.get("fact_row_bytes", 80.0))
    H = float(one["H"]) if one else float(row.get("observed_hot_fact_rows", 0))
    P = float(one["P"]) if one else float(row.get("observed_hot_dim_rows", 0))

    summary = summary_table(frame)
    sel_slope = record_slope(summary, "salt_selective")
    unif_slope = record_slope(summary, "salt_uniform")

    dec = decide(H, P, n_fact, row_bytes, partitions,
                 two_sided=bool(two and two["P"] > 1.5 * (one["P"] if one else 0)),
                 skewed_partition_threshold=str(row["sp_skewed_partition_threshold"]),
                 skewed_partition_factor=int(row["sp_skewed_partition_factor"]))

    sel = summary[(summary["arm"] == "salt_selective")].sort_values("k")
    best_k = int(sel.loc[sel["wall_s"].idxmin(), "k"]) if len(sel) else 1

    # The crossover: AQE's advantage decays as the probe side thickens while
    # salting's grows. This is the study's headline, so it is computed here
    # rather than read off a table by hand.
    crossover: Dict[float, Dict[str, float]] = {}
    for mult, group in summary.groupby("mult"):
        def _wall(arm):
            hit = group[group["arm"] == arm]["wall_s"]
            return float(hit.min()) if len(hit) else float("nan")

        def _skew(arm):
            hit = group[group["arm"] == arm]["skew_ratio"]
            return float(hit.iloc[0]) if len(hit) else float("nan")

        base = _wall("baseline")
        crossover[float(mult)] = {
            "P": float(mult) * (n_dim / int(row["wl_n_engines"])),
            "base_wall": base,
            "aqe_wall": _wall("aqe"),
            "salt_wall": _wall("salt_selective"),
            "aqe_speedup": base / _wall("aqe"),
            "salt_speedup": base / _wall("salt_selective"),
            "base_skew": _skew("baseline"),
            "aqe_skew": _skew("aqe"),
        }


    # Measured hot and median shuffle-partition sizes, taken from the baseline
    # arm's event log. These replace the analytical estimates Hw and
    # (N-H)w/(p-1), which understate the maximum (the hot partition also absorbs
    # cold keys) and overstate the median (they ignore variance among the cold
    # partitions). The AQE trigger is evaluated against the real sizes.
    measured_hot = measured_med = float("nan")
    base_block = frame[frame["arm_name"] == "baseline"]
    if len(base_block) and "join_hot_partition_mb" in base_block:
        measured_hot = float(base_block["join_hot_partition_mb"].median())
        measured_med = float(base_block["join_median_partition_mb"].median())

    macros: Dict[str, str] = {
        "MeasHotMB": f"{measured_hot:.2f}" if measured_hot == measured_hot else "n/a",
        "MeasMedMB": f"{measured_med:.2f}" if measured_med == measured_med else "n/a",
        "NFACT": _fmt_int(n_fact),
        "NDIM": _fmt_int(n_dim),
        "NPART": str(partitions),
        "HHOT": _fmt_int(H),
        "PHOT": _fmt_int(P),
        "ROWBYTES": f"{row_bytes:.0f}",
        "RHOVAL": f"{dec.rho:.2f}",
        "HOTMB": f"{dec.hot_partition_bytes/1e6:.1f}",
        "THRESHMB": f"{dec.trigger_threshold_bytes/1e6:.1f}",
        "BESTK": str(best_k),
        "SelRecSlope": f"{sel_slope['slope']:.0f}",
        "UnifRecSlope": f"{unif_slope['slope']:.0f}",
        "SelRecR": f"{sel_slope['r2']:.3f}",
        "UnifRecR": f"{unif_slope['r2']:.3f}",
        "RecSlopeRatio": f"{unif_slope['slope'] / max(1.0, sel_slope['slope']):.0f}",
        "PredSlopeRatio": f"{float(n_dim) / max(1.0, P):.0f}",
        "CORES": str(one["cores"]) if one else "2",
        "DRIVERMEM": str(row["sp_driver_memory"]).replace("g", "\\,GB"),
        "SPARKVER": (platform or {}).get("spark", "3.5.3"),
        "JAVAVER": (platform or {}).get("java", "21"),
    }
    # One macro set per fit. The earlier version emitted only the first and last
    # fit's R^2, which silently dropped the middle (and worst) one -- that reads
    # as selective reporting even when it is an oversight.
    for index, f in enumerate(ordered):
        tag = ["Thin", "Mid", "Thick"][index] if index < 3 else f"Fit{index}"
        macros[f"Rsq{tag}"] = f"{f['r_squared']:.3f}"
        macros[f"Gamma{tag}"] = f"{f['gamma']:.3g}"
        macros[f"Btstat{tag}"] = f"{f['b_t_statistic']:.2f}"
        macros[f"Ident{tag}"] = "yes" if f["identified"] else "no"
        kstar = f["k_star_unconstrained"]
        macros[f"Kstar{tag}"] = ("not identified" if kstar != kstar
                                 else f"{kstar:.1f}")
    negative_b = sum(1 for f in ordered if f["b"] <= 0)
    macros["NumNegativeB"] = str(negative_b)
    identified_count = sum(1 for f in ordered if f["identified"])
    macros["NumIdentified"] = str(identified_count)
    macros["NumFits"] = str(len(ordered))
    if ordered:
        gammas = [f["gamma"] for f in ordered if f["gamma"] == f["gamma"]
                  and f["gamma"] not in (float("inf"), float("-inf"))
                  and f["gamma"] > 0]
        macros["GammaSpread"] = (
            f"{max(gammas) / max(1e-12, min(gammas)):.0f}" if len(gammas) >= 2
            else "not estimable")
        macros["NumGammaEstimable"] = str(len(gammas))

    if one:
        macros.update({
            "Rsquared": f"$R^{{2}}={one['r_squared']:.3f}$",
            "GAMMAONE": f"{one['gamma']:.3g}",
            "KSTARONE": f"{one['k_star_unconstrained']:.1f}",
            "RHOSAT": f"{one['rho_saturation']:.1f}",
            "KEFF": str(one["k_recommended"]),
            "BINDING": str(one["binding_constraint"]).replace("_", " "),
            "KEMPONE": str(one["k_empirical_best"]),
        })
    if two:
        macros.update({
            "RsquaredTwo": f"$R^{{2}}={two['r_squared']:.3f}$",
            "KSTARTWO": f"{two['k_star_unconstrained']:.1f}",
            "KEMPTWO": str(two["k_empirical_best"]),
            "GAMMATWO": f"{two['gamma']:.3g}",
        })

    # Coalescing control: AQE with partition coalescing disabled isolates the
    # skew-join rule's own contribution from coalescing's. Without this pair the
    # claim "the benefit was coalescing" is an inference; with it, it is a
    # measurement.
    coal_tags = ["Thin", "Mid", "Thick"]
    for index, mult in enumerate(sorted(summary["mult"].unique())):
        if index >= len(coal_tags):
            break
        tag = coal_tags[index]
        block = summary[summary["mult"] == mult]
        for arm, label in (("aqe", "Aqe"), ("baseline", "Base")):
            for flag, suffix in ((True, "Coal"), (False, "NoCoal")):
                hit = block[(block["arm"] == arm) & (block["coalesce"] == flag)] \
                    if "coalesce" in block else block.iloc[0:0]
                if len(hit):
                    row_ = hit.iloc[0]
                    macros[f"{label}{suffix}Wall{tag}"] = f"{float(row_['wall_s']):.2f}"
                    macros[f"{label}{suffix}WallPrecise{tag}"] = f"{float(row_['wall_s']):.2f}"
                    macros[f"{label}{suffix}MaxTask{tag}"] = f"{float(row_['max_task_ms'])/1000:.1f}"
                    macros[f"{label}{suffix}Skew{tag}"] = f"{float(row_['skew_ratio']):.1f}"
                    macros[f"{label}{suffix}Tasks{tag}"] = f"{int(row_['join_tasks'])}"
        bc = block[block["arm"] == "broadcast"]
        if len(bc):
            macros[f"BcastWall{tag}"] = f"{float(bc['wall_s'].iloc[0]):.2f}"
        base = block[block["arm"] == "baseline"]
        best = block[block["arm"] == "salt_selective"]
        if len(base) and len(best):
            b0 = base.iloc[0]
            s0 = best.loc[best["wall_s"].idxmin()]
            macros[f"BaseMaxTask{tag}"] = f"{float(b0['max_task_ms'])/1000:.1f}"
            macros[f"SaltMaxTask{tag}"] = f"{float(s0['max_task_ms'])/1000:.1f}"
            macros[f"SaltBestK{tag}"] = f"{int(s0['k'])}"
            macros[f"SaltTaskCut{tag}"] = \
                f"{float(b0['max_task_ms'])/max(1.0, float(s0['max_task_ms'])):.1f}"

    tags = ["Thin", "Mid", "Thick"]
    for index, (mult, stats) in enumerate(sorted(crossover.items())):
        tag = tags[index] if index < len(tags) else f"Lvl{index}"
        macros[f"P{tag}"] = f"{stats['P']:.0f}"
        macros[f"AqeSpeedup{tag}"] = f"{stats['aqe_speedup']:.2f}"
        macros[f"SaltSpeedup{tag}"] = f"{stats['salt_speedup']:.2f}"
        macros[f"BaseSkew{tag}"] = f"{stats['base_skew']:.0f}"
        macros[f"AqeSkew{tag}"] = f"{stats['aqe_skew']:.1f}"
        macros[f"BaseWall{tag}"] = f"{stats['base_wall']:.1f}"
        macros[f"AqeWall{tag}"] = f"{stats['aqe_wall']:.1f}"
        macros[f"SaltWall{tag}"] = f"{stats['salt_wall']:.1f}"

    headline = headline_table(frame)
    mults = sorted(headline["mult"].unique())
    for mult, suffix in ((mults[0], "One"), (mults[-1], "Two")):
        block = headline[headline["mult"] == mult]
        for arm, macro in (("aqe", "Aqe"), ("salt_selective", "Salt"),
                           ("baseline", "Base"), ("aqe_salt", "AqeSalt"),
                           ("salt_uniform", "Unif")):
            hit = block[block["arm"] == arm]
            if len(hit):
                macros[f"{macro}Speedup{suffix}"] = f"{float(hit['speedup_vs_baseline'].iloc[0]):.2f}"
                macros[f"{macro}Skew{suffix}"] = f"{float(hit['skew_ratio'].iloc[0]):.2f}"
                macros[f"{macro}Wall{suffix}"] = f"{float(hit['wall_s'].iloc[0]):.1f}"
                macros[f"{macro}Shuffle{suffix}"] = f"{float(hit['shuffle_mb'].iloc[0]):.0f}"

    # The noise floor, computed here rather than read from a side file: the
    # comparison is baseline against selective salting at k=1, which are the
    # same physical plan under the same configuration.
    discrepancies = []
    cross_session = 0
    has_session = "session_id" in frame.columns
    for mult in sorted(frame["wl_hot_dim_multiplier"].unique()):
        block = frame[frame["wl_hot_dim_multiplier"] == mult]
        base = block[block["arm_name"] == "baseline"]
        ctrl = block[(block["arm_name"] == "salt_selective") & (block["arm_k"] == 1)]
        if not (len(base) and len(ctrl)):
            continue
        if has_session and base["session_id"].iloc[0] != ctrl["session_id"].iloc[0]:
            # A wall-clock comparison across Spark applications measures JVM
            # warmth and host load, not the plans. Excluded, and counted.
            cross_session += 1
            continue
        lo, hi = sorted((float(base["wall_median_s"].iloc[0]),
                         float(ctrl["wall_median_s"].iloc[0])))
        discrepancies.append(100.0 * (hi - lo) / lo)
    macros["ControlCrossSessionExcluded"] = str(cross_session)
    if discrepancies:
        ordered_d = sorted(discrepancies)
        macros["ControlWorstPct"] = f"{max(discrepancies):.1f}"
        macros["ControlMedianPct"] = f"{ordered_d[len(ordered_d) // 2]:.1f}"
        macros["ControlN"] = str(len(discrepancies))

    lines = ["% Auto-generated by analysis/analyze.py -- do not edit by hand.",
             "% Every measurement quoted in the manuscript is defined here.", ""]
    for name, value in sorted(macros.items()):
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"[numbers] {len(macros)} macros -> {path}")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0] if args else "results/pilot.jsonl",
         args[1] if len(args) > 1 else "results/analysis")
