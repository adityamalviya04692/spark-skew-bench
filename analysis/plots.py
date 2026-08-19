"""Figure generation for the paper.

Print and grayscale constraints drive every choice here. An IEEE paper is read
on paper as often as on screen, and often photocopied, so colour is treated as a
*secondary* channel throughout: every series also carries a distinct marker shape
and dash pattern, and every bar group a distinct hatch. Remove all colour and the
figures remain readable, which is the only reliable test.

The categorical hues are slots 1, 2, 3, 7 and 8 of the reference palette. That
five-slot subset was validated with the palette checker: worst adjacent CVD
separation dE 9.2 (deutan) and worst adjacent normal-vision separation dE 27.6,
both clear of their floors. The aqua slot falls below 3:1 contrast on a white
surface, which obliges relief -- so every figure ships direct labels, and the
same numbers appear in the tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- palette (validated; see module docstring) -----------------------------
SERIES = {
    "baseline":       "#e34948",  # slot 8, red
    "aqe":            "#2a78d6",  # slot 1, blue
    "salt_selective": "#eb6834",  # slot 2, orange
    "salt_uniform":   "#1baf7a",  # slot 3, aqua
    "aqe_salt":       "#4a3aa7",  # slot 7, violet
    "model":          "#52514e",  # neutral ink, never a series colour
}
MARKERS = {"baseline": "s", "aqe": "o", "salt_selective": "^",
           "salt_uniform": "D", "aqe_salt": "v"}
DASHES = {"baseline": (3, 2), "aqe": (), "salt_selective": (5, 2),
          "salt_uniform": (1, 1.5), "aqe_salt": (6, 2, 1, 2)}
HATCH = {"baseline": "", "aqe": "//", "salt_selective": "\\\\",
         "salt_uniform": "xx", "aqe_salt": ".."}

LABEL = {"baseline": "Baseline (no AQE)", "aqe": "AQE only",
         "salt_selective": "Selective salting", "salt_uniform": "Uniform salting",
         "aqe_salt": "AQE + selective salting"}

INK = "#0b0b0b"
MUTED = "#8a8a85"
COLUMN_WIDTH_IN = 3.45   # IEEE two-column text width
DOUBLE_WIDTH_IN = 7.16


def use_paper_style() -> None:
    """Typography and axis treatment matched to IEEEtran body text."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "axes.edgecolor": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e6e2",
        "grid.linewidth": 0.5,
        "grid.alpha": 1.0,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.5,
        "figure.dpi": 200,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
    })


def _save(fig, outdir: Path, name: str) -> List[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        path = outdir / f"{name}.{ext}"
        fig.savefig(path)
        paths.append(path)
    plt.close(fig)
    return paths


def _style_series(arm: str) -> Dict:
    return {
        "color": SERIES.get(arm, MUTED),
        "marker": MARKERS.get(arm, "o"),
        "dashes": DASHES.get(arm, ()),
        "markeredgecolor": "white",
        "markeredgewidth": 0.6,
    }


def fig_u_curve(k: Sequence[float], observed: Sequence[float],
                outdir: Path, name: str = "fig_u_curve",
                model: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
                k_star: Optional[float] = None,
                iqr: Optional[Sequence[float]] = None,
                ylabel: str = "Median wall-clock time (s)",
                title: str = "Salt cardinality and total runtime") -> List[Path]:
    """The headline figure: runtime against salt cardinality, with the fitted model."""
    use_paper_style()
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.5))

    # The model is the continuous curve; the measurements are points on top of
    # it. Drawing the measurements as a joined line would hide the fit underneath
    # and invite the reader to mistake interpolation for prediction.
    if model is not None:
        ax.plot(model[0], model[1], color=SERIES["model"], linewidth=1.2,
                zorder=2, label=r"fitted $T(k)$")

    if iqr is not None:
        ax.errorbar(k, observed, yerr=iqr, fmt="none", ecolor=MUTED,
                    elinewidth=0.7, capsize=2, zorder=3)

    style = _style_series("salt_selective")
    style.pop("dashes", None)
    ax.plot(k, observed, label="measured", zorder=4, linestyle="none",
            markersize=5.5, **style)

    if k_star is not None:
        ax.axvline(k_star, color=SERIES["model"], linewidth=0.8, dashes=(1, 2), zorder=1)
        ax.annotate(rf"$k^{{*}}={k_star:.1f}$", xy=(k_star, max(observed)),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=7, color=SERIES["model"], va="center")

    # Direct label on the measured minimum: relief for the contrast WARN, and the
    # single number a reader most wants off this figure.
    best = int(np.argmin(observed))
    ax.annotate(f"min at $k$={int(k[best])}\n{observed[best]:.1f}s",
                xy=(k[best], observed[best]), xytext=(0, -22),
                textcoords="offset points", fontsize=7, ha="center",
                color=SERIES["salt_selective"])

    ax.set_xscale("log", base=2)
    ax.set_xticks(list(k))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Salt cardinality $k$")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", color=INK)
    ax.legend(frameon=False, loc="upper center", ncol=2, handlelength=2.0,
              columnspacing=1.2, borderpad=0.1)
    ax.margins(y=0.26)
    return _save(fig, outdir, name)


def fig_shuffle_vs_k(k: Sequence[float], shuffle_mb: Sequence[float],
                     outdir: Path, name: str = "fig_shuffle_vs_k",
                     second: Optional[Tuple[str, Sequence[float]]] = None) -> List[Path]:
    """Shuffle volume against k. Deterministic, so this is the portable evidence."""
    use_paper_style()
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.4))
    ax.plot(k, shuffle_mb, label="selective salting", **_style_series("salt_selective"))
    if second is not None:
        ax.plot(k, second[1], label=second[0], **_style_series("salt_uniform"))
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(k))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Salt cardinality $k$")
    ax.set_ylabel("Shuffle read volume (MB)")
    ax.set_title("Shuffle volume against salt cardinality", loc="left", color=INK)
    ax.legend(frameon=False, loc="upper left", handlelength=2.4)
    ax.margins(y=0.18)
    return _save(fig, outdir, name)


def fig_arm_comparison(arms: Sequence[str], values: Sequence[float],
                       outdir: Path, name: str = "fig_arm_comparison",
                       ylabel: str = "Join-stage skew ratio (max/median task time)",
                       title: str = "Residual skew by strategy",
                       reference: Optional[float] = None,
                       reference_label: str = "no skew") -> List[Path]:
    """Grouped bars across strategies, hatched so the ranking survives grayscale."""
    use_paper_style()
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.5))
    positions = np.arange(len(arms))
    for pos, arm, value in zip(positions, arms, values):
        ax.bar(pos, value, width=0.62, color=SERIES.get(arm, MUTED),
               edgecolor="white", linewidth=1.0, hatch=HATCH.get(arm, ""), zorder=3)
        ax.annotate(f"{value:.2f}", xy=(pos, value), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=7, color=INK)
    if reference is not None:
        ax.axhline(reference, color=SERIES["model"], linewidth=0.8, dashes=(3, 2), zorder=2)
        ax.annotate(reference_label, xy=(len(arms) - 0.4, reference), xytext=(0, 2),
                    textcoords="offset points", ha="right", fontsize=6.5, color=MUTED)
    ax.set_xticks(positions)
    ax.set_xticklabels([LABEL.get(a, a).replace(" ", "\n", 1) for a in arms], fontsize=6.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", color=INK)
    ax.margins(y=0.20)
    return _save(fig, outdir, name)


def fig_curves_by_p(k: Sequence[float], series: Dict[str, Sequence[float]],
                    outdir: Path, name: str = "fig_curves_by_p") -> List[Path]:
    """One U-curve per probe-side hot-key count.

    The model says the minimum should move left as P grows, because replication
    gets more expensive while straggler relief does not change. This figure is
    where that prediction either shows up or does not.
    """
    use_paper_style()
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.6))
    order = ["aqe", "salt_selective", "salt_uniform", "aqe_salt", "baseline"]
    for index, (label, values) in enumerate(series.items()):
        style = _style_series(order[index % len(order)])
        ax.plot(k, values, label=label, **style)
        best = int(np.argmin(values))
        ax.plot([k[best]], [values[best]], marker="o", markersize=9,
                markerfacecolor="none", markeredgewidth=1.1,
                markeredgecolor=style["color"], zorder=5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(k))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Salt cardinality $k$")
    ax.set_ylabel("Median wall-clock time (s)")
    ax.set_title("Runtime against salt cardinality, by probe thickness",
                 loc="left", color=INK)
    ax.legend(frameon=False, loc="upper center", ncol=3, handlelength=1.8,
              columnspacing=1.0, borderpad=0.1)
    ax.margins(y=0.26)
    return _save(fig, outdir, name)


def fig_kstar_vs_p(p_values: Sequence[float], k_star: Sequence[float],
                   k_empirical: Sequence[float], outdir: Path,
                   name: str = "fig_kstar_vs_p") -> List[Path]:
    """Predicted and measured optimum against P, with the $1/\sqrt{P}$ reference."""
    use_paper_style()
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.5))
    order = np.argsort(p_values)
    p_arr = np.asarray(p_values, dtype=float)[order]
    ks_arr = np.asarray(k_star, dtype=float)[order]
    ke_arr = np.asarray(k_empirical, dtype=float)[order]

    reference = ks_arr[0] * np.sqrt(p_arr[0] / p_arr)
    ax.plot(p_arr, reference, color=SERIES["model"], linewidth=1.1,
            dashes=(4, 2), label=r"$k^{*}\propto 1/\sqrt{P}$", zorder=2)
    ax.plot(p_arr, ks_arr, label="model $k^{*}$", zorder=3,
            **_style_series("aqe"))
    style = _style_series("salt_selective")
    style.pop("dashes", None)
    ax.plot(p_arr, ke_arr, label="measured best $k$", linestyle="none",
            markersize=6, zorder=4, **style)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(list(p_arr))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Probe-side hot-key rows $P$")
    ax.set_ylabel("Optimal salt cardinality")
    ax.set_title("Predicted and measured optimum against $P$",
                 loc="left", color=INK)
    ax.legend(frameon=False, loc="upper right", handlelength=2.0)
    ax.margins(y=0.20)
    return _save(fig, outdir, name)


def fig_model_projection(cores: Sequence[int], k_recommended: Sequence[int],
                         k_star: float, rho: float, outdir: Path,
                         name: str = "fig_model_projection") -> List[Path]:
    """Which ceiling binds, as a function of available parallelism."""
    use_paper_style()
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.4))
    ax.plot(cores, k_recommended, label=r"recommended $k$",
            **_style_series("salt_selective"))
    ax.axhline(k_star, color=SERIES["aqe"], linewidth=1.0, dashes=(5, 2),
               label=rf"cost optimum $k^{{*}}={k_star:.1f}$")
    ax.axhline(rho, color=SERIES["model"], linewidth=1.0, dashes=(1, 1.5),
               label=rf"saturation $\rho={rho:.1f}$")
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(cores))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Concurrent task slots $C$")
    ax.set_ylabel(r"Recommended salt cardinality $k$")
    ax.set_title("Which ceiling binds", loc="left", color=INK)
    ax.legend(frameon=False, loc="lower right", handlelength=2.4)
    ax.margins(y=0.20)
    return _save(fig, outdir, name)


def fig_straggler(arms: Sequence[str], max_task_ms: Sequence[float],
                  straggler_index: Sequence[float], n_tasks: Sequence[int],
                  outdir: Path, name: str = "fig_straggler",
                  title: str = "Critical path by strategy") -> List[Path]:
    """Absolute maximum task duration, with the straggler index and task count.

    This replaces an earlier figure built on max-over-median task duration,
    which was not comparable across arms: AQE's partition coalescing reduces the
    task count, which moves the median and therefore the ratio without touching
    the straggler. On our data that inverted the ranking -- the arm with the
    longest critical path appeared to have the least skew.

    The bars show the critical path itself, which has no denominator to move.
    The task count is printed on each bar, because a reader comparing any ratio
    across arms needs to see immediately when the denominators differ.
    """
    use_paper_style()
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.6))
    positions = np.arange(len(arms))
    for pos, arm, ms, index, tasks in zip(positions, arms, max_task_ms,
                                          straggler_index, n_tasks):
        ax.bar(pos, ms / 1000.0, width=0.62, color=SERIES.get(arm, MUTED),
               edgecolor="white", linewidth=1.0, hatch=HATCH.get(arm, ""), zorder=3)
        ax.annotate(f"{ms/1000.0:.1f}s", xy=(pos, ms / 1000.0), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=7, color=INK)
        ax.annotate(f"{int(tasks)} tasks\nidx {index:.1f}", xy=(pos, 0),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=6, color="white" if ms > 0 else MUTED)
    ax.set_xticks(positions)
    ax.set_xticklabels([LABEL.get(a, a).replace(" ", "\n", 1) for a in arms],
                       fontsize=6.5)
    ax.set_ylabel("Longest join-stage task (s)")
    ax.set_title(title, loc="left", color=INK)
    ax.margins(y=0.22)
    return _save(fig, outdir, name)
