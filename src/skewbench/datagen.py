"""Zipfian-skewed synthetic data generation with an aerospace engine-telemetry schema.

Two tables are produced:

``fact_telemetry``
    The large, skewed side. One row per engine sensor snapshot. The join key
    ``engine_id`` is drawn from a truncated Zipf distribution over ``n_engines``
    distinct engines, so a small number of engines dominate the table. This
    mirrors a real fleet, where a handful of problem units generate a
    disproportionate share of recorded events.

``dim_maintenance``
    The smaller probe side. One row per maintenance work order. Keys are
    uniform except for the hot key(s), which carry ``hot_dim_multiplier`` times
    the mean per-key row count. A multiplier of 1 gives the classic one-sided
    skew; larger multipliers make the skew increasingly two-sided, which is the
    case AQE's split-and-replicate degenerates on.

    Drawing the probe side from the *same* Zipf law as the fact side would be
    the obvious design and is the wrong one: the hot key then contributes
    ``H x P`` output rows on its own, which at these scales is order 10^10 and
    makes the workload a cardinality explosion rather than a join benchmark.
    Bounding the probe-side hot-key count keeps the output tractable and, more
    usefully, makes ``P`` an independent variable -- which is exactly what is
    needed to test the cost model's prediction that the optimal salt
    cardinality falls as ``1/sqrt(P)``.

The schema follows the NASA C-MAPSS turbofan degradation convention (unit
number, operational cycle, three operational settings, N sensor channels) so
that the generator can be swapped for the real dataset via
:mod:`skewbench.cmapss` without touching the benchmark.

Generation is chunked through pandas and written as Parquet. A manifest records
the exact quantities the cost model needs, so that no downstream stage has to
re-derive them by scanning.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from skewbench.config import WorkloadSpec

CHUNK_ROWS = 500_000
FLIGHT_PHASES = np.array(["takeoff", "climb", "cruise", "descent", "landing"])
FINDING_CODES = np.array(["FC-BLADE", "FC-SEAL", "FC-BEARING", "FC-FUEL",
                          "FC-VIBR", "FC-EGT", "FC-NONE"])


def zipf_weights(n: int, theta: float) -> np.ndarray:
    """Normalised truncated-Zipf probabilities over ``n`` ranks.

    ``p_i is proportional to i ** -theta`` for i in 1..n. ``theta=0`` gives the
    uniform distribution, which is the natural no-skew control.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    ranks = np.arange(1, n + 1, dtype=np.float64)
    weights = ranks ** (-float(theta))
    return weights / weights.sum()


def engine_id(rank_index: np.ndarray) -> np.ndarray:
    """Map zero-based rank indices to stable, readable engine identifiers."""
    return np.char.add("ENG-", np.char.zfill(rank_index.astype(str), 6))


def _sample_keys(rng: np.random.Generator, size: int, cdf: np.ndarray) -> np.ndarray:
    """Inverse-transform sampling of key ranks from a precomputed CDF."""
    return np.searchsorted(cdf, rng.random(size), side="right")


def _telemetry_chunk(rng: np.random.Generator, size: int, cdf: np.ndarray,
                     n_sensors: int, offset: int) -> pd.DataFrame:
    ranks = _sample_keys(rng, size, cdf)
    frame = pd.DataFrame({
        "event_id": np.arange(offset, offset + size, dtype=np.int64),
        "engine_id": engine_id(ranks),
        "cycle": rng.integers(1, 400, size=size, dtype=np.int32),
        "ts": (1_750_000_000 + rng.integers(0, 31_536_000, size=size)).astype(np.int64),
        "flight_phase": FLIGHT_PHASES[rng.integers(0, len(FLIGHT_PHASES), size=size)],
        "op_setting_1": rng.normal(0.0, 1.0, size).astype(np.float32),
        "op_setting_2": rng.normal(0.0, 1.0, size).astype(np.float32),
        "op_setting_3": np.full(size, 100.0, dtype=np.float32),
    })
    # Sensor channels. Means and scales loosely follow C-MAPSS magnitudes so that
    # the payload compresses like real telemetry rather than like random noise.
    for s in range(1, n_sensors + 1):
        centre = 100.0 * (1 + (s % 7))
        frame[f"sensor_{s:02d}"] = rng.normal(centre, centre * 0.02, size).astype(np.float32)
    return frame


def _maintenance_chunk(rng: np.random.Generator, size: int, cdf: np.ndarray,
                       offset: int) -> pd.DataFrame:
    ranks = _sample_keys(rng, size, cdf)
    return pd.DataFrame({
        "wo_id": np.arange(offset, offset + size, dtype=np.int64),
        "engine_id": engine_id(ranks),
        "wo_ts": (1_750_000_000 + rng.integers(0, 31_536_000, size=size)).astype(np.int64),
        "shop_visit": rng.integers(1, 12, size=size, dtype=np.int32),
        "part_no": np.char.add("P-", np.char.zfill(
            rng.integers(0, 50_000, size=size).astype(str), 5)),
        "finding_code": FINDING_CODES[rng.integers(0, len(FINDING_CODES), size=size)],
        "labour_hours": np.round(rng.gamma(2.0, 3.0, size), 2).astype(np.float32),
    })


def _write_chunked(path: Path, n_rows: int, builder, **kwargs) -> None:
    path.mkdir(parents=True, exist_ok=True)
    written = 0
    part = 0
    while written < n_rows:
        size = min(CHUNK_ROWS, n_rows - written)
        frame = builder(size=size, offset=written, **kwargs)
        frame.to_parquet(path / f"part-{part:05d}.parquet", index=False,
                         compression="snappy")
        written += size
        part += 1


def generate(spec: WorkloadSpec, root: str | Path, overwrite: bool = False) -> Dict:
    """Materialise a workload to Parquet and return its manifest.

    The manifest is the contract between generation and analysis. It carries the
    exact hot-key row counts on both sides, which the cost model consumes as
    ``H`` and ``P_hot``; deriving them here rather than by a later scan keeps the
    measured pipeline free of bookkeeping work.
    """
    root = Path(root) / spec.slug
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and not overwrite:
        return json.loads(manifest_path.read_text())
    if root.exists() and overwrite:
        shutil.rmtree(root)

    if spec.source == "cmapss":
        return _generate_cmapss(spec, root, manifest_path)

    rng = np.random.default_rng(spec.seed)
    fact_weights = zipf_weights(spec.n_engines, spec.theta)
    fact_cdf = np.cumsum(fact_weights)

    dim_weights = probe_weights(spec)
    dim_cdf = np.cumsum(dim_weights)

    _write_chunked(root / "fact_telemetry", spec.n_fact, _telemetry_chunk,
                   rng=rng, cdf=fact_cdf, n_sensors=spec.n_sensors)
    _write_chunked(root / "dim_maintenance", spec.n_dim, _maintenance_chunk,
                   rng=rng, cdf=dim_cdf)

    hot_ranks = np.arange(spec.hot_keys)
    hot_ids: List[str] = engine_id(hot_ranks).tolist()

    # Expected counts follow directly from the sampling law; they are exact in
    # expectation and used by the analytical model. Observed counts are measured
    # separately by the runner and reported alongside.
    expected_hot_fact = float(spec.n_fact * fact_weights[:spec.hot_keys].sum())
    expected_hot_dim = float(spec.n_dim * dim_weights[:spec.hot_keys].sum())
    expected_output_rows = expected_hot_fact * (expected_hot_dim / max(1, spec.hot_keys))

    manifest = {
        "spec": spec.__dict__ if not hasattr(spec, "_asdict") else spec._asdict(),
        "slug": spec.slug,
        "root": str(root),
        "fact_path": str(root / "fact_telemetry"),
        "dim_path": str(root / "dim_maintenance"),
        "hot_keys": hot_ids,
        "expected_hot_fact_rows": expected_hot_fact,
        "expected_hot_dim_rows": expected_hot_dim,
        "expected_hot_fact_share": expected_hot_fact / spec.n_fact,
        "expected_hot_dim_share": expected_hot_dim / spec.n_dim,
        "hot_dim_multiplier": spec.hot_dim_multiplier,
        "theoretical_skew_ratio": theoretical_skew_ratio(spec),
        "expected_hot_key_output_rows": expected_output_rows,
        "bytes_on_disk": _dir_bytes(root),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def theoretical_skew_ratio(spec: WorkloadSpec, shuffle_partitions: int = 16) -> float:
    """Hot-key rows divided by the mean rows per shuffle partition.

    This is the dimensionless quantity ``rho`` used by the decision rule. It is a
    property of the data and the partition count alone, so it transfers across
    hardware -- which is precisely why the rule built on it is portable.
    """
    weights = zipf_weights(spec.n_engines, spec.theta)
    hot_rows = spec.n_fact * weights[:spec.hot_keys].sum()
    mean_partition_rows = spec.n_fact / shuffle_partitions
    return float(hot_rows / mean_partition_rows)


def _dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


def summarise(spec: WorkloadSpec) -> Tuple[float, float]:
    """Return (hot-key share of the fact table, Gini coefficient of the key law)."""
    weights = zipf_weights(spec.n_engines, spec.theta)
    share = float(weights[:spec.hot_keys].sum())
    sorted_w = np.sort(weights)
    n = len(sorted_w)
    index = np.arange(1, n + 1)
    gini = float((2 * (index * sorted_w).sum()) / (n * sorted_w.sum()) - (n + 1) / n)
    return share, gini


def probe_weights(spec: WorkloadSpec) -> np.ndarray:
    """Probe-side key probabilities: uniform, with the hot key(s) amplified.

    The hot keys receive ``hot_dim_multiplier`` times the uniform share and the
    remaining mass is spread evenly over the cold keys, so the table size is
    unchanged and only its concentration moves. A multiplier of 1 reproduces
    the uniform case exactly.
    """
    n, h = spec.n_engines, spec.hot_keys
    if h >= n:
        raise ValueError("hot_keys must be smaller than n_engines")
    uniform = 1.0 / n
    hot_share = spec.hot_dim_multiplier * uniform
    if hot_share * h >= 1.0:
        raise ValueError(
            f"hot_dim_multiplier={spec.hot_dim_multiplier} with hot_keys={h} "
            f"claims all probe-side mass; reduce it below {n / h:g}"
        )
    weights = np.full(n, (1.0 - hot_share * h) / (n - h), dtype=np.float64)
    weights[:h] = hot_share
    return weights


def expected_join_output_rows(spec: WorkloadSpec) -> float:
    """Rows the hot key alone contributes to the join output.

    Used as a feasibility guard: a workload whose hot key alone produces more
    output than the machine can process is not measuring skew mitigation, it is
    measuring a cartesian product.
    """
    fact_w = zipf_weights(spec.n_engines, spec.theta)
    probe_w = probe_weights(spec)
    return float(spec.n_fact * fact_w[0] * spec.n_dim * probe_w[0])


def _generate_cmapss(spec: WorkloadSpec, root: Path, manifest_path: Path) -> Dict:
    """Materialise a workload whose payload is real NASA C-MAPSS telemetry.

    The key law is still ours -- see :func:`skewbench.cmapss.to_benchmark_tables`
    for why C-MAPSS keys are near-uniform and cannot supply the skew themselves.
    What is real here is every sensor value, and therefore the column
    distributions, the Parquet compression ratio, and the serialised row width
    that the shuffle actually moves.
    """
    from skewbench import cmapss as cm

    source_dir = Path("data/cmapss")
    frame = cm.load_subset(source_dir, spec.cmapss_subset)
    tables = cm.to_benchmark_tables(
        frame, spec.n_fact, spec.n_dim,
        hot_dim_multiplier=spec.hot_dim_multiplier,
        theta=spec.theta, hot_keys=spec.hot_keys, seed=spec.seed)

    for name, table in tables.items():
        target = root / name
        target.mkdir(parents=True, exist_ok=True)
        for index in range(0, len(table), CHUNK_ROWS):
            table.iloc[index:index + CHUNK_ROWS].to_parquet(
                target / f"part-{index // CHUNK_ROWS:05d}.parquet",
                index=False, compression="snappy")

    fact, dim = tables["fact_telemetry"], tables["dim_maintenance"]
    hot_ids = fact["engine_id"].value_counts().index[:spec.hot_keys].tolist()
    hot_fact = int(fact["engine_id"].isin(hot_ids).sum())
    hot_dim = int(dim["engine_id"].isin(hot_ids).sum())

    manifest = {
        "spec": spec.__dict__,
        "slug": spec.slug,
        "root": str(root),
        "fact_path": str(root / "fact_telemetry"),
        "dim_path": str(root / "dim_maintenance"),
        "hot_keys": hot_ids,
        "source": "cmapss",
        "cmapss_subset": spec.cmapss_subset,
        "cmapss_observed_theta": cm.observed_theta(frame),
        "expected_hot_fact_rows": float(hot_fact),
        "expected_hot_dim_rows": float(hot_dim),
        "expected_hot_fact_share": hot_fact / max(1, len(fact)),
        "expected_hot_dim_share": hot_dim / max(1, len(dim)),
        "hot_dim_multiplier": spec.hot_dim_multiplier,
        "theoretical_skew_ratio": theoretical_skew_ratio(spec),
        "expected_hot_key_output_rows": float(hot_fact) * float(hot_dim) / max(1, spec.hot_keys),
        "bytes_on_disk": _dir_bytes(root),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    return manifest
