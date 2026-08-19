"""Adapter for the NASA C-MAPSS turbofan degradation dataset.

The synthetic generator in :mod:`skewbench.datagen` follows the C-MAPSS schema
deliberately, so that a study can be repeated on real telemetry by swapping the
source without touching the benchmark. This module performs that swap.

C-MAPSS ships as whitespace-delimited text with 26 unnamed columns: unit
number, time in cycles, three operational settings, and 21 sensor
measurements. It is small -- the largest subset, FD004, has roughly 61k rows --
so it is used here as a *key distribution and payload donor* rather than as a
volume source: engine identifiers and sensor rows are tiled up to the requested
scale while preserving the real per-unit row-count distribution.

The dataset is not redistributed with this repository. Download the four
training subsets from the NASA Prognostics Data Repository and point
``root`` at the directory containing ``train_FD001.txt`` and its siblings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

COLUMNS: List[str] = (
    ["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i:02d}" for i in range(1, 22)]
)
SUBSETS = ("FD001", "FD002", "FD003", "FD004")


def load_subset(root: str | Path, subset: str = "FD004",
                split: str = "train") -> pd.DataFrame:
    """Read one C-MAPSS subset into a typed frame."""
    if subset not in SUBSETS:
        raise ValueError(f"unknown subset {subset!r}; expected one of {SUBSETS}")
    path = Path(root) / f"{split}_{subset}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the C-MAPSS turbofan degradation "
            "dataset from the NASA Prognostics Data Repository and point "
            "--cmapss-root at the directory containing train_FD00*.txt."
        )
    frame = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    frame = frame.iloc[:, : len(COLUMNS)]
    frame.columns = COLUMNS
    frame["unit"] = frame["unit"].astype(np.int32)
    frame["cycle"] = frame["cycle"].astype(np.int32)
    return frame


def key_distribution(frame: pd.DataFrame) -> pd.Series:
    """Rows per engine unit -- the real key-frequency law, for comparison to Zipf."""
    return frame.groupby("unit").size().sort_values(ascending=False)


def observed_theta(frame: pd.DataFrame) -> float:
    """Fit a Zipf exponent to the observed key frequencies by log-log regression.

    Reported so that a synthetic ``theta`` can be chosen to match real data
    rather than picked for convenience. C-MAPSS units are near-uniform by
    construction, so the fitted exponent is small; the value is useful as a
    calibration floor, not as evidence of skew.
    """
    counts = key_distribution(frame).to_numpy(dtype=float)
    ranks = np.arange(1, len(counts) + 1, dtype=float)
    slope, _ = np.polyfit(np.log(ranks), np.log(counts), 1)
    return float(-slope)


def remaining_useful_life(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the standard RUL label: cycles remaining until each unit's last cycle.

    Included so that a downstream prediction task can be attached to the
    pipeline, which is how a data-quality or partitioning decision can be scored
    against an accuracy metric rather than against a self-defined one.
    """
    last = frame.groupby("unit")["cycle"].transform("max")
    out = frame.copy()
    out["rul"] = (last - out["cycle"]).astype(np.int32)
    return out


def to_benchmark_tables(frame: pd.DataFrame, n_fact: int, n_dim: int,
                        hot_dim_multiplier: float = 1.0,
                        theta: Optional[float] = None,
                        hot_keys: int = 1,
                        seed: int = 20260818) -> Dict[str, pd.DataFrame]:
    """Project C-MAPSS into the harness's fact and dimension schema.

    Sensor payloads are sampled with replacement from the real rows, so column
    value distributions -- and therefore Parquet compression behaviour and
    shuffle width -- are those of real turbofan telemetry rather than of
    Gaussian noise.

    The join key needs more care, and the reason is a property of the dataset
    worth stating. C-MAPSS units each run to failure, so their cycle counts are
    comparable by construction: fitted Zipf exponents across the four subsets
    are 0.21 to 0.33 and Gini coefficients 0.12 to 0.18, i.e. **essentially
    unskewed**. C-MAPSS therefore cannot serve as a skewed workload on its own.
    That is not a defect of the dataset; it reflects a controlled run-to-failure
    experiment rather than an operating fleet, where a few problem units
    accumulate disproportionate events and maintenance history.

    Two modes follow. With ``theta=None`` the empirical key law is used, which
    gives a realistic but near-uniform control. With ``theta`` supplied, a
    truncated Zipf law is imposed over the same real unit identifiers while
    payloads stay real -- the honest combination of real data and controlled
    skew, and the one used for the skew experiments.
    """
    rng = np.random.default_rng(seed)
    counts = key_distribution(frame)
    units = counts.index.to_numpy()
    if theta is None:
        weights = counts.to_numpy(dtype=float)
        weights = weights / weights.sum()
    else:
        ranks = np.arange(1, len(units) + 1, dtype=float)
        weights = ranks ** (-float(theta))
        weights = weights / weights.sum()

    fact_units = rng.choice(units, size=n_fact, p=weights)
    donors = rng.integers(0, len(frame), size=n_fact)
    fact = frame.iloc[donors].reset_index(drop=True)
    fact["engine_id"] = np.char.add("ENG-", np.char.zfill(fact_units.astype(str), 6))
    fact["event_id"] = np.arange(n_fact, dtype=np.int64)
    fact["flight_phase"] = rng.choice(
        ["takeoff", "climb", "cruise", "descent", "landing"], size=n_fact)
    fact = fact.drop(columns=["unit"])

    # Probe side: uniform except for the hot key(s), amplified by the
    # multiplier. Mirrors skewbench.datagen.probe_weights so that synthetic and
    # C-MAPSS-backed workloads are directly comparable.
    n_units = len(units)
    uniform = 1.0 / n_units
    hot_share = hot_dim_multiplier * uniform
    if hot_share * hot_keys >= 1.0:
        raise ValueError(
            f"hot_dim_multiplier={hot_dim_multiplier} with hot_keys={hot_keys} "
            f"claims all probe-side mass; reduce it below {n_units / hot_keys:g}"
        )
    dim_weights = np.full(n_units, (1.0 - hot_share * hot_keys) / (n_units - hot_keys))
    dim_weights[:hot_keys] = hot_share
    dim_units = rng.choice(units, size=n_dim, p=dim_weights)
    dim = pd.DataFrame({
        "wo_id": np.arange(n_dim, dtype=np.int64),
        "engine_id": np.char.add("ENG-", np.char.zfill(dim_units.astype(str), 6)),
        "wo_ts": (1_750_000_000 + rng.integers(0, 31_536_000, n_dim)).astype(np.int64),
        "shop_visit": rng.integers(1, 12, n_dim, dtype=np.int32),
        "part_no": np.char.add("P-", np.char.zfill(
            rng.integers(0, 50_000, n_dim).astype(str), 5)),
        "finding_code": rng.choice(
            ["FC-BLADE", "FC-SEAL", "FC-BEARING", "FC-FUEL", "FC-VIBR",
             "FC-EGT", "FC-NONE"], size=n_dim),
        "labour_hours": np.round(rng.gamma(2.0, 3.0, n_dim), 2).astype(np.float32),
    })
    return {"fact_telemetry": fact, "dim_maintenance": dim}
