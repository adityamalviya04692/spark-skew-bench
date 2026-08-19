"""Experiment specifications.

Every knob that can change a measured outcome lives in one of these dataclasses,
and every result record carries the full specification that produced it. This is
what makes a results file self-describing and a run reproducible from the file
alone.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# The six strategies under evaluation.
ARMS = (
    "baseline",        # sort-merge join, AQE disabled, no mitigation
    "aqe",             # AQE enabled with skew-join optimisation
    "salt_uniform",    # manual salting, every key salted, AQE disabled
    "salt_selective",  # manual salting, hot keys only, AQE disabled
    "aqe_salt",        # AQE enabled *and* selective salting
    "broadcast",       # broadcast hash join
)


@dataclass(frozen=True)
class WorkloadSpec:
    """Shape of the synthetic workload.

    Attributes
    ----------
    n_fact : rows in the skewed fact table (engine telemetry).
    n_dim : rows in the probe table (maintenance work orders).
    n_engines : number of distinct join keys.
    theta : Zipf exponent. 0.0 is uniform; higher is more skewed.
    hot_dim_multiplier : how many times the mean per-key row count the hot key
        carries on the *probe* side. 1.0 is one-sided skew (a uniform probe
        side); larger values make the skew increasingly two-sided, which is the
        case AQE's split-and-replicate cannot fix.

        This is a continuous knob rather than a boolean for two reasons. It
        bounds the join's output cardinality, which a doubly-Zipf design does
        not -- with both sides drawn from the same law the hot key alone yields
        H x P output rows, order 10^10 at the scales here, which is a
        cardinality explosion rather than a benchmark. And it lets the probe-side
        hot-key count P be swept directly, which is what tests the cost model's
        prediction that the optimum falls as the inverse square root of P.
    hot_keys : number of keys treated as "hot" by selective salting.
    source : ``"synthetic"`` generates payloads from parameterised
        distributions; ``"cmapss"`` samples real NASA turbofan telemetry rows
        for the payload while imposing the requested key law, so that column
        value distributions, compression behaviour and shuffle width are those
        of real sensor data.
    n_sensors : payload width, i.e. sensor columns on the fact table.
    seed : RNG seed. Generation is deterministic given the seed.
    """

    n_fact: int = 2_000_000
    n_dim: int = 20_000
    n_engines: int = 2_000
    theta: float = 1.2
    hot_dim_multiplier: float = 1.0
    hot_keys: int = 1
    n_sensors: int = 12
    source: str = "synthetic"   # "synthetic" | "cmapss"
    cmapss_subset: str = "FD004"
    seed: int = 20260818

    @property
    def two_sided(self) -> bool:
        """True when the probe side is materially over-represented on the hot key."""
        return self.hot_dim_multiplier > 1.5

    @property
    def mean_dim_rows_per_key(self) -> float:
        return self.n_dim / self.n_engines

    @property
    def expected_hot_dim_rows(self) -> float:
        """P: the probe-side rows carrying the hot key(s)."""
        return self.hot_dim_multiplier * self.mean_dim_rows_per_key * self.hot_keys

    @property
    def slug(self) -> str:
        """Short, filesystem-safe, collision-resistant identifier for this workload."""
        prefix = "cm" if self.source == "cmapss" else "f"
        tail = f"_{self.cmapss_subset}" if self.source == "cmapss" else ""
        # The seed is part of the identity. generate() returns a cached manifest
        # when one exists for the slug, so a slug that ignores the seed makes a
        # seed sweep a silent no-op while every results row still records the
        # requested seed -- fabricated variance, and hard to notice.
        return (
            f"{prefix}{self.n_fact//1000}k_d{self.n_dim//1000}k_e{self.n_engines}"
            f"_th{self.theta:g}_m{self.hot_dim_multiplier:g}"
            f"_hk{self.hot_keys}_s{self.n_sensors}{tail}_sd{self.seed}"
        )


@dataclass(frozen=True)
class SparkSpec:
    """Spark runtime configuration for one measurement."""

    master: str = "local[2]"
    driver_memory: str = "4g"
    shuffle_partitions: int = 16
    aqe_enabled: bool = False
    aqe_skew_join_enabled: bool = True
    aqe_coalesce_enabled: bool = True
    skewed_partition_factor: int = 5
    skewed_partition_threshold: str = "16MB"
    advisory_partition_size: str = "16MB"
    autobroadcast_threshold: int = -1  # -1 disables automatic broadcast
    local_dir: Optional[str] = None

    @property
    def cores(self) -> int:
        """Number of local cores requested, parsed from the master URL."""
        if self.master.startswith("local["):
            inner = self.master[len("local[") : -1]
            if inner in ("*", ""):
                import os

                return os.cpu_count() or 1
            return int(inner)
        return 0  # unknown for cluster masters


@dataclass(frozen=True)
class ArmSpec:
    """One mitigation strategy, with its salt cardinality where applicable."""

    name: str
    k: int = 1

    def __post_init__(self) -> None:
        if self.name not in ARMS:
            raise ValueError(f"unknown arm {self.name!r}; expected one of {ARMS}")
        if self.k < 1:
            raise ValueError("salt cardinality k must be >= 1")
        if self.name in ("baseline", "aqe", "broadcast") and self.k != 1:
            raise ValueError(f"arm {self.name!r} does not take a salt cardinality")

    @property
    def salts(self) -> bool:
        return self.name in ("salt_uniform", "salt_selective", "aqe_salt")

    @property
    def label(self) -> str:
        return f"{self.name}(k={self.k})" if self.salts else self.name


@dataclass
class RunSpec:
    """A complete, reproducible measurement request."""

    workload: WorkloadSpec = field(default_factory=WorkloadSpec)
    spark: SparkSpec = field(default_factory=SparkSpec)
    arm: ArmSpec = field(default_factory=lambda: ArmSpec("baseline"))
    repetitions: int = 5
    warmup: int = 1
    notes: str = ""

    @property
    def run_id(self) -> str:
        """Stable hash over the full specification, used as the Spark job group id."""
        payload = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha1(payload).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workload": asdict(self.workload),
            "spark": asdict(self.spark),
            "arm": asdict(self.arm),
            "repetitions": self.repetitions,
            "warmup": self.warmup,
            "notes": self.notes,
        }


def flatten(spec: RunSpec) -> Dict[str, Any]:
    """Flatten a RunSpec into a single dict suitable for a results row."""
    out: Dict[str, Any] = {"run_id": spec.run_id}
    for group, values in (
        ("wl", asdict(spec.workload)),
        ("sp", asdict(spec.spark)),
        ("arm", asdict(spec.arm)),
    ):
        for key, value in values.items():
            out[f"{group}_{key}"] = value
    out["repetitions"] = spec.repetitions
    out["warmup"] = spec.warmup
    return out


def expand_grid(
    workloads: List[WorkloadSpec],
    sparks: List[SparkSpec],
    arms: List[ArmSpec],
    repetitions: int = 5,
    warmup: int = 1,
) -> List[RunSpec]:
    """Cartesian product of the three axes, skipping incoherent combinations.

    An arm that enables AQE is only meaningful against a SparkSpec that also
    enables it; pairing them the other way would silently mislabel the result.
    """
    runs: List[RunSpec] = []
    for wl in workloads:
        for sp in sparks:
            for arm in arms:
                arm_wants_aqe = arm.name in ("aqe", "aqe_salt")
                if arm_wants_aqe != sp.aqe_enabled:
                    continue
                runs.append(
                    RunSpec(workload=wl, spark=sp, arm=arm,
                            repetitions=repetitions, warmup=warmup)
                )
    return runs
