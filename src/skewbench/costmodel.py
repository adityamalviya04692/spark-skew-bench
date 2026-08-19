r"""Analytical salt-cardinality cost model, and the AQE-sufficiency rule.

The model
---------
Salting a hot key across ``k`` sub-keys changes two things at once, in opposite
directions. Folklore names only the first.

1. **Straggler relief.** The hot key contributes ``H`` rows to the skewed side.
   Unsalted, they land in one shuffle partition and one task must process all of
   them. Salted, they spread over ``k`` partitions, so the critical task sees
   ``H / k`` rows. This term falls as ``1 / k``.

2. **Replication cost.** Every sub-key must still meet its partners, so the
   ``P`` matching rows on the probe side are replicated once per salt value. The
   join therefore shuffles ``P * (k - 1)`` rows it did not shuffle before. This
   term rises linearly in ``k``.

Writing ``a`` for the unit cost of processing a row on the critical task and
``b`` for the unit cost of shuffling a replicated probe row, total time is

.. math::

    T(k) = a \frac{H}{k} + b P (k - 1) + c

with ``c`` collecting everything independent of ``k`` -- scanning, the cold-key
join, the final aggregation. ``T`` is convex on ``k > 0``, so it has a single
interior minimum, found by setting the derivative to zero:

.. math::

    \frac{dT}{dk} = -\frac{aH}{k^{2}} + bP = 0
    \qquad\Longrightarrow\qquad
    k^{*} = \sqrt{\frac{aH}{bP}} = \sqrt{\gamma \frac{H}{P}}

where ``gamma = a / b`` is the only quantity that must be calibrated, and it is a
property of the engine and hardware rather than of the workload. Calibrate it
once; reuse it across workloads.

Two ceilings bound the useful range
-----------------------------------
The closed form above is unconstrained, and taken literally it will sometimes
recommend a ``k`` that buys nothing.

*Saturation.* Once the hot partition has been divided below the size of an
ordinary partition, the hot task is no longer the straggler and further division
relieves nothing. That happens at ``k = rho``, where ``rho = H / (N / p)`` is the
hot key's rows over the mean rows per partition. **Salting beyond ``rho`` is
always waste** -- a bound that follows immediately from the model and that, as
far as we can find, appears nowhere in the practitioner literature.

*Parallelism.* Sub-partitions beyond the number of concurrently schedulable task
slots ``C`` queue rather than overlap, so the straggler term stops falling.

The operational rule is therefore ``k_eff = clamp(min(k*, rho, C), 1, k_max)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------
# Cost model
# --------------------------------------------------------------------------

@dataclass
class CostParams:
    """Fitted coefficients of ``T(k) = a H / k + b P (k - 1) + c``.

    Standard errors are carried alongside the point estimates because the
    replication coefficient ``b`` is the entire denominator of ``k*``: if ``b``
    is statistically indistinguishable from zero, ``k*`` is unbounded and
    reporting a finite value for it is reporting noise as a measurement.
    """

    a: float
    b: float
    c: float
    r_squared: float = float("nan")
    n_points: int = 0
    se_a: float = float("nan")
    se_b: float = float("nan")
    se_c: float = float("nan")

    @property
    def gamma(self) -> float:
        """``a / b`` -- the engine-and-hardware constant that sets ``k*``."""
        return self.a / self.b if self.b > 0 else float("inf")

    @property
    def b_t_statistic(self) -> float:
        """``|b| / se(b)``. Below about 2, ``b`` is not resolved by the data."""
        if not (self.se_b > 0):
            return float("nan")
        return abs(self.b) / self.se_b

    @property
    def identified(self) -> bool:
        """Whether the fit resolves the parameters ``k*`` depends on."""
        return bool(self.a > 0 and self.b > 0 and self.b_t_statistic >= 2.0)

    def diagnosis(self) -> str:
        if self.a <= 0:
            return "not identified: fitted a <= 0 (no straggler term resolved)"
        if self.b <= 0:
            return "not identified: fitted b <= 0 (no replication cost resolved)"
        if self.b_t_statistic < 2.0:
            return (f"not identified: |b|/se(b) = {self.b_t_statistic:.2f} < 2, "
                    "so the replication coefficient is indistinguishable from zero")
        return f"identified: |b|/se(b) = {self.b_t_statistic:.2f}"


def predict_time(k: np.ndarray | float, H: float, P: float,
                 params: CostParams) -> np.ndarray | float:
    """Evaluate ``T(k)`` for the given workload quantities."""
    k_arr = np.asarray(k, dtype=float)
    return params.a * H / k_arr + params.b * P * (k_arr - 1.0) + params.c


def fit(ks: Sequence[float], times: Sequence[float], H: float, P: float) -> CostParams:
    """Least-squares fit of the three coefficients.

    ``T`` is linear in ``(a, b, c)`` given the basis ``[H/k, P(k-1), 1]``, so this
    is an ordinary linear least-squares problem -- no initial guess, no local
    minima, no optimiser tuning. That matters for reproducibility: the fit is a
    deterministic function of the measurements.
    """
    k_arr = np.asarray(ks, dtype=float)
    t_arr = np.asarray(times, dtype=float)
    if k_arr.size < 3:
        raise ValueError("need at least three distinct k values to fit three coefficients")
    if np.any(k_arr <= 0):
        raise ValueError("salt cardinality must be positive")

    design = np.column_stack([H / k_arr, P * (k_arr - 1.0), np.ones_like(k_arr)])
    coeffs, *_ = np.linalg.lstsq(design, t_arr, rcond=None)
    a, b, c = (float(v) for v in coeffs)

    residual = t_arr - design @ coeffs
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((t_arr - t_arr.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Standard errors from the OLS covariance matrix. Without these the fit
    # reports a number for k* whether or not the data support one.
    dof = k_arr.size - design.shape[1]
    se_a = se_b = se_c = float("nan")
    if dof > 0:
        sigma_squared = ss_res / dof
        try:
            covariance = sigma_squared * np.linalg.inv(design.T @ design)
            se_a, se_b, se_c = (float(np.sqrt(abs(covariance[i, i]))) for i in range(3))
        except np.linalg.LinAlgError:  # pragma: no cover - singular design
            pass

    return CostParams(a=a, b=b, c=c, r_squared=r2, n_points=int(k_arr.size),
                      se_a=se_a, se_b=se_b, se_c=se_c)


def optimal_k_unconstrained(H: float, P: float, params: CostParams,
                            strict: bool = False) -> float:
    """The closed form ``k* = sqrt(a H / (b P))``.

    Returns NaN rather than a finite-looking number when the fit does not
    identify the parameters. An earlier version silently returned ``inf`` for
    ``b <= 0`` and ``1.0`` for ``a <= 0`` -- both are the model being falsified,
    dressed up as an answer.
    """
    if strict and not params.identified:
        return float("nan")
    if params.b <= 0 or P <= 0 or params.a <= 0:
        return float("nan")
    value = (params.a * H) / (params.b * P)
    return math.sqrt(value) if value > 0 else float("nan")


def saturation_k(H: float, n_rows: int, shuffle_partitions: int,
                 account_for_collisions: bool = True) -> float:
    """``rho``: the point beyond which splitting the hot key relieves nothing.

    The naive form ``rho = H / (N/p)`` assumes the ``k`` salted sub-keys land in
    ``k`` distinct shuffle partitions. They do not: ``(key, salt)`` pairs are
    hashed into the same ``p`` partitions and collide, so the number of
    partitions actually reached by ``k`` salts is approximately
    ``p (1 - (1 - 1/p)^k)``. Ignoring this made the bound too tight, and the
    measured optimum exceeded it -- salting past the naive ``rho`` kept helping.

    With ``account_for_collisions`` we invert that relation, returning the ``k``
    at which the expected number of distinct partitions reached equals the naive
    bound. This is the honest version of the ceiling.
    """
    mean_partition_rows = n_rows / max(1, shuffle_partitions)
    naive = H / mean_partition_rows if mean_partition_rows > 0 else 1.0
    if not account_for_collisions:
        return naive

    p = max(1, int(shuffle_partitions))
    target = min(naive, float(p) - 1e-9)
    if target <= 1.0:
        return max(1.0, naive)
    # Solve p (1 - (1 - 1/p)^k) = target for k.
    ratio = 1.0 - target / p
    if ratio <= 0.0:
        return float(p)
    return math.log(ratio) / math.log(1.0 - 1.0 / p)


@dataclass
class KRecommendation:
    """A salt cardinality, with every bound that shaped it made explicit."""

    k_star_unconstrained: float
    k_saturation: float
    k_parallelism: int
    k_recommended: int
    binding_constraint: str
    predicted_time: Optional[float] = None

    def explain(self) -> str:
        return (
            f"k* (unconstrained) = {self.k_star_unconstrained:.2f}; "
            f"saturation bound rho = {self.k_saturation:.2f}; "
            f"parallelism bound C = {self.k_parallelism}; "
            f"recommended k = {self.k_recommended} "
            f"(binding: {self.binding_constraint})"
        )


def recommend_k(H: float, P: float, n_rows: int, shuffle_partitions: int,
                cores: int, params: CostParams, k_max: int = 256) -> KRecommendation:
    """Apply the closed form and both ceilings."""
    k_star = optimal_k_unconstrained(H, P, params)
    k_sat = saturation_k(H, n_rows, shuffle_partitions)
    k_par = max(1, int(cores))

    candidates: Dict[str, float] = {
        "saturation": k_sat,
        "parallelism": float(k_par),
    }
    # An unidentified fit contributes no cost-optimum candidate at all, rather
    # than contributing a spurious one.
    if k_star == k_star:  # not NaN
        candidates["cost-optimum"] = k_star
    binding = min(candidates, key=lambda name: candidates[name])
    k_eff = int(max(1, min(k_max, round(candidates[binding]))))

    return KRecommendation(
        k_star_unconstrained=float(k_star),
        k_saturation=float(k_sat),
        k_parallelism=k_par,
        k_recommended=k_eff,
        binding_constraint=binding,
        predicted_time=float(predict_time(k_eff, H, P, params)),
    )


# --------------------------------------------------------------------------
# AQE sufficiency rule
# --------------------------------------------------------------------------

@dataclass
class AQEDecision:
    """Whether AQE alone is expected to handle this join, and why."""

    aqe_triggers: bool
    aqe_sufficient: bool
    recommendation: str
    rho: float
    hot_partition_bytes: float
    median_partition_bytes: float
    trigger_threshold_bytes: float
    reasons: List[str] = field(default_factory=list)

    def explain(self) -> str:
        head = f"decision: {self.recommendation}"
        return head + "\n  - " + "\n  - ".join(self.reasons)


def _parse_bytes(text: str | int) -> float:
    """Accept Spark-style size strings such as ``'256MB'`` or a raw byte count."""
    if isinstance(text, (int, float)):
        return float(text)
    units = {"B": 1, "K": 1e3, "KB": 1e3, "M": 1e6, "MB": 1e6,
             "G": 1e9, "GB": 1e9, "T": 1e12, "TB": 1e12}
    cleaned = str(text).strip().upper()
    for suffix in sorted(units, key=len, reverse=True):
        if cleaned.endswith(suffix):
            return float(cleaned[: -len(suffix)]) * units[suffix]
    return float(cleaned)


def decide(H: float, P: float, n_fact: int, row_bytes: float,
           shuffle_partitions: int, two_sided: bool,
           skewed_partition_factor: int = 5,
           skewed_partition_threshold: str | int = "256MB",
           advisory_partition_size: str | int = "64MB",
           rho_floor: float = 2.0) -> AQEDecision:
    """The decision procedure this study proposes.

    The logic mirrors what Spark actually does, and the rule has *three*
    conjunctive preconditions rather than the two that are usually documented.

    1. The partition must exceed an absolute byte threshold
       (``skewedPartitionThresholdInBytes``).
    2. It must exceed ``skewedPartitionFactor`` times the median partition size.
    3. It must exceed ``advisoryPartitionSizeInBytes`` -- because that is the
       size the rule splits *into*. A hot partition smaller than the advisory
       size cannot be divided into more than one piece, so the optimisation is a
       no-op however skewed the partition is relative to its peers.

    The third condition is the one practitioners do not know about, and in our
    measurements it was the binding one: with a 7.6 MB hot partition and the
    default advisory size, the rule never fired on a join whose task-time skew
    ratio was 100:1. Lowering the advisory size below the hot partition made it
    fire immediately, with no other change.

    Beyond triggering, the remaining question is whether firing is *enough*. The
    hinge there is the opposite side: when it is also concentrated on the same
    key, each split inherits a large partner, probe-side work multiplies with
    the split count, and relief on the critical path is substantially cancelled.
    """
    reasons: List[str] = []
    rho = saturation_k(H, n_fact, shuffle_partitions)
    hot_bytes = H * row_bytes
    cold_rows = max(0.0, n_fact - H)
    median_bytes = (cold_rows / max(1, shuffle_partitions - 1)) * row_bytes
    threshold = max(_parse_bytes(skewed_partition_threshold),
                    skewed_partition_factor * median_bytes)

    if rho < rho_floor:
        reasons.append(
            f"rho = {rho:.2f} < {rho_floor}: the hot key is smaller than a couple of "
            f"ordinary partitions, so there is no straggler to remove."
        )
        return AQEDecision(False, True, "do nothing", rho, hot_bytes, median_bytes,
                           threshold, reasons)

    advisory = _parse_bytes(advisory_partition_size)
    reasons.append(
        f"hot partition is approximately {hot_bytes/1e6:.1f} MB against a trigger "
        f"threshold of {threshold/1e6:.1f} MB "
        f"(max of the absolute threshold and {skewed_partition_factor}x the "
        f"{median_bytes/1e6:.1f} MB median), and an advisory split size of "
        f"{advisory/1e6:.1f} MB."
    )

    splittable = hot_bytes > advisory
    triggers = hot_bytes > threshold and splittable

    if not splittable:
        reasons.append(
            f"the hot partition ({hot_bytes/1e6:.1f} MB) is smaller than "
            f"advisoryPartitionSizeInBytes ({advisory/1e6:.1f} MB), so there is "
            "no split to make and the rule is a no-op regardless of how skewed "
            "the partition is. This precondition is rarely documented and is "
            "frequently the binding one. Lower the advisory size below the hot "
            "partition, or salt explicitly."
        )
        return AQEDecision(False, False,
                           "salt (AQE cannot split: advisory size too large)",
                           rho, hot_bytes, median_bytes, threshold, reasons)

    if not triggers:
        reasons.append(
            "AQE will not classify this partition as skewed, so skew-join "
            "optimisation never fires. Either lower "
            "skewedPartitionThresholdInBytes or salt explicitly."
        )
        return AQEDecision(False, False, "salt (AQE will not trigger)", rho,
                           hot_bytes, median_bytes, threshold, reasons)

    if two_sided:
        reasons.append(
            "both sides are concentrated on the same key. AQE splits the skewed "
            "partition and replicates the matching partition to every split, so "
            "the probe-side work multiplies with the split count and the "
            "straggler is only partly relieved."
        )
        return AQEDecision(True, False, "salt in addition to AQE", rho, hot_bytes,
                           median_bytes, threshold, reasons)

    reasons.append(
        "skew is one-sided and above the trigger, which is the case "
        "OptimizeSkewedJoin is designed for. Manual salting adds replication "
        "cost for little further gain."
    )
    return AQEDecision(True, True, "AQE alone is sufficient", rho, hot_bytes,
                       median_bytes, threshold, reasons)
