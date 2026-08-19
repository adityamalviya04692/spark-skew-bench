"""Tests for the cost model and the decision procedure.

The model is the paper's theoretical claim, so these tests check the
mathematics -- convexity, exact parameter recovery, the closed form, and the
behaviour of each ceiling -- rather than merely that the functions run.
"""

import math

import numpy as np
import pytest

from skewbench.costmodel import (AQEDecision, CostParams, decide, fit,
                                 optimal_k_unconstrained, predict_time,
                                 recommend_k, saturation_k)

H, P = 400_000.0, 2_000.0
TRUTH = CostParams(a=2e-4, b=5e-4, c=10.0)


def test_fit_recovers_known_parameters_exactly():
    ks = np.array([1, 2, 4, 8, 16, 32, 64], float)
    times = predict_time(ks, H, P, TRUTH)
    fitted = fit(ks, times, H, P)
    assert fitted.a == pytest.approx(TRUTH.a, rel=1e-6)
    assert fitted.b == pytest.approx(TRUTH.b, rel=1e-6)
    assert fitted.c == pytest.approx(TRUTH.c, rel=1e-6)
    assert fitted.r_squared == pytest.approx(1.0, abs=1e-9)


def test_closed_form_matches_numerical_minimum():
    analytic = optimal_k_unconstrained(H, P, TRUTH)
    grid = np.linspace(1, 200, 20_000)
    numeric = grid[int(np.argmin(predict_time(grid, H, P, TRUTH)))]
    assert analytic == pytest.approx(numeric, rel=1e-3)


def test_cost_is_convex_in_k():
    ks = np.linspace(1, 100, 500)
    t = np.asarray(predict_time(ks, H, P, TRUTH))
    assert np.all(np.diff(t, 2) > -1e-9)  # non-negative second difference


def test_kstar_scales_as_sqrt_of_skew():
    """Doubling H multiplies k* by sqrt(2), not by 2. This is the paper's
    counter-intuitive structural claim; it must hold exactly."""
    base = optimal_k_unconstrained(H, P, TRUTH)
    doubled = optimal_k_unconstrained(2 * H, P, TRUTH)
    assert doubled / base == pytest.approx(math.sqrt(2), rel=1e-9)


def test_saturation_bound_definition():
    # Naive form: hot-key rows over mean rows per partition.
    assert saturation_k(400_000, 2_000_000, 16,
                        account_for_collisions=False) == pytest.approx(3.2)
    # Collision-aware form is looser, because k salts reach fewer than k
    # distinct partitions.
    assert saturation_k(400_000, 2_000_000, 16) == pytest.approx(3.458, abs=0.01)


def test_parallelism_ceiling_binds_when_smallest():
    rec = recommend_k(H, P, n_rows=2_000_000, shuffle_partitions=200,
                      cores=2, params=TRUTH)
    assert rec.binding_constraint == "parallelism"
    assert rec.k_recommended == 2


def test_saturation_ceiling_binds_when_smallest():
    rec = recommend_k(H, P, n_rows=2_000_000, shuffle_partitions=8,
                      cores=64, params=TRUTH)
    assert rec.binding_constraint == "saturation"


def test_cost_optimum_binds_when_smallest():
    # Large probe-side hot key makes replication expensive, so k* is small.
    rec = recommend_k(H, P=200_000.0, n_rows=2_000_000,
                      shuffle_partitions=200, cores=64, params=TRUTH)
    assert rec.binding_constraint == "cost-optimum"


def test_recommended_k_never_below_one():
    rec = recommend_k(H=10.0, P=1e9, n_rows=2_000_000,
                      shuffle_partitions=16, cores=8, params=TRUTH)
    assert rec.k_recommended >= 1


def test_fit_reports_standard_errors():
    ks = np.array([1, 2, 4, 8, 16, 32, 64], float)
    times = predict_time(ks, H, P, TRUTH)
    fitted = fit(ks, times, H, P)
    # A noiseless fit has essentially zero residual, hence ~zero standard error.
    assert fitted.se_b == pytest.approx(0.0, abs=1e-6)
    assert fitted.identified


def test_unidentified_fit_refuses_to_report_kstar():
    """The guard that matters: when the replication coefficient is not resolved,
    k* is unbounded and must not be reported as a number."""
    rng = np.random.default_rng(0)
    ks = np.array([1, 2, 4, 8, 16, 32, 64], float)
    # Times that fall then stay flat: no replication cost is observable.
    times = 10.0 + 40.0 / ks + rng.normal(0, 0.4, ks.size)
    fitted = fit(ks, times, H, P)
    assert not fitted.identified
    assert math.isnan(optimal_k_unconstrained(H, P, fitted, strict=True))
    assert "not identified" in fitted.diagnosis()


def test_negative_coefficients_yield_nan_not_a_number():
    bad = CostParams(a=-1.0, b=1e-4, c=0.0, se_b=1e-6)
    assert math.isnan(optimal_k_unconstrained(H, P, bad))
    worse = CostParams(a=1e-4, b=-1.0, c=0.0, se_b=1e-6)
    assert math.isnan(optimal_k_unconstrained(H, P, worse))


def test_saturation_accounts_for_hash_collisions():
    """k salted sub-keys do not reach k distinct partitions -- they collide."""
    naive = saturation_k(445_743, 2_000_000, 16, account_for_collisions=False)
    aware = saturation_k(445_743, 2_000_000, 16, account_for_collisions=True)
    assert aware > naive, "collision-aware bound must be looser than the naive one"
    assert aware == pytest.approx(3.91, abs=0.05)


def test_unidentified_fit_does_not_bind_the_recommendation():
    unidentified = CostParams(a=1e-4, b=1e-9, c=0.0, se_b=1.0)
    rec = recommend_k(H, P, n_rows=2_000_000, shuffle_partitions=16,
                      cores=8, params=unidentified)
    assert rec.binding_constraint in ("saturation", "parallelism")


def test_fit_requires_three_points():
    with pytest.raises(ValueError):
        fit([1, 2], [10.0, 5.0], H, P)


# --- decision procedure ---------------------------------------------------

def _decide(**kw):
    # advisory_partition_size must sit below the hot partition, or the rule
    # cannot split and every case collapses to the same answer. That is itself
    # a finding (see test_advisory_size_blocks_the_rule), so the default here is
    # chosen to exercise the *other* branches.
    base = dict(H=800_000, P=2_000, n_fact=2_000_000, row_bytes=80,
                shuffle_partitions=16, two_sided=False,
                skewed_partition_threshold="16MB",
                advisory_partition_size="8MB")
    base.update(kw)
    return decide(**base)


def test_no_skew_returns_do_nothing():
    d = _decide(H=1_000)
    assert d.recommendation == "do nothing"
    assert d.aqe_sufficient


def test_one_sided_above_threshold_prefers_aqe_alone():
    d = _decide(two_sided=False)
    assert d.aqe_triggers and d.aqe_sufficient
    assert d.recommendation == "AQE alone is sufficient"


def test_two_sided_requires_salting():
    d = _decide(two_sided=True)
    assert d.aqe_triggers and not d.aqe_sufficient
    assert "salt" in d.recommendation


def test_below_absolute_threshold_aqe_never_fires():
    d = _decide(skewed_partition_threshold="10GB")
    assert not d.aqe_triggers
    assert "will not trigger" in d.recommendation


def test_advisory_size_blocks_the_rule():
    """The third, rarely-documented precondition.

    A hot partition smaller than advisoryPartitionSizeInBytes cannot be split
    into more than one piece, so OptimizeSkewedJoin is a no-op however skewed
    the partition is. This was the binding condition in our measurements.
    """
    d = _decide(advisory_partition_size="1GB")
    assert not d.aqe_triggers
    assert "cannot split" in d.recommendation
    assert any("advisoryPartitionSizeInBytes" in r for r in d.reasons)


def test_lowering_advisory_size_unblocks_the_rule():
    blocked = _decide(advisory_partition_size="1GB")
    unblocked = _decide(advisory_partition_size="1MB")
    assert not blocked.aqe_triggers
    assert unblocked.aqe_triggers and unblocked.aqe_sufficient


def test_decision_always_explains_itself():
    for kwargs in ({}, {"two_sided": True}, {"H": 1_000},
                   {"skewed_partition_threshold": "10GB"},
                   {"advisory_partition_size": "1GB"}):
        d = _decide(**kwargs)
        assert isinstance(d, AQEDecision)
        assert d.reasons and all(isinstance(r, str) for r in d.reasons)
        assert d.explain().startswith("decision:")
