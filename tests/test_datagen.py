"""Tests for the workload generator.

These are correctness tests for the *skew* the generator claims to produce. A
silent bug here would invalidate every downstream measurement while leaving the
pipeline apparently healthy, so the properties are checked directly rather than
inferred from a smoke test.
"""

import numpy as np
import pytest

from skewbench.config import WorkloadSpec
from skewbench.datagen import (engine_id, expected_join_output_rows, probe_weights,
                               summarise, theoretical_skew_ratio, zipf_weights)


def test_zipf_weights_normalised():
    for theta in (0.0, 0.5, 1.0, 1.5, 2.0):
        w = zipf_weights(500, theta)
        assert w.shape == (500,)
        assert np.isclose(w.sum(), 1.0)
        assert np.all(w > 0)


def test_zipf_theta_zero_is_uniform():
    w = zipf_weights(100, 0.0)
    assert np.allclose(w, 1.0 / 100)


def test_zipf_is_monotone_decreasing():
    w = zipf_weights(100, 1.2)
    assert np.all(np.diff(w) <= 0)


def test_hot_share_increases_with_theta():
    shares = [summarise(WorkloadSpec(theta=t, n_engines=2000))[0]
              for t in (0.0, 0.5, 0.9, 1.2, 1.5)]
    assert shares == sorted(shares)
    assert shares[0] < 0.01 < shares[-1]


def test_gini_increases_with_theta():
    ginis = [summarise(WorkloadSpec(theta=t, n_engines=2000))[1]
             for t in (0.0, 0.9, 1.5)]
    assert ginis == sorted(ginis)
    assert ginis[0] == pytest.approx(0.0, abs=1e-6)


def test_skew_ratio_scales_with_partitions():
    spec = WorkloadSpec(theta=1.5, n_engines=2000, n_fact=2_000_000)
    r16 = theoretical_skew_ratio(spec, 16)
    r200 = theoretical_skew_ratio(spec, 200)
    # rho is linear in the partition count by construction.
    assert r200 / r16 == pytest.approx(200 / 16, rel=1e-9)


def test_engine_ids_are_stable_and_padded():
    ids = engine_id(np.array([0, 7, 1234]))
    assert list(ids) == ["ENG-000000", "ENG-000007", "ENG-001234"]


def test_slug_is_deterministic_and_distinguishing():
    a = WorkloadSpec(theta=1.5, hot_dim_multiplier=1)
    b = WorkloadSpec(theta=1.5, hot_dim_multiplier=32)
    assert a.slug == WorkloadSpec(theta=1.5, hot_dim_multiplier=1).slug
    assert a.slug != b.slug


def test_slug_distinguishes_seed_and_source():
    """The slug is the data cache key. If it ignores a field, changing that
    field silently reuses stale data while the results row records the new
    value -- which manufactures variance that was never measured."""
    base = WorkloadSpec(theta=1.2, seed=1)
    assert base.slug != WorkloadSpec(theta=1.2, seed=2).slug
    assert base.slug != WorkloadSpec(theta=1.2, seed=1, source="cmapss").slug


# --- probe-side construction ---------------------------------------------

def test_probe_weights_are_a_distribution():
    for mult in (1.0, 5.0, 32.0):
        w = probe_weights(WorkloadSpec(hot_dim_multiplier=mult))
        assert np.isclose(w.sum(), 1.0)
        assert np.all(w > 0)


def test_multiplier_one_is_exactly_uniform():
    w = probe_weights(WorkloadSpec(hot_dim_multiplier=1.0, n_engines=500))
    assert np.allclose(w, 1.0 / 500)


def test_hot_key_share_scales_with_multiplier():
    n = 2000
    for mult in (2.0, 10.0, 40.0):
        w = probe_weights(WorkloadSpec(hot_dim_multiplier=mult, n_engines=n))
        assert w[0] == pytest.approx(mult / n)


def test_multiplier_that_claims_all_mass_is_rejected():
    with pytest.raises(ValueError):
        probe_weights(WorkloadSpec(hot_dim_multiplier=5000, n_engines=2000))


def test_two_sided_property_tracks_the_multiplier():
    assert not WorkloadSpec(hot_dim_multiplier=1.0).two_sided
    assert WorkloadSpec(hot_dim_multiplier=10.0).two_sided


def test_output_cardinality_is_linear_in_the_multiplier():
    """The guard that motivated the design: doubling the multiplier doubles the
    hot key's output contribution, so the workload stays bounded and steerable."""
    base = expected_join_output_rows(WorkloadSpec(hot_dim_multiplier=1.0))
    tenx = expected_join_output_rows(WorkloadSpec(hot_dim_multiplier=10.0))
    assert tenx / base == pytest.approx(10.0, rel=1e-6)
