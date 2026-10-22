"""Tests for the Monte Carlo draw simulation harness."""

import numpy as np
import pytest
import scipy.stats

from poc_lottery_randomness_auditor.engine.simulate import (
    cyclic_draws,
    fair_draws,
    mc_p_value,
    null_distribution,
    sticky_pair_draws,
    weighted_draws,
)
from poc_lottery_randomness_auditor.games import LOTTO


def test_fair_draws_contract():
    rng = np.random.default_rng(1)
    draws = fair_draws(LOTTO.eras[1], 100, rng)
    assert draws.shape == (100, 6)
    assert draws.min() >= 1 and draws.max() <= 59
    assert all(len(set(row)) == 6 for row in draws)
    assert np.all(np.sort(draws, axis=1) == draws)


def test_seeded_reproducibility():
    a = fair_draws(LOTTO.eras[1], 50, np.random.default_rng(7))
    b = fair_draws(LOTTO.eras[1], 50, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_weighted_draws_detectably_bias():
    weights = np.full(59, 1.0)
    weights[0] = 10.0
    draws = weighted_draws(LOTTO.eras[1], weights, 4000, np.random.default_rng(3))
    p_value = scipy.stats.chisquare(np.bincount(draws.ravel(), minlength=60)[1:]).pvalue
    assert p_value < 1e-6


def test_sticky_pair_boost_forces_pair():
    draws = sticky_pair_draws(LOTTO.eras[1], (1, 2), 1.0, 100, np.random.default_rng(5))
    assert draws.shape == (100, 6)
    assert np.all(np.any(draws == 1, axis=1))
    assert np.all(np.any(draws == 2, axis=1))


def test_cyclic_draws_force_favoured_on_period():
    draws = cyclic_draws(LOTTO.eras[1], 7, 3, 9, np.random.default_rng(11))
    assert draws.shape == (9, 6)
    assert np.all(np.any(draws[[2, 5, 8]] == 7, axis=1))


def test_null_distribution_reproducible():
    def statistic(draws: np.ndarray) -> float:
        return float(draws[:, 0].mean())

    a = null_distribution(statistic, LOTTO.eras[1], 20, 50, np.random.default_rng(13))
    b = null_distribution(statistic, LOTTO.eras[1], 20, 50, np.random.default_rng(13))
    assert a.shape == (50,)
    assert np.array_equal(a, b)
    assert np.isfinite(a).all()


def test_mc_p_value_bounds():
    null = np.arange(1000.0)
    assert mc_p_value(9999.0, null) == pytest.approx(1 / 1001)
    assert mc_p_value(-1.0, null) == pytest.approx(1.0)
