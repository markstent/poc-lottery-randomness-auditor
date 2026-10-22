"""Tests for the Bayesian uniformity layer."""

from datetime import date

import numpy as np
import pandas as pd
import pytest
import scipy.stats

from poc_lottery_randomness_auditor.data import COLUMNS
from poc_lottery_randomness_auditor.engine.bayesian import (
    ROPE_DELTA,
    bayesian_uniformity,
    certified_delta,
    posterior_intervals,
)
from poc_lottery_randomness_auditor.engine.simulate import fair_draws
from poc_lottery_randomness_auditor.games import LOTTO


def draws_to_df(draws: np.ndarray) -> pd.DataFrame:
    rows = []
    for offset, draw in enumerate(draws):
        rows.append([offset + 1, date(2020, 1, 1), 1, *draw.tolist(), 1, "M", "L1"])
    return pd.DataFrame(rows, columns=list(COLUMNS))


def test_posterior_interval_matches_scipy():
    counts = np.array([5, 5, 5, 5])
    lower, upper = posterior_intervals(counts, 0.95)[0]
    reference = scipy.stats.beta(1 + 5, 1 + 20 - 5)
    assert lower == pytest.approx(reference.ppf(0.025))
    assert upper == pytest.approx(reference.ppf(0.975))


def test_certified_delta_small_for_huge_uniform_sample():
    counts = np.full(59, 100_000)
    assert certified_delta(counts) < ROPE_DELTA


def test_certified_delta_large_for_skewed_sample():
    counts = np.full(59, 100)
    counts[0] = 400
    assert certified_delta(counts) > 0.5


def test_bayesian_uniformity_result_contract():
    draws = fair_draws(LOTTO.eras[1], 2000, np.random.default_rng(51))
    result = bayesian_uniformity(draws_to_df(draws), LOTTO.eras[1])
    assert result.method == "bayesian"
    assert result.p_value is None
    assert result.details["rope_delta"] == ROPE_DELTA
    assert len(result.details["intervals"]) == 59
