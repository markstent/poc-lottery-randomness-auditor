"""Tests for frequency uniformity and homogeneity checks."""

from datetime import date

import numpy as np
import pandas as pd
import pytest
import scipy.stats

from poc_lottery_randomness_auditor.data import COLUMNS
from poc_lottery_randomness_auditor.engine.frequencies import (
    chi2_uniform,
    frequency_test,
    homogeneity_test,
)
from poc_lottery_randomness_auditor.engine.simulate import fair_draws, weighted_draws
from poc_lottery_randomness_auditor.games import LOTTO


def draws_to_df(draws: np.ndarray, start_draw_no: int = 1) -> pd.DataFrame:
    era = LOTTO.eras[1]
    rows = []
    for offset, draw in enumerate(draws):
        rows.append(
            [
                start_draw_no + offset,
                date(2020, 1, 1),
                1,
                *draw.tolist(),
                int(offset % era.pool_size) + 1,
                "Lotto1",
                "L1",
            ]
        )
    return pd.DataFrame(rows, columns=list(COLUMNS))


@pytest.fixture
def era2_draws_df() -> pd.DataFrame:
    draws = fair_draws(LOTTO.eras[1], 200, np.random.default_rng(19))
    return draws_to_df(draws)


def test_chi2_uniform_known_answer():
    statistic, p_value = chi2_uniform(np.full(49, 120))
    assert statistic == pytest.approx(0.0)
    assert p_value == pytest.approx(1.0)


def test_chi2_matches_scipy():
    counts = np.array([17, 23, 11, 29, 20])
    statistic, p_value = chi2_uniform(counts)
    reference = scipy.stats.chisquare(counts)
    assert statistic == pytest.approx(reference.statistic, rel=1e-12)
    assert p_value == pytest.approx(reference.pvalue, rel=1e-12)


def test_bonus_never_leaks_into_main(era2_draws_df):
    first = frequency_test(era2_draws_df, LOTTO.eras[1], "main")
    shuffled = era2_draws_df.assign(
        bonus=np.random.default_rng(0).permutation(era2_draws_df["bonus"])
    )
    second = frequency_test(shuffled, LOTTO.eras[1], "main")
    assert first.statistic == second.statistic


def test_planted_weight_bias_fires():
    weights = np.full(59, 1.0)
    weights[4] = 4.0
    draws = weighted_draws(LOTTO.eras[1], weights, 5000, np.random.default_rng(11))
    result = frequency_test(draws_to_df(draws), LOTTO.eras[1], "main")
    assert result.p_value is not None
    assert result.p_value < 0.01


def test_homogeneity_identical_counts():
    counts_a = np.full(59, 100)
    counts_b = np.full(59, 100)
    result = homogeneity_test(counts_a, counts_b, LOTTO.eras[1], "main")
    assert result.statistic == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)
    assert result.era == "era-59"
    assert result.subject == "main"


def test_homogeneity_drops_unseen_categories():
    counts_a = np.zeros(59, dtype=int)
    counts_a[[0, 1]] = [10, 5]
    counts_b = np.zeros(59, dtype=int)
    counts_b[[0, 2]] = [8, 7]
    result = homogeneity_test(counts_a, counts_b, LOTTO.eras[1], "bonus")
    reference = scipy.stats.chi2_contingency([[10, 5, 0], [8, 0, 7]])
    assert result.statistic == pytest.approx(reference.statistic, rel=1e-12)
    assert result.p_value == pytest.approx(reference.pvalue, rel=1e-12)
    assert result.details["dof"] == 2
