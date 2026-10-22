"""Tests for per-number gap analysis and overdue tables."""

from datetime import date

import numpy as np
import pandas as pd
import pytest
import scipy.stats

from poc_lottery_randomness_auditor.data import COLUMNS
from poc_lottery_randomness_auditor.engine.gaps import (
    gap_ks,
    gaps_test,
    overdue_table,
)
from poc_lottery_randomness_auditor.engine.simulate import weighted_draws
from poc_lottery_randomness_auditor.games import LOTTO


def draws_to_df(draws: np.ndarray) -> pd.DataFrame:
    rows = []
    for offset, draw in enumerate(draws):
        rows.append([offset + 1, date(2020, 1, 1), 1, *draw.tolist(), 1, "M", "L1"])
    return pd.DataFrame(rows, columns=list(COLUMNS))


@pytest.fixture
def tiny_draws_df() -> pd.DataFrame:
    return draws_to_df(np.array([[1, 2, 3, 4, 5, 6], [10, 11, 12, 13, 14, 15]]))


def test_gap_ks_matches_scipy():
    waits = np.array([3, 1, 8, 2, 11, 4, 5])
    assert gap_ks(waits, 6 / 59) == pytest.approx(
        scipy.stats.kstest(waits, scipy.stats.geom(6 / 59).cdf).statistic
    )


def test_suppressed_number_fires():
    weights = np.full(59, 1.0)
    weights[13] = 0.0
    draws = weighted_draws(LOTTO.eras[1], weights, 4000, np.random.default_rng(21))
    result = gaps_test(
        draws_to_df(draws), LOTTO.eras[1], np.random.default_rng(22), replicates=1000
    )
    assert result.p_value is not None
    assert result.p_value < 0.01


def test_overdue_table_correct(tiny_draws_df):
    table = overdue_table(tiny_draws_df, LOTTO.eras[1], top=5)
    assert table["draws_since_seen"].is_monotonic_decreasing
    assert set(table.columns) == {"number", "draws_since_seen", "expected_wait"}
    assert len(table) == 5
