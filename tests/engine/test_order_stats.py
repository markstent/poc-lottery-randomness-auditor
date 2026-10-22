"""Tests for order-statistic distributions of the main six."""

from datetime import date
from math import comb

import numpy as np
import pandas as pd
import pytest

from poc_lottery_randomness_auditor.data import COLUMNS
from poc_lottery_randomness_auditor.engine.order_stats import (
    min_pmf,
    order_stats_test,
)
from poc_lottery_randomness_auditor.engine.simulate import weighted_draws
from poc_lottery_randomness_auditor.games import LOTTO


def draws_to_df(draws: np.ndarray) -> pd.DataFrame:
    rows = []
    for offset, draw in enumerate(draws):
        rows.append([offset + 1, date(2020, 1, 1), 1, *draw.tolist(), 1, "M", "L1"])
    return pd.DataFrame(rows, columns=list(COLUMNS))


def test_min_pmf_known_answer():
    era = LOTTO.eras[0]
    pmf = min_pmf(era)
    assert pmf[1] == pytest.approx(6 / 49)
    assert pmf.sum() == pytest.approx(1.0)
    for m in (1, 10, 30, 44):
        assert pmf[m:].sum() == pytest.approx(
            comb(era.pool_size - m + 1, era.picks) / comb(era.pool_size, era.picks)
        )


def test_low_bias_fires():
    weights = np.linspace(2.0, 0.5, 59)
    draws = weighted_draws(LOTTO.eras[1], weights, 4000, np.random.default_rng(31))
    result = order_stats_test(
        draws_to_df(draws), LOTTO.eras[1], np.random.default_rng(32), replicates=1000
    )
    assert result.p_value is not None
    assert result.p_value < 0.01
