"""Tests for pair co-occurrence Monte Carlo checks."""

from datetime import date
from math import comb

import numpy as np
import pandas as pd
import pytest

from poc_lottery_randomness_auditor.data import COLUMNS
from poc_lottery_randomness_auditor.engine.pairs import (
    expected_pair_count,
    pair_counts,
    pairs_omnibus,
    pairs_test,
)
from poc_lottery_randomness_auditor.engine.simulate import (
    fair_draws,
    sticky_pair_draws,
)
from poc_lottery_randomness_auditor.games import LOTTO


def draws_to_df(draws: np.ndarray, start_draw_no: int = 1) -> pd.DataFrame:
    rows = []
    for offset, draw in enumerate(draws):
        rows.append(
            [
                start_draw_no + offset,
                date(2020, 1, 1),
                1,
                *draw.tolist(),
                1,
                "Lotto1",
                "L1",
            ]
        )
    return pd.DataFrame(rows, columns=list(COLUMNS))


def test_pair_counts_symmetric_zero_diagonal():
    df = draws_to_df(np.array([[1, 2, 3, 4, 5, 6], [1, 2, 7, 8, 9, 10]]))
    counts = pair_counts(df, LOTTO.eras[1])
    assert counts.shape == (59, 59)
    assert counts[0, 1] == 2
    assert counts[1, 0] == 2
    assert np.all(np.diag(counts) == 0)
    assert np.array_equal(counts, counts.T)


def test_expected_pair_count_exact():
    assert expected_pair_count(100, LOTTO.eras[1]) == pytest.approx(
        100 * comb(57, 4) / comb(59, 6)
    )


def test_omnibus_zero_at_exact_expectation():
    counts = np.full((59, 59), 10.0)
    np.fill_diagonal(counts, 0.0)
    assert pairs_omnibus(counts, 10.0) == pytest.approx(0.0)


def test_planted_sticky_pair_fires():
    draws = sticky_pair_draws(
        LOTTO.eras[1], (7, 42), 0.3, 3000, np.random.default_rng(5)
    )
    result = pairs_test(
        draws_to_df(draws), LOTTO.eras[1], np.random.default_rng(6), replicates=1000
    )
    assert result.p_value is not None
    assert result.p_value < 0.01


def test_fair_seed_does_not_fire():
    draws = fair_draws(LOTTO.eras[1], 3000, np.random.default_rng(8))
    result = pairs_test(
        draws_to_df(draws), LOTTO.eras[1], np.random.default_rng(9), replicates=1000
    )
    assert result.p_value is not None
    assert result.p_value > 0.01
