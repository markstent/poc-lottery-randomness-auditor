"""Tests for runs tests on per-draw bit sequences."""

from datetime import date

import numpy as np
import pandas as pd

from poc_lottery_randomness_auditor.data import COLUMNS
from poc_lottery_randomness_auditor.engine.runs import (
    runs_test,
    runs_z,
    sum_high_bits,
    sum_parity_bits,
)
from poc_lottery_randomness_auditor.games import Era


def draws_to_df(draws: np.ndarray) -> pd.DataFrame:
    rows = []
    for offset, draw in enumerate(draws):
        rows.append([offset + 1, date(2020, 1, 1), 1, *draw.tolist(), 1, "M", "L1"])
    return pd.DataFrame(rows, columns=list(COLUMNS))


def test_sum_bit_helpers():
    df = draws_to_df(np.array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 7]]))
    assert sum_parity_bits(df).tolist() == [True, False]
    toy = Era("toy", date(2020, 1, 1), None, 10, 6, True, 1)
    high_df = draws_to_df(np.array([[1, 2, 3, 4, 5, 6], [5, 6, 7, 8, 9, 10]]))
    assert sum_high_bits(high_df, toy).tolist() == [False, True]


def test_runs_z_alternating_and_blocked():
    alternating = np.arange(50) % 2
    n_runs, z = runs_z(alternating)
    assert n_runs == 50 and z > 5
    blocked = np.array([0] * 25 + [1] * 25)
    n_runs_b, z_b = runs_z(blocked)
    assert n_runs_b == 2 and z_b < -5


def test_planted_alternation_fires():
    result = runs_test(np.arange(400) % 2, np.random.default_rng(2), replicates=2000)
    assert result.p_value is not None
    assert result.p_value < 0.01


def test_fair_bits_do_not_fire():
    bits = np.random.default_rng(4).integers(0, 2, 400)
    result = runs_test(bits, np.random.default_rng(5), replicates=2000)
    assert result.p_value is not None
    assert result.p_value > 0.01
    assert result.era == "sequence"
