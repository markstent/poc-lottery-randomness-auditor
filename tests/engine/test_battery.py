"""Tests for battery orchestration and multiple-comparison control."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from poc_lottery_randomness_auditor.data import COLUMNS
from poc_lottery_randomness_auditor.engine.battery import (
    benjamini_hochberg,
    benjamini_yekutieli,
    run_battery,
)
from poc_lottery_randomness_auditor.engine.simulate import sticky_pair_draws
from poc_lottery_randomness_auditor.games import LOTTO


def era2_df(draws: np.ndarray) -> pd.DataFrame:
    rows = []
    for offset, draw in enumerate(draws):
        rows.append(
            [
                offset + 1,
                date(2020, 1, 1),
                1,
                *draw.tolist(),
                1,
                "Lotto1",
                "L1",
                "era-59",
            ]
        )
    return pd.DataFrame(rows, columns=[*COLUMNS, "era"])


@pytest.fixture
def era2_sticky_df() -> pd.DataFrame:
    draws = sticky_pair_draws(
        LOTTO.eras[1], (7, 42), 0.5, 1200, np.random.default_rng(61)
    )
    return era2_df(draws)


def test_bh_known_answer():
    assert benjamini_hochberg([0.001, 0.02, 0.5]) == [True, True, False]


def test_by_known_answer():
    assert benjamini_yekutieli([0.001, 0.02, 0.5]) == [True, False, False]


def test_battery_flags_planted_pair(era2_sticky_df):
    result = run_battery(era2_sticky_df, LOTTO, seed=1, replicates=500)
    assert any("pairs" in key for key in result.rejections_bh)


def test_battery_deterministic(era2_sticky_df):
    first = run_battery(era2_sticky_df, LOTTO, seed=1, replicates=500)
    second = run_battery(era2_sticky_df, LOTTO, seed=1, replicates=500)
    assert [result.p_value for result in first.results] == [
        result.p_value for result in second.results
    ]
