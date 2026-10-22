"""Tests for serial independence of draw-level features."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from poc_lottery_randomness_auditor.data import COLUMNS
from poc_lottery_randomness_auditor.engine.serial import (
    acf_max,
    draw_features,
    serial_test,
)
from poc_lottery_randomness_auditor.games import LOTTO


def draws_to_df(draws: np.ndarray) -> pd.DataFrame:
    rows = []
    for offset, draw in enumerate(draws):
        rows.append([offset + 1, date(2020, 1, 1), 1, *draw.tolist(), 1, "M", "L1"])
    return pd.DataFrame(rows, columns=list(COLUMNS))


@pytest.fixture
def tiny_draws_df() -> pd.DataFrame:
    return draws_to_df(np.array([[1, 2, 3, 4, 5, 6], [10, 11, 12, 13, 14, 15]]))


def drifting_draws(t: np.ndarray) -> np.ndarray:
    base = 1 + (t * 20 // len(t))
    return np.column_stack([base + offset for offset in range(6)])


def test_draw_features_values(tiny_draws_df):
    features = draw_features(tiny_draws_df)
    main_columns = [f"n{i}" for i in range(1, 7)]
    assert features.loc[0, "sum"] == int(tiny_draws_df.loc[0, main_columns].sum())
    assert features.loc[0, "range"] == 5
    assert features.loc[0, "odd_count"] == 3
    assert features.loc[1, "sum"] == int(tiny_draws_df.loc[1, main_columns].sum())


def test_acf_max_white_noise_small():
    rng = np.random.default_rng(12)
    features = pd.DataFrame(
        {
            "sum": rng.normal(size=500),
            "range": rng.normal(size=500),
            "odd_count": rng.normal(size=500),
        }
    )
    assert acf_max(features) < 0.2


def test_planted_drift_fires():
    t = np.arange(800)
    draws = drifting_draws(t)
    result = serial_test(
        draws_to_df(draws), LOTTO.eras[1], np.random.default_rng(13), replicates=1000
    )
    assert result.p_value is not None
    assert result.p_value < 0.01
