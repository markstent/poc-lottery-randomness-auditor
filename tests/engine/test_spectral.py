"""Tests for spectral analysis of per-number indicator series."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from poc_lottery_randomness_auditor.data import COLUMNS
from poc_lottery_randomness_auditor.engine.simulate import cyclic_draws
from poc_lottery_randomness_auditor.engine.spectral import (
    periodogram,
    spectral_test,
)
from poc_lottery_randomness_auditor.games import LOTTO


def draws_to_df(draws: np.ndarray) -> pd.DataFrame:
    rows = []
    for offset, draw in enumerate(draws):
        rows.append([offset + 1, date(2020, 1, 1), 1, *draw.tolist(), 1, "M", "L1"])
    return pd.DataFrame(rows, columns=list(COLUMNS))


def test_periodogram_finds_known_period():
    bits = (np.arange(500) % 5 == 0).astype(float)
    frequencies, power = periodogram(bits)
    assert frequencies[np.argmax(power)] == pytest.approx(1 / 5, abs=1 / 500)


def test_cyclic_rig_fires():
    draws = cyclic_draws(
        LOTTO.eras[1],
        favoured=7,
        period=10,
        n_draws=2000,
        rng=np.random.default_rng(41),
    )
    result = spectral_test(
        draws_to_df(draws), LOTTO.eras[1], np.random.default_rng(42), replicates=500
    )
    assert result.p_value is not None
    assert result.p_value < 0.01
