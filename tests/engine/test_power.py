"""Tests for the power analysis grid."""

from poc_lottery_randomness_auditor.engine.power import (
    PowerScenario,
    power_grid,
)
from poc_lottery_randomness_auditor.games import LOTTO


def test_power_high_for_strong_bias():
    scenarios = [PowerScenario("frequency", "weighted", level=4.0, n_draws=3000)]
    frame = power_grid(scenarios, LOTTO.eras[1], replicates=100, seed=7)
    assert frame.loc[0, "power"] > 0.9


def test_power_near_alpha_for_null_bias():
    scenarios = [PowerScenario("frequency", "weighted", level=1.0, n_draws=3000)]
    frame = power_grid(scenarios, LOTTO.eras[1], replicates=100, seed=7)
    assert frame.loc[0, "power"] < 0.2
