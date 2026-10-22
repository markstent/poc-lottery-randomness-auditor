"""Power analysis for showcased bias detectors."""

from dataclasses import dataclass
from functools import cache

import numpy as np
import pandas as pd

from poc_lottery_randomness_auditor.engine.frequencies import chi2_uniform
from poc_lottery_randomness_auditor.engine.pairs import (
    _pair_counts_array,
    expected_pair_count,
    pairs_omnibus,
)
from poc_lottery_randomness_auditor.engine.simulate import (
    cyclic_draws,
    fair_draws,
    mc_p_value,
    null_distribution,
    sticky_pair_draws,
    weighted_draws,
)
from poc_lottery_randomness_auditor.engine.spectral import _peak_table
from poc_lottery_randomness_auditor.games import Era


@dataclass(frozen=True)
class PowerScenario:
    """One power-grid cell: a detector against one bias kind and level."""

    family: str
    bias_kind: str
    level: float
    n_draws: int


DEFAULT_SCENARIOS: list[PowerScenario] = [
    PowerScenario(family, bias, level, n_draws)
    for family, bias, levels in (
        ("frequency", "weighted", (1.5, 2.5, 4.0)),
        ("pairs", "sticky", (0.1, 0.2, 0.4)),
        ("spectral", "cyclic", (10.0, 20.0, 40.0)),
    )
    for level in levels
    for n_draws in (1000, 3000)
]


def _validate(scenario: PowerScenario) -> None:
    expected = {
        "frequency": "weighted",
        "pairs": "sticky",
        "spectral": "cyclic",
    }
    if expected.get(scenario.family) != scenario.bias_kind:
        raise ValueError(
            f"unknown power scenario {scenario.family!r} vs {scenario.bias_kind!r}"
        )
    if scenario.n_draws < 1:
        raise ValueError("n_draws must be at least 1")


def _rigged_draws(
    scenario: PowerScenario,
    era: Era,
    rng: np.random.Generator,
) -> np.ndarray:
    if scenario.bias_kind == "weighted":
        if scenario.level <= 0:
            raise ValueError("weighted level must be positive")
        if scenario.level == 1.0:
            return fair_draws(era, scenario.n_draws, rng)
        weights = np.ones(era.pool_size)
        weights[0] = scenario.level
        return weighted_draws(era, weights, scenario.n_draws, rng)
    if scenario.bias_kind == "sticky":
        if not 0.0 <= scenario.level <= 1.0:
            raise ValueError("sticky level must be a boost between 0 and 1")
        if scenario.level == 0.0:
            return fair_draws(era, scenario.n_draws, rng)
        pair = (7, 42) if era.pool_size >= 42 else (1, 2)
        return sticky_pair_draws(era, pair, scenario.level, scenario.n_draws, rng)
    if scenario.bias_kind == "cyclic":
        period = round(scenario.level)
        if period < 1:
            return fair_draws(era, scenario.n_draws, rng)
        favoured = 7 if era.pool_size >= 7 else 1
        return cyclic_draws(era, favoured, period, scenario.n_draws, rng)
    raise ValueError(f"unknown bias kind {scenario.bias_kind!r}")


def _family_statistic(family: str, draws: np.ndarray, era: Era) -> float:
    if family == "frequency":
        counts = np.bincount(draws.ravel().astype(int), minlength=era.pool_size + 1)[1:]
        return chi2_uniform(counts)[0]
    if family == "pairs":
        expected = expected_pair_count(len(draws), era)
        return pairs_omnibus(_pair_counts_array(draws, era), expected)
    if family == "spectral":
        table = _peak_table(draws, era)
        return float(table["ratio"].max()) if len(table) else 0.0
    raise ValueError(f"unknown power family {family!r}")


@cache
def _fair_null(
    family: str, era: Era, n_draws: int, replicates: int, seed: int
) -> np.ndarray:
    def statistic(draws: np.ndarray) -> float:
        return _family_statistic(family, draws, era)

    return null_distribution(
        statistic,
        era,
        n_draws,
        replicates,
        np.random.default_rng(seed),
    )


def power_grid(
    scenarios: list[PowerScenario],
    era: Era,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Estimate power for each scenario at the 0.01 threshold.

    The fair null for one ``(family, n_draws)`` pair is cached and shared across
    levels, as documented, so stronger and weaker biases are compared against
    exactly the same null.
    """
    if replicates < 1:
        raise ValueError("replicates must be at least 1")
    base_rng = np.random.default_rng(seed)
    scenario_seeds = base_rng.integers(
        0, np.iinfo(np.int64).max, size=len(scenarios), dtype=np.int64
    )
    records = []
    for scenario, scenario_seed in zip(scenarios, scenario_seeds, strict=True):
        _validate(scenario)
        null = _fair_null(
            scenario.family,
            era,
            scenario.n_draws,
            replicates,
            seed,
        )
        rng = np.random.default_rng(int(scenario_seed))
        statistics = np.array(
            [
                _family_statistic(
                    scenario.family,
                    _rigged_draws(scenario, era, rng),
                    era,
                )
                for _ in range(replicates)
            ]
        )
        p_values = np.array([mc_p_value(statistic, null) for statistic in statistics])
        records.append(
            {
                "family": scenario.family,
                "bias_kind": scenario.bias_kind,
                "level": scenario.level,
                "n_draws": scenario.n_draws,
                "power": float(np.mean(p_values < 0.01)),
            }
        )
    return pd.DataFrame(
        records,
        columns=["family", "bias_kind", "level", "n_draws", "power"],
    )
