"""Pair co-occurrence counts and Monte Carlo omnibus test."""

from itertools import combinations
from math import comb

import numpy as np
import pandas as pd

from poc_lottery_randomness_auditor.engine.simulate import (
    DEFAULT_REPLICATES,
    mc_p_value,
    null_distribution,
)
from poc_lottery_randomness_auditor.engine.types import TestResult
from poc_lottery_randomness_auditor.games import Era

_MAIN_BALLS: tuple[str, ...] = ("n1", "n2", "n3", "n4", "n5", "n6")


def _pair_counts_array(draws: np.ndarray, era: Era) -> np.ndarray:
    draws = np.sort(np.asarray(draws, dtype=int), axis=1)
    columns = np.array(list(combinations(range(era.picks), 2)))
    first = draws[:, columns[:, 0]].ravel() - 1
    second = draws[:, columns[:, 1]].ravel() - 1
    flat = np.bincount(
        first * era.pool_size + second,
        minlength=era.pool_size * era.pool_size,
    )
    counts = flat.reshape(era.pool_size, era.pool_size)
    return counts + counts.T


def pair_counts(draws_df: pd.DataFrame, era: Era) -> np.ndarray:
    """Return symmetric pair counts over the six main balls."""
    return _pair_counts_array(draws_df[list(_MAIN_BALLS)].to_numpy(), era)


def expected_pair_count(n_draws: int, era: Era) -> float:
    """Return the expected count for one pair under fair draws."""
    if n_draws < 0:
        raise ValueError("n_draws must be non-negative")
    return float(
        n_draws
        * comb(era.pool_size - 2, era.picks - 2)
        / comb(era.pool_size, era.picks)
    )


def pairs_omnibus(counts: np.ndarray, expected: float) -> float:
    """Return the maximum absolute standardised pair residual."""
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise ValueError("counts must be a square matrix")
    if expected <= 0:
        raise ValueError("expected must be positive")
    upper = np.triu_indices(counts.shape[0], k=1)
    residuals = (counts[upper] - expected) / np.sqrt(expected)
    return float(np.max(np.abs(residuals)))


def _pair_entries(counts: np.ndarray) -> list[tuple[int, int, int]]:
    entries = []
    for first in range(counts.shape[0]):
        for second in range(first + 1, counts.shape[1]):
            entries.append((first + 1, second + 1, int(counts[first, second])))
    return entries


def pairs_test(
    draws_df: pd.DataFrame,
    era: Era,
    rng: np.random.Generator,
    replicates: int = DEFAULT_REPLICATES,
) -> TestResult:
    """Run a Monte Carlo omnibus test over all main-ball pairs."""
    counts = pair_counts(draws_df, era)
    expected = expected_pair_count(len(draws_df), era)
    observed = pairs_omnibus(counts, expected)

    def statistic(draws: np.ndarray) -> float:
        return pairs_omnibus(_pair_counts_array(draws, era), expected)

    null = null_distribution(statistic, era, len(draws_df), replicates, rng)
    return TestResult(
        family="pairs",
        era=era.name,
        subject="main",
        statistic=observed,
        p_value=mc_p_value(observed, null),
        method="Monte Carlo max standardised pair residual",
        details={
            "expected": expected,
            "pairs": _pair_entries(counts),
            "replicates": replicates,
        },
    )
