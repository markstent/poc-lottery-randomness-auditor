"""Runs tests on time-ordered per-draw bit sequences."""

import numpy as np
import pandas as pd

from poc_lottery_randomness_auditor.engine.simulate import (
    DEFAULT_REPLICATES,
    mc_p_value,
)
from poc_lottery_randomness_auditor.engine.types import TestResult
from poc_lottery_randomness_auditor.games import Era

_MAIN_BALLS: tuple[str, ...] = ("n1", "n2", "n3", "n4", "n5", "n6")


def sum_parity_bits(draws_df: pd.DataFrame) -> np.ndarray:
    """Return one bit per draw: True when the six main balls sum to odd."""
    return (draws_df[list(_MAIN_BALLS)].sum(axis=1).to_numpy() % 2).astype(bool)


def sum_high_bits(draws_df: pd.DataFrame, era: Era) -> np.ndarray:
    """Return one bit per draw: True when the main-six sum is above the median."""
    median = era.picks * (era.pool_size + 1) / 2
    return draws_df[list(_MAIN_BALLS)].sum(axis=1).to_numpy() > median


def _bits(bits: np.ndarray) -> np.ndarray:
    values = np.asarray(bits)
    if values.dtype == bool:
        values = values.astype(int)
    if values.ndim != 1 or values.size < 2:
        raise ValueError(
            "bits must be a one-dimensional array with at least two values"
        )
    if not np.isin(values, [0, 1]).all():
        raise ValueError("bits must contain only 0 and 1")
    return values.astype(int)


def runs_z(bits: np.ndarray) -> tuple[int, float]:
    """Return observed run count and Wald-Wolfowitz z against independence."""
    values = _bits(bits)
    runs = int(1 + np.count_nonzero(values[1:] != values[:-1]))
    n_one = int(values.sum())
    n_zero = int(values.size - n_one)
    if n_one == 0 or n_zero == 0:
        return runs, 0.0
    mean = 1.0 + 2.0 * n_one * n_zero / values.size
    variance = (2.0 * n_one * n_zero * (2.0 * n_one * n_zero - values.size)) / (
        values.size**2 * (values.size - 1)
    )
    return runs, float((runs - mean) / np.sqrt(variance))


def runs_test(
    bits: np.ndarray,
    rng: np.random.Generator,
    replicates: int = DEFAULT_REPLICATES,
    *,
    era: str = "sequence",
    subject: str = "bits",
) -> TestResult:
    """Run a two-sided permutation test on the run count of ``bits``.

    The null is the run-count z of random permutations of the same bits. The
    p-value is upper tail on ``abs(z)``, so unusually many and unusually few
    runs both count as evidence.
    """
    values = _bits(bits)
    observed_runs, observed_z = runs_z(values)
    null_z = np.empty(replicates, dtype=float)
    for index in range(replicates):
        _, null_z[index] = runs_z(rng.permutation(values))
    n_one = int(values.sum())
    n_zero = int(values.size - n_one)
    expected_runs = (
        1.0 + 2.0 * n_one * n_zero / values.size
        if n_one and n_zero
        else float(observed_runs)
    )
    trace = 1 + np.concatenate(([0], np.cumsum(values[1:] != values[:-1])))
    return TestResult(
        family="runs",
        era=era,
        subject=subject,
        statistic=float(observed_runs),
        p_value=mc_p_value(abs(observed_z), np.abs(null_z)),
        method="permutation two-sided runs",
        details={
            "z": observed_z,
            "n_runs": observed_runs,
            "expected_runs": expected_runs,
            "trace": trace.astype(int).tolist(),
            "replicates": replicates,
        },
    )
