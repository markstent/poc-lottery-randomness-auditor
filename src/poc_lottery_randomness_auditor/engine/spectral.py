"""Spectral analysis of per-number appearance indicators."""

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


def periodogram(bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return frequencies and power for the demeaned indicator series."""
    series = np.asarray(bits, dtype=float)
    if series.ndim != 1 or series.size < 2:
        raise ValueError(
            "bits must be a one-dimensional array with at least two values"
        )
    demeaned = series - series.mean()
    frequencies = np.fft.rfftfreq(series.size, d=1.0)
    power = np.abs(np.fft.rfft(demeaned)) ** 2
    return frequencies, power


def _presence(draws: np.ndarray, era: Era) -> np.ndarray:
    draws = np.asarray(draws, dtype=int)
    present = np.zeros((len(draws), era.pool_size), dtype=float)
    rows = np.repeat(np.arange(len(draws)), draws.shape[1])
    present[rows, draws.ravel() - 1] = 1.0
    return present


def _power(draws: np.ndarray, era: Era) -> tuple[np.ndarray, np.ndarray]:
    present = _presence(draws, era)
    demeaned = present - present.mean(axis=0)
    frequencies = np.fft.rfftfreq(len(draws), d=1.0)
    power = np.abs(np.fft.rfft(demeaned, axis=0)) ** 2
    return frequencies, power


def _peak_table(draws: np.ndarray, era: Era) -> pd.DataFrame:
    frequencies, power = _power(draws, era)
    if power.shape[0] < 2:
        return pd.DataFrame(columns=["number", "frequency", "power", "ratio"])
    body = power[1:]
    medians = np.median(body, axis=0)
    ratios = np.divide(body, medians, out=np.zeros_like(body), where=medians > 0)
    peak_indices = np.argmax(ratios, axis=0)
    records = []
    for number, peak_index in enumerate(peak_indices, start=1):
        records.append(
            {
                "number": number,
                "frequency": float(frequencies[1:][peak_index]),
                "power": float(body[peak_index, number - 1]),
                "ratio": float(ratios[peak_index, number - 1]),
            }
        )
    return pd.DataFrame.from_records(records)


def spectral_omnibus(draws_df: pd.DataFrame, era: Era) -> float:
    """Return the maximum peak-to-median periodogram ratio over all numbers."""
    table = _peak_table(draws_df[list(_MAIN_BALLS)].to_numpy(), era)
    return float(table["ratio"].max()) if len(table) else 0.0


def spectral_test(
    draws_df: pd.DataFrame,
    era: Era,
    rng: np.random.Generator,
    replicates: int = DEFAULT_REPLICATES,
) -> TestResult:
    """Run a Monte Carlo test on the maximum spectral peak ratio."""
    draws = draws_df[list(_MAIN_BALLS)].to_numpy()
    table = _peak_table(draws, era)
    observed = float(table["ratio"].max()) if len(table) else 0.0

    def statistic(null_draws: np.ndarray) -> float:
        null_table = _peak_table(null_draws, era)
        return float(null_table["ratio"].max()) if len(null_table) else 0.0

    null = null_distribution(statistic, era, len(draws_df), replicates, rng)
    return TestResult(
        family="spectral",
        era=era.name,
        subject="main",
        statistic=observed,
        p_value=mc_p_value(observed, null),
        method="Monte Carlo max periodogram peak ratio",
        details={"peaks": table.to_dict("records"), "replicates": replicates},
    )
