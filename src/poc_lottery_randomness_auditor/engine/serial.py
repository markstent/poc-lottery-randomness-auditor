"""Serial independence checks for draw-level features."""

import numpy as np
import pandas as pd

from poc_lottery_randomness_auditor.engine.simulate import (
    DEFAULT_REPLICATES,
    mc_p_value,
    null_distribution,
)
from poc_lottery_randomness_auditor.engine.types import TestResult
from poc_lottery_randomness_auditor.games import Era

FEATURES: tuple[str, ...] = ("sum", "range", "odd_count")
_MAIN_BALLS: tuple[str, ...] = ("n1", "n2", "n3", "n4", "n5", "n6")


def _features_from_array(draws: np.ndarray) -> pd.DataFrame:
    draws = np.asarray(draws, dtype=int)
    return pd.DataFrame(
        {
            "sum": draws.sum(axis=1),
            "range": draws.max(axis=1) - draws.min(axis=1),
            "odd_count": (draws % 2).sum(axis=1),
        }
    )


def draw_features(draws_df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per draw with sum, range and odd-count features."""
    return _features_from_array(draws_df[list(_MAIN_BALLS)].to_numpy())


def _acf_vector(values: np.ndarray, lags: int) -> list[float]:
    series = np.asarray(values, dtype=float)
    if series.size < 2 or lags < 1:
        return []
    series = series - series.mean()
    denominator = float(np.dot(series, series))
    if denominator == 0.0:
        return [0.0] * min(lags, series.size - 1)
    return [
        float(np.dot(series[:-lag], series[lag:]) / denominator)
        for lag in range(1, min(lags, series.size - 1) + 1)
    ]


def _acf_vectors(features: pd.DataFrame, lags: int) -> dict[str, list[float]]:
    return {
        feature: _acf_vector(features[feature].to_numpy(), lags)
        for feature in FEATURES
        if feature in features.columns
    }


def acf_max(features: pd.DataFrame, lags: int = 10) -> float:
    """Return the maximum absolute autocorrelation across features and lags."""
    vectors = _acf_vectors(features, lags)
    values = [value for vector in vectors.values() for value in vector]
    return float(max((abs(value) for value in values), default=0.0))


def serial_test(
    draws_df: pd.DataFrame,
    era: Era,
    rng: np.random.Generator,
    replicates: int = DEFAULT_REPLICATES,
    lags: int = 10,
) -> TestResult:
    """Run a Monte Carlo test on the maximum absolute feature autocorrelation."""
    features = draw_features(draws_df)
    observed = acf_max(features, lags)

    def statistic(draws: np.ndarray) -> float:
        return acf_max(_features_from_array(draws), lags)

    null = null_distribution(statistic, era, len(draws_df), replicates, rng)
    return TestResult(
        family="serial",
        era=era.name,
        subject="main-features",
        statistic=observed,
        p_value=mc_p_value(observed, null),
        method="Monte Carlo max absolute autocorrelation",
        details={
            "lags": lags,
            "acf": _acf_vectors(features, lags),
            "replicates": replicates,
        },
    )
