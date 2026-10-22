"""Order-statistic distributions for the six main balls."""

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


def _comb(n: int, k: int) -> int:
    return comb(n, k) if 0 <= k <= n else 0


def min_pmf(era: Era) -> np.ndarray:
    """Return ``P(min = m)`` at index ``m`` for ``m`` in ``1..pool_size``."""
    total = comb(era.pool_size, era.picks)
    pmf = np.zeros(era.pool_size + 1, dtype=float)
    for value in range(1, era.pool_size - era.picks + 2):
        pmf[value] = _comb(era.pool_size - value, era.picks - 1) / total
    return pmf


def _max_pmf(era: Era) -> np.ndarray:
    total = comb(era.pool_size, era.picks)
    pmf = np.zeros(era.pool_size + 1, dtype=float)
    for value in range(era.picks, era.pool_size + 1):
        pmf[value] = _comb(value - 1, era.picks - 1) / total
    return pmf


def _spread_pmf(era: Era) -> np.ndarray:
    total = comb(era.pool_size, era.picks)
    pmf = np.zeros(era.pool_size, dtype=float)
    for spread in range(era.picks - 1, era.pool_size):
        pmf[spread] = (
            (era.pool_size - spread) * _comb(spread - 1, era.picks - 2) / total
        )
    return pmf


def _references(era: Era) -> dict[str, np.ndarray]:
    return {"min": min_pmf(era), "max": _max_pmf(era), "spread": _spread_pmf(era)}


def _values(draws: np.ndarray) -> dict[str, np.ndarray]:
    low = draws.min(axis=1)
    high = draws.max(axis=1)
    return {"min": low, "max": high, "spread": high - low}


def _ks_discrete(values: np.ndarray, pmf: np.ndarray) -> float:
    values = np.asarray(values, dtype=int)
    pmf = np.asarray(pmf, dtype=float)
    if values.size == 0 or pmf.sum() <= 0:
        raise ValueError("values and pmf must be non-empty")
    ordered = np.sort(values)
    support = np.arange(pmf.size)
    reference = np.cumsum(pmf)
    empirical_le = np.searchsorted(ordered, support, side="right") / ordered.size
    empirical_lt = np.searchsorted(ordered, support, side="left") / ordered.size
    return float(
        np.max(
            np.maximum(
                np.abs(empirical_le - reference),
                np.abs(reference - empirical_lt),
            )
        )
    )


def _ks_from_array(draws: np.ndarray, era: Era) -> float:
    values = _values(np.asarray(draws, dtype=int))
    references = _references(era)
    return float(
        max(_ks_discrete(values[name], references[name]) for name in references)
    )


def _cdf_points(values: np.ndarray, pmf: np.ndarray) -> dict[str, list[float]]:
    ordered = np.sort(np.asarray(values, dtype=int))
    support = np.arange(pmf.size)
    empirical = np.searchsorted(ordered, support, side="right") / ordered.size
    return {
        "support": support.tolist(),
        "empirical": empirical.tolist(),
        "reference": np.cumsum(pmf).tolist(),
    }


def order_stats_ks(draws_df: pd.DataFrame, era: Era) -> float:
    """Return the maximum KS over min, max and spread against exact references."""
    draws = draws_df[list(_MAIN_BALLS)].to_numpy()
    return _ks_from_array(draws, era)


def order_stats_test(
    draws_df: pd.DataFrame,
    era: Era,
    rng: np.random.Generator,
    replicates: int = DEFAULT_REPLICATES,
) -> TestResult:
    """Run a Monte Carlo test on the maximum order-statistic KS distance."""
    draws = draws_df[list(_MAIN_BALLS)].to_numpy()
    observed = _ks_from_array(draws, era)

    def statistic(null_draws: np.ndarray) -> float:
        return _ks_from_array(null_draws, era)

    null = null_distribution(statistic, era, len(draws_df), replicates, rng)
    values = _values(draws)
    references = _references(era)
    return TestResult(
        family="order-stats",
        era=era.name,
        subject="main",
        statistic=observed,
        p_value=mc_p_value(observed, null),
        method="Monte Carlo max KS of min, max and spread",
        details={
            "cdfs": {
                name: _cdf_points(values[name], references[name]) for name in references
            },
            "replicates": replicates,
        },
    )
