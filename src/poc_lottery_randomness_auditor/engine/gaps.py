"""Gap analysis for how long each number waits between appearances."""

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


def gap_ks(waits: np.ndarray, p: float) -> float:
    """Return the KS distance of observed waits against ``geom(p)``."""
    waits = np.asarray(waits, dtype=int)
    if waits.size == 0:
        raise ValueError("waits must be non-empty")
    if (waits < 1).any():
        raise ValueError("waits must be at least 1")
    if not 0.0 < p < 1.0:
        raise ValueError("p must be between 0 and 1")
    ordered = np.sort(waits)
    values = np.unique(ordered)
    model = 1.0 - (1.0 - p) ** values
    empirical_le = np.searchsorted(ordered, values, side="right") / ordered.size
    empirical_lt = np.searchsorted(ordered, values, side="left") / ordered.size
    distance = np.maximum(np.abs(empirical_le - model), np.abs(model - empirical_lt))
    return float(np.max(distance))


def _presence(draws: np.ndarray, era: Era) -> np.ndarray:
    draws = np.asarray(draws, dtype=int)
    present = np.zeros((len(draws), era.pool_size), dtype=bool)
    rows = np.repeat(np.arange(len(draws)), draws.shape[1])
    present[rows, draws.ravel() - 1] = True
    return present


def _waits_by_number(draws: np.ndarray, era: Era) -> list[np.ndarray]:
    present = _presence(draws, era)
    waits = []
    for number in range(era.pool_size):
        appearances = np.flatnonzero(present[:, number]) + 1
        boundaries = np.concatenate(([0], appearances, [len(draws) + 1]))
        waits.append(np.diff(boundaries))
    return waits


def _per_number_ks(draws: np.ndarray, era: Era) -> np.ndarray:
    p = era.picks / era.pool_size
    return np.array([gap_ks(waits, p) for waits in _waits_by_number(draws, era)])


def gaps_omnibus(draws_df: pd.DataFrame, era: Era) -> float:
    """Return the maximum per-number gap KS statistic."""
    draws = draws_df[list(_MAIN_BALLS)].to_numpy()
    return float(np.max(_per_number_ks(draws, era)))


def overdue_table(draws_df: pd.DataFrame, era: Era, top: int = 10) -> pd.DataFrame:
    """Return the most overdue numbers, exploratory only.

    ``draws_since_seen`` is zero for a number seen in the latest draw and
    ``len(draws_df)`` for a number never seen in the frame.
    """
    if top < 1:
        raise ValueError("top must be at least 1")
    present = _presence(draws_df[list(_MAIN_BALLS)].to_numpy(), era)
    expected_wait = era.pool_size / era.picks
    records = []
    for number in range(1, era.pool_size + 1):
        appearances = np.flatnonzero(present[:, number - 1]) + 1
        last_seen = int(appearances[-1]) if appearances.size else 0
        records.append(
            {
                "number": number,
                "draws_since_seen": len(draws_df) - last_seen,
                "expected_wait": expected_wait,
            }
        )
    table = pd.DataFrame(records)
    return table.sort_values(
        ["draws_since_seen", "number"], ascending=[False, True]
    ).head(top)


def gaps_test(
    draws_df: pd.DataFrame,
    era: Era,
    rng: np.random.Generator,
    replicates: int = DEFAULT_REPLICATES,
) -> TestResult:
    """Run a Monte Carlo test on the maximum per-number gap KS statistic."""
    draws = draws_df[list(_MAIN_BALLS)].to_numpy()
    per_number = _per_number_ks(draws, era)
    observed = float(np.max(per_number))

    def statistic(null_draws: np.ndarray) -> float:
        return float(np.max(_per_number_ks(null_draws, era)))

    null = null_distribution(statistic, era, len(draws_df), replicates, rng)
    overdue = overdue_table(draws_df, era)
    return TestResult(
        family="gaps",
        era=era.name,
        subject="main",
        statistic=observed,
        p_value=mc_p_value(observed, null),
        method="Monte Carlo max per-number gap KS",
        details={
            "ks": per_number.tolist(),
            "overdue": overdue.to_dict("records"),
            "overdue_exploratory": True,
            "replicates": replicates,
        },
    )
