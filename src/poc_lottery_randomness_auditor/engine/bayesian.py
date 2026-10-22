"""Bayesian uniformity layer.

Model: a symmetric Dirichlet(1) prior over ball probabilities and a
multinomial likelihood for the six main balls. Each marginal posterior is
Beta(1 + count_i, 1 + total - count_i). ``certified_delta`` is read as: at
the chosen credibility, every ball is within ``delta`` of its fair share.
"""

import numpy as np
import pandas as pd
import scipy.stats

from poc_lottery_randomness_auditor.engine.types import TestResult
from poc_lottery_randomness_auditor.games import Era

ROPE_DELTA: float = 0.05
_MAIN_BALLS: tuple[str, ...] = ("n1", "n2", "n3", "n4", "n5", "n6")


def _validate(counts: np.ndarray, credibility: float) -> tuple[np.ndarray, float]:
    counts = np.asarray(counts, dtype=int)
    if counts.ndim != 1 or counts.size == 0:
        raise ValueError("counts must be a non-empty one-dimensional array")
    if (counts < 0).any() or counts.sum() == 0:
        raise ValueError("counts must be non-negative with a positive total")
    if not 0.0 < credibility < 1.0:
        raise ValueError("credibility must be between 0 and 1")
    return counts, float(counts.sum())


def posterior_intervals(counts: np.ndarray, credibility: float = 0.95) -> np.ndarray:
    """Return Beta marginal credible intervals under a Dirichlet(1) posterior."""
    counts, total = _validate(counts, credibility)
    alpha = 1 + counts
    beta = 1 + total - counts
    tail = (1.0 - credibility) / 2.0
    return scipy.stats.beta(alpha, beta).ppf([[tail], [1.0 - tail]]).T


def certified_delta(counts: np.ndarray, credibility: float = 0.95) -> float:
    """Return the smallest delta containing every interval inside fair*(1 ± delta)."""
    counts, _ = _validate(counts, credibility)
    intervals = posterior_intervals(counts, credibility)
    uniform = 1.0 / counts.size
    lower_gap = 1.0 - intervals[:, 0] / uniform
    upper_gap = intervals[:, 1] / uniform - 1.0
    return float(max(0.0, np.max(lower_gap), np.max(upper_gap)))


def bayesian_uniformity(draws_df: pd.DataFrame, era: Era) -> TestResult:
    """Return the Bayesian main-ball uniformity result for one era."""
    values = draws_df[list(_MAIN_BALLS)].to_numpy().ravel()
    counts = np.bincount(values.astype(int), minlength=era.pool_size + 1)[1:]
    intervals = posterior_intervals(counts)
    delta = certified_delta(counts)
    return TestResult(
        family="bayesian",
        era=era.name,
        subject="main",
        statistic=delta,
        p_value=None,
        method="bayesian",
        details={
            "intervals": intervals.tolist(),
            "rope_delta": ROPE_DELTA,
            "credibility": 0.95,
            "within_rope": bool(delta < ROPE_DELTA),
        },
    )
