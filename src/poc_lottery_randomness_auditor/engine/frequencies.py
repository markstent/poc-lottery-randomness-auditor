"""Chi-squared frequency uniformity and homogeneity tests."""

import numpy as np
import pandas as pd
import scipy.stats

from poc_lottery_randomness_auditor.engine.types import TestResult
from poc_lottery_randomness_auditor.games import Era

_MAIN_BALLS: tuple[str, ...] = ("n1", "n2", "n3", "n4", "n5", "n6")


def _counts(values: np.ndarray, pool_size: int) -> np.ndarray:
    return np.bincount(values.astype(int), minlength=pool_size + 1)[1:]


def _validate_counts(counts: np.ndarray, pool_size: int, name: str) -> np.ndarray:
    counts = np.asarray(counts, dtype=int)
    if counts.shape != (pool_size,):
        raise ValueError(f"{name} must have shape ({pool_size},)")
    if (counts < 0).any():
        raise ValueError(f"{name} must be non-negative")
    if counts.sum() == 0:
        raise ValueError(f"{name} must contain at least one observation")
    return counts


def chi2_uniform(counts: np.ndarray) -> tuple[float, float]:
    """Return the chi-squared statistic and p-value against uniformity."""
    counts = np.asarray(counts, dtype=int)
    if counts.ndim != 1 or counts.size == 0:
        raise ValueError("counts must be a non-empty one-dimensional array")
    if (counts < 0).any():
        raise ValueError("counts must be non-negative")
    result = scipy.stats.chisquare(counts)
    return float(result.statistic), float(result.pvalue)


def frequency_test(draws_df: pd.DataFrame, era: Era, subject: str) -> TestResult:
    """Test main-ball or bonus-ball frequency uniformity for one era."""
    if subject == "main":
        values = draws_df[list(_MAIN_BALLS)].to_numpy().ravel()
    elif subject == "bonus":
        if not era.has_bonus:
            raise ValueError(f"{era.name} has no bonus ball")
        values = draws_df["bonus"].to_numpy()
    else:
        raise ValueError(f"unknown frequency subject {subject!r}")

    counts = _counts(values, era.pool_size)
    statistic, p_value = chi2_uniform(counts)
    return TestResult(
        family="frequency",
        era=era.name,
        subject=subject,
        statistic=statistic,
        p_value=p_value,
        method="chi-square uniformity",
        details={"counts": counts.tolist(), "expected": values.size / era.pool_size},
    )


def homogeneity_test(
    counts_a: np.ndarray,
    counts_b: np.ndarray,
    era: Era,
    subject: str,
) -> TestResult:
    """Test whether two count vectors for one era come from one distribution."""
    counts_a = _validate_counts(counts_a, era.pool_size, "counts_a")
    counts_b = _validate_counts(counts_b, era.pool_size, "counts_b")
    result = scipy.stats.chi2_contingency(np.vstack([counts_a, counts_b]))
    return TestResult(
        family="homogeneity",
        era=era.name,
        subject=subject,
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        method="chi-square homogeneity",
        details={
            "counts_a": counts_a.tolist(),
            "counts_b": counts_b.tolist(),
            "dof": int(result.dof),
        },
    )
