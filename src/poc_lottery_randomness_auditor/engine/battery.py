"""Run the statistical battery and control the false discovery rate."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from poc_lottery_randomness_auditor.engine.bayesian import bayesian_uniformity
from poc_lottery_randomness_auditor.engine.frequencies import (
    frequency_test,
    homogeneity_test,
)
from poc_lottery_randomness_auditor.engine.gaps import gaps_test
from poc_lottery_randomness_auditor.engine.order_stats import order_stats_test
from poc_lottery_randomness_auditor.engine.pairs import pairs_test
from poc_lottery_randomness_auditor.engine.runs import (
    runs_test,
    sum_high_bits,
    sum_parity_bits,
)
from poc_lottery_randomness_auditor.engine.serial import serial_test
from poc_lottery_randomness_auditor.engine.simulate import (
    DEFAULT_REPLICATES,
    MASTER_SEED,
)
from poc_lottery_randomness_auditor.engine.spectral import spectral_test
from poc_lottery_randomness_auditor.engine.types import TestResult
from poc_lottery_randomness_auditor.games import GameSpec

_MAIN_BALLS: tuple[str, ...] = ("n1", "n2", "n3", "n4", "n5", "n6")


@dataclass(frozen=True)
class BatteryResult:
    """All test results plus BH and BY rejection keys."""

    results: tuple[TestResult, ...]
    rejections_bh: tuple[str, ...]
    rejections_by: tuple[str, ...]


def _validate_p_values(p_values: Sequence[float], q: float) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    if values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    if not 0.0 < q < 1.0:
        raise ValueError("q must be between 0 and 1")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("p_values must be finite and inside [0, 1]")
    return values


def _fdr_mask(p_values: Sequence[float], q: float, harmonic: bool) -> list[bool]:
    values = _validate_p_values(p_values, q)
    m = values.size
    if m == 0:
        return []
    order = np.argsort(values)
    ordered = values[order]
    adjustment = np.sum(1.0 / np.arange(1, m + 1)) if harmonic else 1.0
    thresholds = (np.arange(1, m + 1) / m) * (q / adjustment)
    below = ordered <= thresholds
    if not below.any():
        return [False] * m
    last = int(np.max(np.flatnonzero(below)))
    ordered_rejections = np.zeros(m, dtype=bool)
    ordered_rejections[: last + 1] = True
    rejections = np.zeros(m, dtype=bool)
    rejections[order] = ordered_rejections
    return rejections.tolist()


def benjamini_hochberg(p_values: Sequence[float], q: float = 0.05) -> list[bool]:
    """Return per-p-value Benjamini-Hochberg rejection flags."""
    return _fdr_mask(p_values, q, harmonic=False)


def benjamini_yekutieli(p_values: Sequence[float], q: float = 0.05) -> list[bool]:
    """Return per-p-value Benjamini-Yekutieli rejection flags."""
    return _fdr_mask(p_values, q, harmonic=True)


def _key(result: TestResult) -> str:
    return f"{result.era}:{result.family}:{result.subject}"


def _counts(values: np.ndarray, pool_size: int) -> np.ndarray:
    return np.bincount(values.astype(int), minlength=pool_size + 1)[1:]


def _main_counts(frame: pd.DataFrame, pool_size: int) -> np.ndarray:
    return _counts(frame[list(_MAIN_BALLS)].to_numpy().ravel(), pool_size)


def _bonus_counts(frame: pd.DataFrame, pool_size: int) -> np.ndarray:
    return _counts(frame["bonus"].to_numpy(), pool_size)


def _rejections(
    results: tuple[TestResult, ...], spec: GameSpec
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    rejections_bh: list[str] = []
    rejections_by: list[str] = []
    for era in spec.eras:
        era_results = [
            result
            for result in results
            if result.era == era.name and result.p_value is not None
        ]
        if not era_results:
            continue
        p_values = [float(result.p_value) for result in era_results]
        rejections_bh.extend(
            _key(result)
            for result, rejected in zip(
                era_results, benjamini_hochberg(p_values), strict=True
            )
            if rejected
        )
        rejections_by.extend(
            _key(result)
            for result, rejected in zip(
                era_results, benjamini_yekutieli(p_values), strict=True
            )
            if rejected
        )
    return tuple(rejections_bh), tuple(rejections_by)


def run_battery(
    draws: pd.DataFrame,
    spec: GameSpec,
    seed: int = MASTER_SEED,
    replicates: int = DEFAULT_REPLICATES,
) -> BatteryResult:
    """Run every statistical family for every era present in ``draws``.

    RNG scheme: one base ``default_rng(seed)`` draws seven child seeds per era,
    consumed in the fixed order pairs, runs parity, runs high, serial, gaps,
    order statistics and spectral. The same seed therefore reproduces the same
    battery exactly.
    """
    base_rng = np.random.default_rng(seed)
    results: list[TestResult] = []
    for era in spec.eras:
        frame = (
            draws[draws["era"] == era.name]
            .sort_values(["date", "round"])
            .reset_index(drop=True)
        )
        if frame.empty:
            continue
        child_seeds = base_rng.integers(
            0, np.iinfo(np.int64).max, size=7, dtype=np.int64
        )

        def child(index: int, seeds: np.ndarray = child_seeds) -> np.random.Generator:
            return np.random.default_rng(int(seeds[index]))

        results.append(frequency_test(frame, era, "main"))
        if era.has_bonus:
            results.append(frequency_test(frame, era, "bonus"))
        if era.rounds_per_draw == 2:
            round_one = frame[frame["round"] == 1]
            round_two = frame[frame["round"] == 2]
            results.append(
                homogeneity_test(
                    _main_counts(round_one, era.pool_size),
                    _main_counts(round_two, era.pool_size),
                    era,
                    "main-round1-vs-round2",
                )
            )
            if era.has_bonus:
                results.append(
                    homogeneity_test(
                        _bonus_counts(round_one, era.pool_size),
                        _bonus_counts(round_two, era.pool_size),
                        era,
                        "bonus-round1-vs-round2",
                    )
                )

        results.append(pairs_test(frame, era, child(0), replicates))
        results.append(
            runs_test(
                sum_parity_bits(frame),
                child(1),
                replicates,
                era=era.name,
                subject="sum-parity",
            )
        )
        results.append(
            runs_test(
                sum_high_bits(frame, era),
                child(2),
                replicates,
                era=era.name,
                subject="sum-high",
            )
        )
        results.append(serial_test(frame, era, child(3), replicates))
        results.append(gaps_test(frame, era, child(4), replicates))
        results.append(order_stats_test(frame, era, child(5), replicates))
        results.append(spectral_test(frame, era, child(6), replicates))
        results.append(bayesian_uniformity(frame, era))

    all_results = tuple(results)
    rejections_bh, rejections_by = _rejections(all_results, spec)
    return BatteryResult(
        results=all_results,
        rejections_bh=rejections_bh,
        rejections_by=rejections_by,
    )
