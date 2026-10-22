"""Monte Carlo draw simulation under fair and alternative hypotheses."""

from collections.abc import Callable

import numpy as np

from poc_lottery_randomness_auditor.games import Era

MASTER_SEED: int = 20261021
DEFAULT_REPLICATES: int = 10_000


def _sorted_sample(indices: np.ndarray, picks: int) -> np.ndarray:
    return np.sort(indices[:, :picks] + 1, axis=1)


def _validate_era(era: Era) -> None:
    if era.pool_size < era.picks:
        raise ValueError(f"{era.name}: pool_size must be at least picks")


def fair_draws(era: Era, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n_draws`` fair Lotto boards for ``era``."""
    _validate_era(era)
    uniforms = rng.random((n_draws, era.pool_size))
    return _sorted_sample(np.argsort(uniforms, axis=1), era.picks)


def weighted_draws(
    era: Era,
    weights: np.ndarray,
    n_draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw boards with balls sampled without replacement proportional to weights."""
    _validate_era(era)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (era.pool_size,):
        raise ValueError(f"{era.name}: weights must have shape ({era.pool_size},)")
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError(f"{era.name}: weights must be positive and finite")
    uniforms = rng.random((n_draws, era.pool_size))
    keys = uniforms ** (1.0 / weights)
    return _sorted_sample(np.argpartition(keys, -era.picks, axis=1), era.picks)


def sticky_pair_draws(
    era: Era,
    pair: tuple[int, int],
    boost: float,
    n_draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw boards where ``pair`` is forced into each board with probability boost."""
    _validate_era(era)
    first, second = sorted(pair)
    if (
        first == second
        or not 1 <= first <= era.pool_size
        or not 1 <= second <= era.pool_size
    ):
        raise ValueError(f"{era.name}: invalid sticky pair {pair}")
    if not 0.0 <= boost <= 1.0:
        raise ValueError("boost must be between 0 and 1")

    remaining = np.array(
        [ball for ball in range(1, era.pool_size + 1) if ball not in pair]
    )
    forced = rng.random(n_draws) < boost
    draws = np.empty((n_draws, era.picks), dtype=int)
    for index in range(n_draws):
        if forced[index]:
            rest = rng.choice(remaining, size=era.picks - 2, replace=False)
            draws[index] = np.sort(np.concatenate(([first, second], rest)))
        else:
            draws[index] = fair_draws(era, 1, rng)[0]
    return draws


def cyclic_draws(
    era: Era,
    favoured: int,
    period: int,
    n_draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw boards with ``favoured`` forced into every ``period``-th board."""
    _validate_era(era)
    if not 1 <= favoured <= era.pool_size:
        raise ValueError(f"{era.name}: invalid favoured ball {favoured}")
    if period < 1:
        raise ValueError("period must be at least 1")

    remaining = np.array(
        [ball for ball in range(1, era.pool_size + 1) if ball != favoured]
    )
    draws = np.empty((n_draws, era.picks), dtype=int)
    for index in range(n_draws):
        if (index + 1) % period == 0:
            rest = rng.choice(remaining, size=era.picks - 1, replace=False)
            draws[index] = np.sort(np.concatenate(([favoured], rest)))
        else:
            draws[index] = fair_draws(era, 1, rng)[0]
    return draws


def null_distribution(
    stat: Callable[[np.ndarray], float],
    era: Era,
    n_draws: int,
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate ``stat`` under fair draws for ``replicates`` replicates."""
    if replicates < 0:
        raise ValueError("replicates must be non-negative")
    return np.array(
        [stat(fair_draws(era, n_draws, rng)) for _ in range(replicates)],
        dtype=float,
    )


def mc_p_value(observed: float, null: np.ndarray) -> float:
    """Return the upper-tail Monte Carlo p-value for ``observed``."""
    null = np.asarray(null, dtype=float)
    return float((1 + np.count_nonzero(null >= observed)) / (1 + null.size))
