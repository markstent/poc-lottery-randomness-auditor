"""Snapshot loading, checksum verification and draw validation."""

import hashlib
import itertools
from pathlib import Path

import pandas as pd

from poc_lottery_randomness_auditor.games import GameSpec

COLUMNS: tuple[str, ...] = (
    "draw_no",
    "date",
    "round",
    "n1",
    "n2",
    "n3",
    "n4",
    "n5",
    "n6",
    "bonus",
    "machine",
    "ball_set",
)

_MAIN_BALLS: tuple[str, ...] = ("n1", "n2", "n3", "n4", "n5", "n6")


class ValidationError(Exception):
    """Raised when draw data fails validation."""


class ChecksumError(Exception):
    """Raised when a snapshot file does not match its SHA256SUMS entry."""


def verify_checksums(snapshot_dir: Path) -> None:
    """Verify each file listed in ``SHA256SUMS`` against its sha256 digest.

    Lines use the ``shasum -a 256`` format: ``"<hex>  <filename>"``.

    Raises:
        ChecksumError: If a listed file is missing or its digest differs.
    """
    sums_path = snapshot_dir / "SHA256SUMS"
    for line in sums_path.read_text().splitlines():
        if not line.strip():
            continue
        expected, _, filename = line.partition("  ")
        path = snapshot_dir / filename
        if not path.is_file():
            raise ChecksumError(f"file listed in SHA256SUMS is missing: {filename}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ChecksumError(
                f"checksum mismatch for {filename}: "
                f"expected {expected}, got {actual}"
            )


def load_draws(snapshot_dir: Path, spec: GameSpec) -> pd.DataFrame:
    """Load, validate and era-tag a pinned snapshot of draws.

    Returns:
        A DataFrame with the columns of ``COLUMNS`` plus an ``era`` column of
        era names, sorted by ``(date, round)``.
    """
    verify_checksums(snapshot_dir)
    df = pd.read_csv(snapshot_dir / "draws.csv", parse_dates=["date"])
    df["date"] = df["date"].dt.date
    validate_draws(df, spec)
    df["era"] = df["date"].map(lambda d: spec.era_for_date(d).name)
    return df.sort_values(["date", "round"]).reset_index(drop=True)


def validate_draws(df: pd.DataFrame, spec: GameSpec) -> None:
    """Validate draw rows against the era rules of ``spec``.

    Raises:
        ValidationError: On the first offending row; messages name the
            offending ``draw_no``.
    """
    duplicated = df.duplicated(subset=["draw_no", "round"], keep=False)
    if duplicated.any():
        row = df.loc[duplicated].iloc[0]
        raise ValidationError(
            f"draw {row['draw_no']}: duplicate row for round {row['round']}"
        )

    for row in df.itertuples(index=False):
        balls = [getattr(row, name) for name in _MAIN_BALLS]
        if any(a >= b for a, b in itertools.pairwise(balls)):
            raise ValidationError(
                f"draw {row.draw_no}: main balls not strictly increasing: {balls}"
            )
        if row.bonus in balls:
            raise ValidationError(
                f"draw {row.draw_no}: bonus {row.bonus} appears among main balls"
            )
        try:
            era = spec.era_for_date(row.date)
        except ValueError as exc:
            raise ValidationError(f"draw {row.draw_no}: {exc}") from exc
        out_of_pool = [b for b in (*balls, row.bonus) if not 1 <= b <= era.pool_size]
        if out_of_pool:
            raise ValidationError(
                f"draw {row.draw_no}: balls {out_of_pool} outside"
                f" 1..{era.pool_size} for {era.name}"
            )

    for draw_no, group in df.groupby("draw_no"):
        era = spec.era_for_date(group["date"].iloc[0])
        expected = set(range(1, era.rounds_per_draw + 1))
        actual = set(group["round"])
        if actual != expected:
            raise ValidationError(
                f"draw {draw_no}: rounds {sorted(actual)} do not match expected"
                f" {sorted(expected)} for {era.name}"
            )
