"""Snapshot loading, checksum verification and draw validation."""

import csv
import hashlib
import io
import itertools
from dataclasses import dataclass
from datetime import date
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


@dataclass(frozen=True)
class Mismatch:
    """One differing field between two sources for the same draw round."""

    draw_no: int
    round: int
    field: str
    primary: str
    secondary: str


@dataclass(frozen=True)
class ReconciliationReport:
    """Summary of a field-by-field reconciliation keyed on draw and round."""

    matched: int
    only_primary: tuple[tuple[int, int], ...]
    only_secondary: tuple[tuple[int, int], ...]
    mismatched: tuple[Mismatch, ...]


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
                f"checksum mismatch for {filename}: expected {expected}, got {actual}"
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


_OFFICIAL_COLUMNS: tuple[str, ...] = (
    "DrawDate",
    "Round",
    "Ball 1",
    "Ball 2",
    "Ball 3",
    "Ball 4",
    "Ball 5",
    "Ball 6",
    "Bonus Ball",
    "Guaranteed Millionaire Maker",
    "Ball Set",
    "Machine",
    "DrawNumber",
)

_KAGGLE_COLUMNS: tuple[str, ...] = (
    "No.",
    "Day",
    "DD",
    "MMM",
    "YYYY",
    "N1",
    "N2",
    "N3",
    "N4",
    "N5",
    "N6",
    "BN",
    "Jackpot",
    "Wins",
    "Machine",
    "Set",
)

_MONTHS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


def _csv_rows(text: str, expected: tuple[str, ...], source: str) -> list[list[str]]:
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        raise ValueError(f"unexpected {source} format: empty file")
    header = [cell.strip() for cell in rows[0]]
    if header != list(expected):
        raise ValueError(f"unexpected {source} format: header {header!r}")
    data = [row for row in rows[1:] if any(cell.strip() for cell in row)]
    if not data:
        raise ValueError(f"unexpected {source} format: no data rows")
    bad = [row for row in data if len(row) != len(expected)]
    if bad:
        raise ValueError(f"unexpected {source} format: ragged row {bad[0]!r}")
    return data


def _parse_int(value: str, source: str, column: str) -> int:
    try:
        return int(value.strip())
    except ValueError as exc:
        raise ValueError(f"unexpected {source} format: {column}={value!r}") from exc


def _parse_day_month_year(value: str, source: str) -> date:
    try:
        day, month, year = (part.strip() for part in value.split("-"))
        return date(int(year), _MONTHS[month], int(day))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"unexpected {source} format: date={value!r}") from exc


def _parse_kaggle_date(row: list[str], source: str) -> date:
    try:
        return date(int(row[4]), _MONTHS[row[3].strip()], int(row[2]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"unexpected {source} format: date row {row!r}") from exc


def _normalise_machine(value: str) -> str:
    return "".join(value.split())


def _normalise_ball_set(value: str) -> str:
    cleaned = "".join(value.split()).upper()
    return f"L{cleaned}" if cleaned.isdigit() else cleaned


def parse_official(text: str) -> pd.DataFrame:
    """Parse the official 180 day Lotto CSV download into ``COLUMNS``."""
    rows = _csv_rows(text, _OFFICIAL_COLUMNS, "official draw history")
    records = []
    for row in rows:
        record = {
            "draw_no": _parse_int(row[12], "official draw history", "DrawNumber"),
            "date": _parse_day_month_year(row[0], "official draw history"),
            "round": _parse_int(row[1], "official draw history", "Round"),
            "bonus": _parse_int(row[8], "official draw history", "Bonus Ball"),
            "machine": _normalise_machine(row[11]),
            "ball_set": _normalise_ball_set(row[10]),
        }
        for index, name in enumerate(_MAIN_BALLS, start=2):
            record[name] = _parse_int(row[index], "official draw history", name)
        records.append(record)
    return pd.DataFrame(records, columns=list(COLUMNS))


def parse_kaggle(text: str) -> pd.DataFrame:
    """Parse the Kaggle ``lohi.csv`` full-history file into ``COLUMNS``."""
    rows = _csv_rows(text, _KAGGLE_COLUMNS, "Kaggle Lotto")
    records = []
    for row in rows:
        mains = sorted(
            _parse_int(row[index], "Kaggle Lotto", f"N{index - 4}")
            for index in range(5, 11)
        )
        record = {
            "draw_no": _parse_int(row[0], "Kaggle Lotto", "No."),
            "date": _parse_kaggle_date(row, "Kaggle Lotto"),
            "round": 1,
            "bonus": _parse_int(row[11], "Kaggle Lotto", "BN"),
            "machine": _normalise_machine(row[14]),
            "ball_set": _normalise_ball_set(row[15]),
        }
        record.update(dict(zip(_MAIN_BALLS, mains, strict=True)))
        records.append(record)
    return pd.DataFrame(records, columns=list(COLUMNS))


def reconcile(primary: pd.DataFrame, secondary: pd.DataFrame) -> ReconciliationReport:
    """Reconcile two draw frames field by field on ``(draw_no, round)``."""
    fields = [
        column
        for column in COLUMNS
        if column not in {"draw_no", "round"}
        and column in primary.columns
        and column in secondary.columns
    ]
    primary_rows = {
        (int(row.draw_no), int(row.round)): row
        for row in primary.itertuples(index=False)
    }
    secondary_rows = {
        (int(row.draw_no), int(row.round)): row
        for row in secondary.itertuples(index=False)
    }
    only_primary = tuple(sorted(set(primary_rows) - set(secondary_rows)))
    only_secondary = tuple(sorted(set(secondary_rows) - set(primary_rows)))

    matched = 0
    mismatched = []
    for key in sorted(set(primary_rows) & set(secondary_rows)):
        row_mismatches = []
        for field in fields:
            primary_value = getattr(primary_rows[key], field)
            secondary_value = getattr(secondary_rows[key], field)
            if primary_value != secondary_value:
                row_mismatches.append(
                    Mismatch(
                        draw_no=key[0],
                        round=key[1],
                        field=field,
                        primary=str(primary_value),
                        secondary=str(secondary_value),
                    )
                )
        if row_mismatches:
            mismatched.extend(row_mismatches)
        else:
            matched += 1
    return ReconciliationReport(
        matched=matched,
        only_primary=only_primary,
        only_secondary=only_secondary,
        mismatched=tuple(mismatched),
    )
