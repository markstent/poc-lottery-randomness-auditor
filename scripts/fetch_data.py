"""Fetch, reconcile and pin the UK Lotto draw snapshot."""

from __future__ import annotations

import hashlib
import io
import sys
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd

from poc_lottery_randomness_auditor.data import (
    ReconciliationReport,
    parse_kaggle,
    parse_official,
    reconcile,
    validate_draws,
)
from poc_lottery_randomness_auditor.games import LOTTO

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT / "data" / "snapshots" / "lotto"
OFFICIAL_URL = (
    "https://api-dfe.national-lottery.co.uk/draw-game/results/6/download"
    "?interval=ONE_EIGHTY"
)
KAGGLE_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "deltaromeo/uk-national-lottery-results-1994-2023"
)
CUTOFF = date(2026, 5, 2)
OVERLAP_START = date(2026, 4, 25)


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "lotto-audit-poc"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _kaggle_lohi_csv(zip_bytes: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        return archive.read("lohi.csv").decode("utf-8")


def _without_source_ids(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["machine", "ball_set"])


def _print_report(title: str, report: ReconciliationReport) -> None:
    print(f"{title}: matched={report.matched}")
    if report.only_primary:
        print(f"  only primary: {report.only_primary}")
    if report.only_secondary:
        print(f"  only secondary: {report.only_secondary}")
    for mismatch in report.mismatched:
        print(
            "  mismatch "
            f"draw={mismatch.draw_no} round={mismatch.round} "
            f"field={mismatch.field} primary={mismatch.primary!r} "
            f"secondary={mismatch.secondary!r}"
        )


def main() -> int:
    kaggle_text = _kaggle_lohi_csv(_fetch(KAGGLE_URL))
    official_text = _fetch(OFFICIAL_URL).decode("utf-8")

    historical = parse_kaggle(kaggle_text)
    historical = historical[historical["date"] <= CUTOFF].copy()
    official = parse_official(official_text)
    recent = official[official["date"] > CUTOFF].copy()

    historical_overlap = historical[historical["date"] >= OVERLAP_START].copy()
    official_overlap = official[
        (official["date"] >= OVERLAP_START) & (official["date"] <= CUTOFF)
    ].copy()

    numbers_report = reconcile(
        _without_source_ids(historical_overlap),
        _without_source_ids(official_overlap),
    )
    _print_report("Overlap winning numbers", numbers_report)
    if (
        numbers_report.mismatched
        or numbers_report.only_primary
        or numbers_report.only_secondary
    ):
        print("reconciliation failed; snapshot not written", file=sys.stderr)
        return 1

    id_report = reconcile(historical_overlap, official_overlap)
    if id_report.mismatched:
        _print_report("Overlap machine and ball-set naming (documented)", id_report)

    combined = pd.concat([historical, recent], ignore_index=True)
    combined = combined.sort_values(["date", "round"]).reset_index(drop=True)
    validate_draws(combined, LOTTO)

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    draws_path = SNAPSHOT_DIR / "draws.csv"
    combined.to_csv(draws_path, index=False)
    digest = hashlib.sha256(draws_path.read_bytes()).hexdigest()
    (SNAPSHOT_DIR / "SHA256SUMS").write_text(f"{digest}  draws.csv\n")
    print(f"wrote {len(combined)} rows to {draws_path}")
    print(f"sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
