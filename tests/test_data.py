"""Tests for snapshot loading, checksum verification and draw validation."""

import shutil
from pathlib import Path

import pandas as pd
import pytest

from poc_lottery_randomness_auditor.data import (
    ChecksumError,
    ValidationError,
    load_draws,
    validate_draws,
)
from poc_lottery_randomness_auditor.games import LOTTO

MINI_SNAPSHOT = Path(__file__).parent / "fixtures" / "snapshot_mini"


@pytest.fixture
def mini_snapshot() -> Path:
    return MINI_SNAPSHOT


@pytest.fixture
def mini_snapshot_tampered(tmp_path: Path) -> Path:
    dest = tmp_path / "snapshot_mini"
    shutil.copytree(MINI_SNAPSHOT, dest)
    csv_path = dest / "draws.csv"
    csv_path.write_text(csv_path.read_text().replace(",Arthur,", ",Mordred,", 1))
    return dest


@pytest.fixture
def df_with_n1_equals_n2(mini_snapshot: Path) -> pd.DataFrame:
    df = load_draws(mini_snapshot, LOTTO)
    mask = df["draw_no"] == 2001
    df.loc[mask, "n1"] = df.loc[mask, "n2"]
    return df


@pytest.fixture
def df_with_50_in_era49(mini_snapshot: Path) -> pd.DataFrame:
    df = load_draws(mini_snapshot, LOTTO)
    df.loc[df["draw_no"] == 2000, "n6"] = 50
    return df


@pytest.fixture
def df_era3_missing_round2(mini_snapshot: Path) -> pd.DataFrame:
    df = load_draws(mini_snapshot, LOTTO)
    keep = ~((df["draw_no"] == 2006) & (df["round"] == 2))
    return df[keep].reset_index(drop=True)


def test_loads_mini_snapshot_with_eras(mini_snapshot):
    df = load_draws(mini_snapshot, LOTTO)
    assert list(df["era"].unique()) == ["era-49", "era-59", "era-59-two-round"]


def test_checksum_tampering_raises(mini_snapshot_tampered):
    with pytest.raises(ChecksumError):
        load_draws(mini_snapshot_tampered, LOTTO)


def test_duplicate_ball_in_draw_raises(df_with_n1_equals_n2):
    with pytest.raises(ValidationError, match="2001"):
        validate_draws(df_with_n1_equals_n2, LOTTO)


def test_ball_outside_era_pool_raises(df_with_50_in_era49):
    with pytest.raises(ValidationError):
        validate_draws(df_with_50_in_era49, LOTTO)


def test_era3_requires_both_rounds(df_era3_missing_round2):
    with pytest.raises(ValidationError, match="round"):
        validate_draws(df_era3_missing_round2, LOTTO)
