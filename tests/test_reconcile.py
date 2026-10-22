"""Tests for field-by-field reconciliation of two draw sources."""

from datetime import date

import pandas as pd
import pytest

from poc_lottery_randomness_auditor.data import COLUMNS, reconcile

ROW_1 = [1, date(2026, 1, 1), 1, 1, 2, 3, 4, 5, 6, 7, "Lotto1", "L1"]
ROW_2 = [2, date(2026, 1, 2), 1, 8, 9, 10, 11, 12, 13, 14, "Lotto2", "L2"]


@pytest.fixture
def primary_df() -> pd.DataFrame:
    return pd.DataFrame([ROW_1, ROW_2], columns=list(COLUMNS))


@pytest.fixture
def secondary_df() -> pd.DataFrame:
    return pd.DataFrame([ROW_1, ROW_2], columns=list(COLUMNS))


@pytest.fixture
def secondary_df_edited() -> pd.DataFrame:
    edited = [
        ROW_1.copy(),
        ROW_2.copy(),
        [9999, date(2026, 1, 3), 1, 1, 2, 3, 4, 5, 6, 7, "Lotto1", "L1"],
    ]
    edited[0][5] = 9  # n3
    return pd.DataFrame(edited, columns=list(COLUMNS))


def test_reconcile_clean(primary_df, secondary_df):
    report = reconcile(primary_df, secondary_df)
    assert report.matched == len(primary_df)
    assert (
        not report.mismatched and not report.only_primary and not report.only_secondary
    )


def test_reconcile_flags_mismatch_and_missing(primary_df, secondary_df_edited):
    report = reconcile(primary_df, secondary_df_edited)
    assert report.mismatched[0].field == "n3"
    assert report.only_secondary == ((9999, 1),)
