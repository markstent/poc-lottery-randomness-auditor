"""Tests for parsing the official National Lottery CSV download."""

from datetime import date
from pathlib import Path

import pytest

from poc_lottery_randomness_auditor.data import COLUMNS, parse_official


def test_parse_official_fixture():
    df = parse_official(Path("tests/fixtures/official_sample.txt").read_text())
    assert len(df) >= 2
    assert list(df.columns) == list(COLUMNS)
    first = df.iloc[0]
    assert first["date"] == date(2026, 10, 17)
    assert first["round"] == 1
    assert [first[f"n{i}"] for i in range(1, 7)] == [6, 19, 30, 45, 46, 57]
    assert first["bonus"] == 50
    assert first["draw_no"] == 3216


def test_parse_official_rejects_drift():
    with pytest.raises(ValueError, match="unexpected official draw history format"):
        parse_official("draw,number\n1,2\n")
