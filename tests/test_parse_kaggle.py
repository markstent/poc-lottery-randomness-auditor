"""Tests for parsing the Kaggle full-history Lotto CSV."""

from datetime import date

import pytest

from poc_lottery_randomness_auditor.data import COLUMNS, parse_kaggle

KAGGLE_SAMPLE = """No., Day,DD,MMM,YYYY,N1,N2,N3,N4,N5,N6,BN,   Jackpot, Wins,   Machine  ,Set
3168, Sat, 2,May,2026, 04,29,12,06,03,40,19,   3892070,    1,  Lancelot  ,  3
3167, Wed,29,Apr,2026, 04,52,32,14,38,22,27,   2000000,    0,   Lotto 5  ,  3
"""


def test_parse_kaggle_sorts_main_balls_and_normalises_ids():
    df = parse_kaggle(KAGGLE_SAMPLE)
    assert list(df.columns) == list(COLUMNS)
    first = df.iloc[0]
    assert first["draw_no"] == 3168
    assert first["date"] == date(2026, 5, 2)
    assert first["round"] == 1
    assert [first[f"n{i}"] for i in range(1, 7)] == [3, 4, 6, 12, 29, 40]
    assert first["bonus"] == 19
    assert first["machine"] == "Lancelot"
    assert first["ball_set"] == "L3"
    assert df.iloc[1]["machine"] == "Lotto5"


def test_parse_kaggle_rejects_drift():
    with pytest.raises(ValueError, match="unexpected Kaggle Lotto format"):
        parse_kaggle("draw,number\n1,2\n")
