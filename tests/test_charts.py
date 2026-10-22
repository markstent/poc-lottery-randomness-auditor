"""Tests for the chart library."""

import pandas as pd
import pytest

from poc_lottery_randomness_auditor.charts import (
    CHARTS_BY_FAMILY,
    frequency_chart,
    power_chart,
)
from poc_lottery_randomness_auditor.engine.types import TestResult


def make_frequency_result() -> TestResult:
    return TestResult(
        family="frequency",
        era="era-59",
        subject="main",
        statistic=1.0,
        p_value=1.0,
        method="test",
        details={"counts": [10, 20, 30], "expected": 20.0},
    )


def make_result_for(name: str) -> TestResult:
    base = {
        "era": "era-59",
        "subject": "main",
        "statistic": 1.0,
        "p_value": 1.0,
        "method": "test",
    }
    details = {
        "frequency": {"counts": [10, 20, 30], "expected": 20.0},
        "pairs": {"pairs": [(1, 2, 5), (1, 3, 3), (2, 3, 4)], "expected": 4.0},
        "runs": {
            "n_runs": 4,
            "expected_runs": 3.0,
            "trace": [1, 2, 2, 3, 4],
            "z": 1.0,
        },
        "serial": {
            "acf": {"sum": [0.1, -0.05], "range": [0.0, 0.02]},
            "lags": 2,
        },
        "gaps": {
            "ks": [0.1, 0.2, 0.05],
            "overdue": [{"number": 1, "draws_since_seen": 2, "expected_wait": 9.8}],
        },
        "order-stats": {
            "cdfs": {
                "min": {
                    "support": [1, 2, 3],
                    "empirical": [0.0, 0.5, 1.0],
                    "reference": [0.0, 0.4, 1.0],
                },
                "max": {
                    "support": [1, 2, 3],
                    "empirical": [0.0, 0.6, 1.0],
                    "reference": [0.0, 0.5, 1.0],
                },
            }
        },
        "spectral": {
            "peaks": [
                {"number": 1, "frequency": 0.1, "power": 10.0, "ratio": 5.0},
                {"number": 2, "frequency": 0.2, "power": 8.0, "ratio": 4.0},
            ]
        },
        "bayesian": {
            "intervals": [[0.01, 0.02], [0.015, 0.025]],
            "rope_delta": 0.05,
            "credibility": 0.95,
        },
    }
    return TestResult(family=name, details=details[name], **base)


@pytest.fixture
def power_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "family": ["frequency", "frequency"],
            "bias_kind": ["weighted", "weighted"],
            "level": [1.5, 2.5],
            "n_draws": [1000, 2000],
            "power": [0.2, 0.8],
        }
    )


def test_frequency_chart_embeds_engine_values():
    result = make_frequency_result()
    spec = frequency_chart(result).to_dict()
    values = spec["layer"][0]["data"]["values"]
    assert [value["count"] for value in values] == [10, 20, 30]


def test_all_charts_render_valid_specs():
    for name, chart in CHARTS_BY_FAMILY.items():
        spec = chart(make_result_for(name)).to_dict()
        assert "mark" in spec or "layer" in spec


def test_power_chart_valid(power_df):
    mark = power_chart(power_df).to_dict()["mark"]
    mark_type = mark.get("type") if isinstance(mark, dict) else mark
    assert mark_type in {"rect", "bar", "line"}
