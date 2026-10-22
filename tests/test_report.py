"""Tests for the HTML and PDF report pipeline."""

from datetime import UTC, datetime

import pandas as pd
import pytest

from poc_lottery_randomness_auditor.charts import frequency_chart
from poc_lottery_randomness_auditor.engine.battery import BatteryResult
from poc_lottery_randomness_auditor.engine.types import TestResult
from poc_lottery_randomness_auditor.report import (
    ReportMeta,
    chart_svg,
    render_html,
    write_pdf,
)


def make_frequency_result() -> TestResult:
    return TestResult(
        family="frequency",
        era="era-59-two-round",
        subject="main",
        statistic=1.0,
        p_value=0.8,
        method="test",
        details={"counts": [10, 20, 30], "expected": 20.0},
    )


@pytest.fixture
def battery_result() -> BatteryResult:
    return BatteryResult(
        results=(make_frequency_result(),),
        rejections_bh=(),
        rejections_by=(),
    )


@pytest.fixture
def power_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "family": ["frequency"],
            "bias_kind": ["weighted"],
            "level": [4.0],
            "n_draws": [3000],
            "power": [0.95],
        }
    )


def make_meta() -> ReportMeta:
    return ReportMeta(
        title="Lotto randomness audit",
        generated_at=datetime(2026, 10, 22, tzinfo=UTC),
        snapshot_sha256="abc123",
        n_draws=10,
        n_eras=1,
    )


def test_chart_svg_outputs_svg():
    svg = chart_svg(frequency_chart(make_frequency_result()))
    assert svg.startswith("<svg") or "<svg" in svg[:200]


def test_render_html_contains_verdicts_and_meta(battery_result, power_df):
    html = render_html(battery_result, power_df, make_meta())
    assert make_meta().snapshot_sha256 in html
    assert "era-59-two-round" in html
    assert html.count("<svg") >= 1


def test_render_html_plain_english_explanations(battery_result, power_df):
    html = render_html(battery_result, power_df, make_meta())
    assert "What this plot shows" in html
    assert "What fair looks like" in html
    assert "What would worry me" in html
    assert "This is ordinary" in html
    assert "most similar histories" in html
    assert "allowing for the number of checks" in html
    assert "1 time in" not in html


def test_render_html_deterministic(battery_result, power_df):
    assert render_html(battery_result, power_df, make_meta()) == render_html(
        battery_result, power_df, make_meta()
    )


def test_write_pdf_produces_pdf(tmp_path):
    try:
        write_pdf("<html><body><h1>Audit</h1></body></html>", tmp_path / "r.pdf")
    except RuntimeError as exc:
        pytest.skip(str(exc))
    assert (tmp_path / "r.pdf").read_bytes().startswith(b"%PDF-")
