"""Altair charts built directly from engine ``TestResult.details`` payloads."""

from collections.abc import Callable

import altair as alt
import pandas as pd

from poc_lottery_randomness_auditor.engine.types import TestResult
from poc_lottery_randomness_auditor.theme import register_theme

register_theme()


def _data(records: list[dict]) -> alt.Data:
    return alt.Data(values=records)


def frequency_chart(r: TestResult) -> alt.Chart:
    counts = r.details["counts"]
    records = [
        {"ball": index + 1, "count": int(count)} for index, count in enumerate(counts)
    ]
    bars = (
        alt.Chart(_data(records))
        .mark_bar()
        .encode(
            x=alt.X("ball:O", title="Ball"),
            y=alt.Y("count:Q", title="Count"),
        )
    )
    expected = (
        alt.Chart(_data([{"expected": float(r.details["expected"])}]))
        .mark_rule(color="#D55E00", strokeDash=[4, 4])
        .encode(y="expected:Q")
    )
    return bars + expected


def pairs_chart(r: TestResult) -> alt.Chart:
    records = [
        {"first": first, "second": second, "count": int(count)}
        for first, second, count in r.details["pairs"]
    ]
    return (
        alt.Chart(_data(records))
        .mark_rect()
        .encode(
            x=alt.X("first:O", title="First ball"),
            y=alt.Y("second:O", title="Second ball"),
            color=alt.Color("count:Q", title="Count"),
        )
    )


def runs_chart(r: TestResult) -> alt.Chart:
    trace = r.details["trace"]
    records = [
        {"draw": index + 1, "runs": int(runs)} for index, runs in enumerate(trace)
    ]
    line = (
        alt.Chart(_data(records))
        .mark_line()
        .encode(
            x=alt.X("draw:Q", title="Draw"),
            y=alt.Y("runs:Q", title="Cumulative runs"),
        )
    )
    expected = (
        alt.Chart(_data([{"expected": float(r.details["expected_runs"])}]))
        .mark_rule(color="#D55E00", strokeDash=[4, 4])
        .encode(y="expected:Q")
    )
    return line + expected


def acf_chart(r: TestResult) -> alt.Chart:
    records = [
        {"feature": feature, "lag": lag + 1, "acf": float(value)}
        for feature, values in r.details["acf"].items()
        for lag, value in enumerate(values)
    ]
    return (
        alt.Chart(_data(records))
        .mark_line(point=True)
        .encode(
            x=alt.X("lag:O", title="Lag"),
            y=alt.Y("acf:Q", title="Autocorrelation"),
            color="feature:N",
        )
    )


def gaps_chart(r: TestResult) -> alt.Chart:
    records = [
        {"number": index + 1, "ks": float(value)}
        for index, value in enumerate(r.details["ks"])
    ]
    return (
        alt.Chart(_data(records))
        .mark_bar()
        .encode(
            x=alt.X("number:O", title="Ball"),
            y=alt.Y("ks:Q", title="Gap KS distance"),
        )
    )


def order_stats_chart(r: TestResult) -> alt.Chart:
    records = []
    for statistic, cdfs in r.details["cdfs"].items():
        for kind in ("empirical", "reference"):
            records.extend(
                {
                    "statistic": statistic,
                    "support": support,
                    "kind": kind,
                    "value": float(value),
                }
                for support, value in zip(cdfs["support"], cdfs[kind], strict=True)
            )
    return (
        alt.Chart(_data(records))
        .mark_line()
        .encode(
            x=alt.X("support:Q", title="Value"),
            y=alt.Y("value:Q", title="CDF"),
            color="statistic:N",
            strokeDash="kind:N",
            detail="statistic:N",
        )
    )


def spectral_chart(r: TestResult) -> alt.Chart:
    return (
        alt.Chart(_data([dict(record) for record in r.details["peaks"]]))
        .mark_circle(size=80)
        .encode(
            x=alt.X("number:O", title="Ball"),
            y=alt.Y("ratio:Q", title="Peak-to-median ratio"),
            color=alt.Color("frequency:Q", title="Frequency"),
        )
    )


def posterior_chart(r: TestResult) -> alt.Chart:
    intervals = r.details["intervals"]
    delta = float(r.details["rope_delta"])
    uniform = 1.0 / len(intervals)
    records = []
    for index, (lower, upper) in enumerate(intervals, start=1):
        records.append(
            {
                "ball": str(index),
                "lower": float(lower),
                "upper": float(upper),
                "mid": float((lower + upper) / 2),
            }
        )
    interval_rules = (
        alt.Chart(_data(records))
        .mark_rule()
        .encode(
            x=alt.X("lower:Q", title="Posterior probability"),
            x2="upper:Q",
            y=alt.Y("ball:O", title="Ball"),
        )
    )
    midpoints = (
        alt.Chart(_data(records)).mark_circle(size=40).encode(x="mid:Q", y="ball:O")
    )
    references = (
        alt.Chart(
            _data(
                [
                    {"value": uniform, "label": "fair"},
                    {"value": uniform * (1 - delta), "label": "ROPE low"},
                    {"value": uniform * (1 + delta), "label": "ROPE high"},
                ]
            )
        )
        .mark_rule(strokeDash=[3, 3])
        .encode(x="value:Q", color="label:N")
    )
    return interval_rules + midpoints + references


def power_chart(df: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(_data(df.to_dict("records")))
        .mark_rect()
        .encode(
            x=alt.X("n_draws:O", title="Draws"),
            y=alt.Y("level:O", title="Bias level"),
            color=alt.Color("power:Q", title="Power"),
        )
    )


CHARTS_BY_FAMILY: dict[str, Callable[[TestResult], alt.Chart]] = {
    "frequency": frequency_chart,
    "pairs": pairs_chart,
    "runs": runs_chart,
    "serial": acf_chart,
    "gaps": gaps_chart,
    "order-stats": order_stats_chart,
    "spectral": spectral_chart,
    "bayesian": posterior_chart,
}
