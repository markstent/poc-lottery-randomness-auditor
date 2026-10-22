"""Render the audit battery to HTML and PDF."""

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from vl_convert import vegalite_to_svg

from poc_lottery_randomness_auditor.charts import CHARTS_BY_FAMILY, power_chart
from poc_lottery_randomness_auditor.engine.battery import BatteryResult
from poc_lottery_randomness_auditor.engine.types import TestResult


@dataclass(frozen=True)
class ReportMeta:
    """Provenance and headline metadata for one report."""

    title: str
    generated_at: datetime
    snapshot_sha256: str
    n_draws: int
    n_eras: int


_TEMPLATE_DIR = Path(__file__).parent / "templates"
_ENV = Environment(
    loader=FileSystemLoader(_TEMPLATE_DIR),
    autoescape=True,
    trim_blocks=True,
    lstrip_blocks=True,
)

_FAMILY_NOTES: dict[str, dict[str, str]] = {
    "frequency": {
        "shows": (
            "Each bar is one ball. It compares how often the ball was drawn with "
            "the even share expected under a fair machine."
        ),
        "fair": (
            "Bars should sit close to the expected line. Some balls land above it "
            "and some below it; with this many balls, a spread of a few percent "
            "either way is normal."
        ),
        "worry": (
            "Concern would be one or two balls far outside the everyday scatter, "
            "especially if the same balls stayed high or low across separate eras."
        ),
    },
    "pairs": {
        "shows": (
            "Each point is one pair of balls. It asks whether pairs turn up "
            "together more often, or less often, than two independent fair draws "
            "would produce."
        ),
        "fair": (
            "Points should cloud around zero with no pair standing far apart. "
            "Because hundreds of pairs are checked, a few moderate outliers are "
            "expected by chance."
        ),
        "worry": (
            "Concern would be the same pair repeatedly far from zero, or a whole "
            "group of balls whose pairs lean the same way."
        ),
    },
    "runs": {
        "shows": (
            "This follows the draw sums in date order and looks for streaks: "
            "long runs above the typical sum, or too much flipping up and down."
        ),
        "fair": (
            "The sequence should drift above and below the middle without a "
            "memory. Streaks happen, but their number should match the fair "
            "reference band."
        ),
        "worry": (
            "Concern would be far fewer streaks than expected, which suggests "
            "mechanical cycling, or far more, which suggests over-correction."
        ),
    },
    "serial": {
        "shows": (
            "This checks whether the numbers in one draw help predict anything "
            "about the next draw, such as repeated balls or shifted sums."
        ),
        "fair": (
            "Knowing the last draw should be useless. The lag measures should "
            "sit inside the fair band rather than trend with draw order."
        ),
        "worry": (
            "Concern would be a stable pattern that persists across eras, not a "
            "single lag that looks odd for a short stretch."
        ),
    },
    "gaps": {
        "shows": (
            "For each ball this measures the waiting time between appearances "
            "and compares the whole waiting-time pattern with the fair pattern."
        ),
        "fair": (
            "Short waits are common and long waits are rare, but long waits must "
            "still occur sometimes. The fair band should contain most of the "
            "observed waiting-time curve."
        ),
        "worry": (
            "Concern would be waits that are too regular, or extreme droughts "
            "and clusters that sit outside the band for several balls at once."
        ),
    },
    "order-stats": {
        "shows": (
            "This summarises each draw by its smallest ball, largest ball and "
            "spread, then compares those summaries with fair reference curves."
        ),
        "fair": (
            "The observed curves should track the reference curves closely. "
            "Random draws can favour low or high numbers for a while without "
            "breaking the overall shape."
        ),
        "worry": (
            "Concern would be the curve sitting outside the reference band over "
            "a broad range, rather than a wiggle at one end."
        ),
    },
    "spectral": {
        "shows": (
            "This turns each ball's appearances into a rhythm and checks for "
            "cycles, such as a ball coming up every Nth draw more often than it "
            "should."
        ),
        "fair": (
            "No single cycle should dominate. The largest peaks should look "
            "like the background noise of a fair sequence."
        ),
        "worry": (
            "Concern would be a sharp peak at the same cycle for several balls, "
            "or a peak that survives when the era is split in half."
        ),
    },
    "bayesian": {
        "shows": (
            "Each interval is a direct estimate of one ball's probability. It "
            "shows the estimate together with the uncertainty left by the "
            "finite number of draws."
        ),
        "fair": (
            "The intervals should cover or straddle the fair line. Wider "
            "intervals mean the era is short, not that the lottery failed."
        ),
        "worry": (
            "Concern would be several intervals clearing the same side of the "
            "fair line, which is much harder to explain than one narrow miss."
        ),
    },
}

_DEFAULT_NOTES = {
    "shows": "This chart summarises one fairness check.",
    "fair": "A fair lottery should land inside the reference band most of the time.",
    "worry": "Concern would be a pattern that repeats across balls or eras.",
}


def chart_svg(chart: alt.Chart) -> str:
    """Convert one Altair chart to an SVG string with deterministic IDs."""
    spec = chart.to_dict()
    svg = vegalite_to_svg(spec)
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    prefix = f"chart-{digest[:12]}"
    ordered_ids = dict.fromkeys(re.findall(r'id="([^"]+)"', svg))
    for index, old_id in enumerate(ordered_ids):
        new_id = f"{prefix}-{index}"
        svg = svg.replace(f'id="{old_id}"', f'id="{new_id}"')
        svg = svg.replace(f"url(#{old_id})", f"url(#{new_id})")
        svg = svg.replace(f'href="#{old_id}"', f'href="#{new_id}"')
    return svg


def _chance_sentence(result: TestResult) -> str:
    if result.p_value is None:
        return (
            "This Bayesian check has no single pass/fail line. Read the interval "
            "against the fair band and the width of the era."
        )
    share = result.p_value * 100
    if result.p_value >= 0.5:
        plain = (
            "This is ordinary. If the lottery were fair, results at least this "
            f"odd would appear in most similar histories (about {share:.0f}%)."
        )
    elif result.p_value >= 0.2:
        plain = (
            "This is still common. If the lottery were fair, results at least "
            f"this odd would appear in a substantial minority of similar "
            f"histories (about {share:.0f}%)."
        )
    elif result.p_value >= 0.05:
        plain = (
            "This is mildly unusual, but not rare. If the lottery were fair, "
            f"results at least this odd would appear in about {share:.1f}% of "
            "similar histories."
        )
    else:
        plain = (
            "This is rare. If the lottery were fair, results at least this odd "
            f"would appear in only about {share:.2f}% of similar histories."
        )
    return (
        f"{plain} The verdict also allows for the number of checks run, so one "
        "surprising result on its own is not treated as proof."
    )


def _family_sections(battery: BatteryResult) -> list[dict[str, Any]]:
    sections = []
    rejections = set(battery.rejections_bh)
    for result in battery.results:
        chart_function = CHARTS_BY_FAMILY.get(result.family)
        if chart_function is None:
            continue
        key = f"{result.era}:{result.family}:{result.subject}"
        notes = _FAMILY_NOTES.get(result.family, _DEFAULT_NOTES)
        if result.p_value is None:
            status = "bayesian"
            verdict = (
                "Bayesian summary: judge the intervals against the fair band "
                "rather than a single cut-off."
            )
        elif key in rejections:
            status = "flagged"
            verdict = "Flagged after allowing for the number of checks."
        else:
            status = "ordinary"
            verdict = "Not flagged after allowing for the number of checks."
        sections.append(
            {
                "era": result.era,
                "title": f"{result.family.replace('-', ' ').title()} ({result.subject})",
                "svg": chart_svg(chart_function(result)),
                "shows": notes["shows"],
                "fair": notes["fair"],
                "worry": notes["worry"],
                "status": status,
                "verdict": verdict,
                "chance": _chance_sentence(result),
            }
        )
    return sections


def _era_sections(battery: BatteryResult) -> list[dict[str, Any]]:
    families = _family_sections(battery)
    eras: list[str] = []
    for result in battery.results:
        if result.era not in eras:
            eras.append(result.era)
    sections = []
    rejections = set(battery.rejections_bh)
    for era in eras:
        flagged = any(key.startswith(f"{era}:") for key in rejections)
        sections.append(
            {
                "name": era,
                "headline": (
                    "At least one check needs follow-up. A statistician should "
                    "inspect the raw draws and machine context before any "
                    "conclusion."
                    if flagged
                    else "Nothing here needs follow-up. These checks found the "
                    "ordinary amount of variation expected from a fair machine."
                ),
                "families": [family for family in families if family["era"] == era],
            }
        )
    return sections


def render_html(battery: BatteryResult, power: pd.DataFrame, meta: ReportMeta) -> str:
    """Render the full audit report as a deterministic HTML string."""
    template = _ENV.get_template("report.html.j2")
    power_svg = "" if power.empty else chart_svg(power_chart(power))
    return template.render(
        meta=meta,
        generated_at=meta.generated_at.isoformat(),
        eras=_era_sections(battery),
        power_svg=power_svg,
    )


def write_pdf(html: str, out_path: Path) -> None:
    """Write rendered report HTML to ``out_path`` as a PDF via WeasyPrint."""
    try:
        from weasyprint import HTML
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "WeasyPrint system libraries are unavailable; install Pango/GObject "
            "to enable PDF output."
        ) from exc
    out_path.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html).write_pdf(out_path)
