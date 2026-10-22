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

_INTUITIONS: dict[str, str] = {
    "frequency": (
        "This counts how often each ball appeared and compares the counts with "
        "what a fair machine would produce."
    ),
    "pairs": (
        "This checks whether any two balls appear together far more or far less "
        "often than chance allows."
    ),
    "runs": (
        "This looks for suspicious streaks or alternation in the sequence of draw sums."
    ),
    "serial": (
        "This checks whether one draw appears to remember anything about the "
        "previous draws."
    ),
    "gaps": (
        "This compares the waiting time between appearances of each ball with "
        "the fair waiting-time pattern."
    ),
    "order-stats": (
        "This compares the smallest, largest and spread of each draw with the "
        "fair reference curves."
    ),
    "spectral": (
        "This looks for hidden cycles in the appearances of individual balls."
    ),
    "bayesian": (
        "This estimates each ball's probability directly and asks whether the "
        "estimate sits inside the fair band."
    ),
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
        return "This check is summarised with credible intervals rather than a p-value."
    times = max(1, round(1.0 / result.p_value))
    return (
        f"A fair lottery would produce a result this extreme about 1 time in {times}."
    )


def _family_sections(battery: BatteryResult) -> list[dict[str, Any]]:
    sections = []
    rejections = set(battery.rejections_bh)
    for result in battery.results:
        chart_function = CHARTS_BY_FAMILY.get(result.family)
        if chart_function is None:
            continue
        key = f"{result.era}:{result.family}:{result.subject}"
        if key in rejections:
            verdict = "Flagged after false-discovery-rate control."
        else:
            verdict = "Not flagged after false-discovery-rate control."
        sections.append(
            {
                "era": result.era,
                "title": f"{result.family.replace('-', ' ').title()} ({result.subject})",
                "svg": chart_svg(chart_function(result)),
                "intuition": _INTUITIONS.get(
                    result.family,
                    "This chart summarises one fairness check.",
                ),
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
                    "This era is flagged for follow-up."
                    if flagged
                    else "No convincing departure from fairness survived multiple-comparison control."
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
