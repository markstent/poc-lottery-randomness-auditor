"""Shared Altair theme for all audit charts."""

import altair as alt

THEME: dict = {
    "config": {
        "background": "white",
        "font": "Helvetica Neue",
        "range": {
            "category": [
                "#0072B2",
                "#E69F00",
                "#009E73",
                "#D55E00",
                "#CC79A7",
                "#56B4E9",
                "#F0E442",
                "#000000",
            ]
        },
        "axis": {
            "gridColor": "#E6E6E6",
            "labelFontSize": 12,
            "titleFontSize": 13,
        },
        "title": {"fontSize": 16, "subtitleFontSize": 13},
        "view": {"continuousWidth": 640, "continuousHeight": 360},
    }
}


@alt.theme.register("lottery_audit", enable=False)
def _lottery_audit_theme() -> alt.theme.ThemeConfig:
    return alt.theme.ThemeConfig(THEME)


def register_theme() -> None:
    """Enable the shared ``lottery_audit`` Altair theme."""
    alt.theme.enable("lottery_audit")
