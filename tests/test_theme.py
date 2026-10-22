"""Tests for the shared Altair theme."""

import altair as alt

from poc_lottery_randomness_auditor.theme import THEME, register_theme


def test_theme_registers_and_enables():
    register_theme()
    assert alt.theme.active == "lottery_audit"
    assert "config" in THEME
    assert len(THEME["config"]["range"]["category"]) >= 6
