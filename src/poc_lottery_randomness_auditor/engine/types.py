"""Shared result types for statistical engine tests."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TestResult:
    """One statistical test result for one era and subject."""

    family: str
    era: str
    subject: str
    statistic: float
    p_value: float | None
    method: str
    details: dict
