"""End-to-end orchestration test on a small synthetic snapshot."""

import hashlib
from datetime import datetime
from pathlib import Path

import pytest

from poc_lottery_randomness_auditor import audit
from poc_lottery_randomness_auditor.engine.power import PowerScenario


@pytest.fixture
def synthetic_snapshot() -> Path:
    return Path("tests/fixtures/snapshot_synth")


def test_end_to_end_deterministic(tmp_path, synthetic_snapshot, monkeypatch):
    # Keep the orchestration test fast and independent of WeasyPrint's system
    # libraries; report.py tests cover real PDF output when Pango is available.
    monkeypatch.setattr(audit, "BATTERY_REPLICATES", 30)
    monkeypatch.setattr(audit, "POWER_REPLICATES", 10)
    monkeypatch.setattr(
        audit,
        "DEFAULT_SCENARIOS",
        [PowerScenario("frequency", "weighted", level=4.0, n_draws=200)],
    )

    def fake_write_pdf(html: str, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(html.encode()).digest()
        out_path.write_bytes(b"%PDF-" + digest[:8])

    monkeypatch.setattr(audit, "write_pdf", fake_write_pdf)
    now = datetime(2026, 10, 21, 12, 0)
    first = audit.run(synthetic_snapshot, tmp_path / "a", seed=1, now=now)
    second = audit.run(synthetic_snapshot, tmp_path / "b", seed=1, now=now)
    assert first.read_bytes() == second.read_bytes()
    assert (tmp_path / "a" / "report.pdf").read_bytes().startswith(b"%PDF-")
