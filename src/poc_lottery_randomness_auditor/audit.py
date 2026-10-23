"""One-command audit orchestration."""

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from poc_lottery_randomness_auditor.data import load_draws
from poc_lottery_randomness_auditor.engine.battery import run_battery
from poc_lottery_randomness_auditor.engine.power import DEFAULT_SCENARIOS, power_grid
from poc_lottery_randomness_auditor.engine.simulate import MASTER_SEED
from poc_lottery_randomness_auditor.games import LOTTO
from poc_lottery_randomness_auditor.report import ReportMeta, render_html, write_pdf

# Keep the one-command POC inside its time budget; the engine default remains
# DEFAULT_REPLICATES for library callers that want maximum precision.
BATTERY_REPLICATES = 500
POWER_REPLICATES = 200
_POWER_COLUMNS = ["family", "bias_kind", "level", "n_draws", "power"]


def _snapshot_sha256(snapshot_dir: Path) -> str:
    return hashlib.sha256((snapshot_dir / "draws.csv").read_bytes()).hexdigest()


def _power(draws: pd.DataFrame, seed: int) -> pd.DataFrame:
    frames = [
        power_grid(DEFAULT_SCENARIOS, era, POWER_REPLICATES, seed)
        for era in LOTTO.eras
        if (draws["era"] == era.name).any()
    ]
    if not frames:
        return pd.DataFrame(columns=_POWER_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def run(
    snapshot_dir: Path,
    out_dir: Path,
    seed: int = MASTER_SEED,
    now: datetime | None = None,
) -> Path:
    """Run the audit and write ``report.html`` and ``report.pdf``."""
    draws = load_draws(snapshot_dir, LOTTO)
    battery = run_battery(draws, LOTTO, seed=seed, replicates=BATTERY_REPLICATES)
    power = _power(draws, seed)
    meta = ReportMeta(
        title=f"{LOTTO.name.title()} randomness audit",
        generated_at=now or datetime.now(UTC),
        snapshot_sha256=_snapshot_sha256(snapshot_dir),
        n_draws=len(draws),
        n_eras=int(draws["era"].nunique()),
    )
    html = render_html(battery, power, meta, game=LOTTO)
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "report.html"
    html_path.write_text(html, encoding="utf-8")
    write_pdf(html, out_dir / "report.pdf")
    return html_path
