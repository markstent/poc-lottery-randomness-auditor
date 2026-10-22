# Lottery randomness auditor POC

Proof-of-concept that audits the UK Lotto winning-number history for detectable deviations from fairness.

## What it does

- Pins and checksums a draw snapshot. Kaggle `lohi.csv` is primary through 2026-05-02; the official 180 day CSV is primary after that.
- Runs frequency, pairs, runs, serial, gap, order-statistic, spectral and Bayesian checks per rules era, with Benjamini-Hochberg and Benjamini-Yekutieli control.
- Checks the two-round era for round-to-round homogeneity.
- Estimates power against planted weighted-ball, sticky-pair and cyclic biases.
- Writes `reports/report.html` and `reports/report.pdf`, and ships an annotated Marimo notebook.

## Quickstart

Install dependencies:

    uv sync

The snapshot is pinned in `data/snapshots/lotto`. To rebuild it from source:

    uv run python scripts/fetch_data.py

Run the audit:

    uv run poc-lottery-randomness-auditor --snapshot data/snapshots/lotto --out reports
    open reports/report.html

Open the notebook:

    uv run marimo run notebooks/lotto_audit.py

PDF output uses WeasyPrint and needs its system libraries (Pango/GObject). Without them the CLI still writes `report.html` and then stops with a clear error.

## Verification

    uv run pytest
    uv run ruff check .
    uv run ruff format --check .
    bash scripts/verify_repro.sh

`verify_repro.sh` runs the CLI twice with the same seed and timestamp and compares the HTML byte-for-byte; it also compares PDFs when WeasyPrint is available.

## Structure

- `src/poc_lottery_randomness_auditor/games.py` - Lotto era rules.
- `src/poc_lottery_randomness_auditor/data.py` - snapshot loading, checksums and validation.
- `src/poc_lottery_randomness_auditor/engine/` - statistical families, battery, Bayesian layer and power analysis.
- `src/poc_lottery_randomness_auditor/charts.py`, `theme.py` - Altair figures.
- `src/poc_lottery_randomness_auditor/report.py` - HTML and PDF report rendering.
- `src/poc_lottery_randomness_auditor/audit.py` - one-command orchestration.
- `notebooks/lotto_audit.py` - annotated Marimo notebook.
- `scripts/fetch_data.py` - reproducible snapshot rebuild.
- `scripts/verify_repro.sh` - reproducibility check.
- `docs/data-sources.md` - source decision and snapshot provenance.
- `docs/superpowers/plans/2026-10-21-lotto-randomness-auditor.md` - implementation plan.

## Data

Pinned snapshot: `data/snapshots/lotto/draws.csv`, 3256 rows, sha256 `f75080537ca6bfe750b5db254fb60b577ff61798ce2a6b55a986d1dc4fce6f91`. See `docs/data-sources.md`.

## Security

Never commit secrets or production data. LiteLLM manages model access, MCP permissions, project budgets and spend tracking.
