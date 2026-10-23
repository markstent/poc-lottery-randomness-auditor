# Lottery randomness auditor

![Status](https://img.shields.io/badge/status-proof%20of%20concept-f39c12)
![Python](https://img.shields.io/badge/python-3.13%2B-3776AB)
![uv](https://img.shields.io/badge/package%20manager-uv-261230)
![Ruff](https://img.shields.io/badge/lint%20%26%20format-ruff-46AEF7)
![pytest](https://img.shields.io/badge/tests-pytest-0A9EDC)
![Marimo](https://img.shields.io/badge/notebook-marimo-1C7832)
![Reproducible output](https://img.shields.io/badge/HTML%20output-byte%20identical-2E8540)

A reproducible proof-of-concept that audits the pinned UK Lotto winning-number snapshot for detectable deviations from fairness.

The audit tests the published numbers only. A pass means the battery found no convincing deviation in this snapshot after multiple-comparison control. It is not a guarantee about machines, ball sets, procedures or future draws.

## At a glance

- Input: pinned snapshot in `data/snapshots/lotto`, checksummed and validated on load.
- Method: era-aware statistical battery with Monte Carlo references and Bayesian summaries.
- Output: `reports/report.html`, optional `reports/report.pdf`, and an annotated Marimo notebook.
- Reproducibility: fixed seed and timestamp produce byte-identical HTML.

## What the battery checks

| Area | Question |
| --- | --- |
| Frequency | Are any balls drawn too often, or not often enough? |
| Homogeneity | Do same-night rounds come from the same process in the two-round era? |
| Pairs | Are any ball pairs unusually sticky or avoidant? |
| Runs | Is the sequence too streaky or too regular? |
| Serial dependence | Does one draw leak information into later draws? |
| Gaps | Do waiting times between appearances match the fair geometric pattern? |
| Order statistics | Do the minimum, maximum and range follow fair reference curves? |
| Spectral | Is there a hidden cycle in ball appearances? |
| Bayesian uniformity | How far are estimated ball probabilities from the fair share? |

The report explains every chart in three blocks: what the plot shows, what fair looks like, and what would worry me.

## Era model

Rules define the fair reference, so eras are never pooled.

| Era | Dates | Rules |
| --- | --- | --- |
| `era-49` | 1994-11-19 to 2015-10-07 | 49 balls, pick 6, bonus ball, one round per draw |
| `era-59` | 2015-10-10 to 2026-06-06 | 59 balls, pick 6, bonus ball, one round per draw |
| `era-59-two-round` | 2026-06-10 onward | 59 balls, pick 6, bonus ball, two rounds per draw |

## Quickstart

Install dependencies:

```bash
uv sync --frozen
```

Run the audit:

```bash
uv run poc-lottery-randomness-auditor --snapshot data/snapshots/lotto --out reports
open reports/report.html
```

Open the notebook in edit mode:

```bash
uv run marimo edit notebooks/lotto_audit.py
```

Open the notebook as an app:

```bash
uv run marimo run notebooks/lotto_audit.py
```

For agent or external-editor workflows, watch the file and rerun stale cells in the browser:

```bash
uv run marimo edit --watch notebooks/lotto_audit.py
```

## Outputs

`reports/` is gitignored.

- `reports/report.html`: deterministic audit report with inline SVG charts.
- `reports/report.pdf`: written by the CLI through WeasyPrint when Pango/GObject are available.

On machines without Pango/GObject, the CLI still writes `report.html` and then stops with a clear WeasyPrint error. A PDF can still be produced from the HTML with a browser print-to-PDF workflow.

## Verification

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
bash scripts/verify_repro.sh
```

`verify_repro.sh` runs the CLI twice with the same seed and timestamp, compares HTML byte-for-byte, and compares PDFs when WeasyPrint system libraries are available.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/poc_lottery_randomness_auditor/games.py` | Lotto era rules |
| `src/poc_lottery_randomness_auditor/data.py` | Snapshot loading, checksums and validation |
| `src/poc_lottery_randomness_auditor/engine/` | Statistical families, battery, Bayesian layer and power analysis |
| `src/poc_lottery_randomness_auditor/charts.py` | Altair figures |
| `src/poc_lottery_randomness_auditor/theme.py` | Shared chart theme |
| `src/poc_lottery_randomness_auditor/report.py` | HTML and PDF report rendering |
| `src/poc_lottery_randomness_auditor/audit.py` | One-command orchestration |
| `notebooks/lotto_audit.py` | Annotated Marimo notebook |
| `scripts/fetch_data.py` | Rebuild the pinned snapshot from source |
| `scripts/verify_repro.sh` | Reproducibility check |
| `docs/data-sources.md` | Source decision and snapshot provenance |

## Data provenance

Pinned snapshot: `data/snapshots/lotto/draws.csv`, 3256 rows, sha256 `f75080537ca6bfe750b5db254fb60b577ff61798ce2a6b55a986d1dc4fce6f91`.

Kaggle `lohi.csv` is primary through 2026-05-02. The official 180 day CSV is primary after that. See `docs/data-sources.md` for the reconciliation decision.

## Security and scope

Never commit secrets or production data. LiteLLM manages model access, MCP permissions, project budgets and spend tracking. This repository is a proof-of-concept, not a certification of any lottery operator.
