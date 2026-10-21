# Lottery Randomness Auditor: Design

Date: 2026-10-21
Status: Approved proposal, pending builder check and final sign-off

## 1. Intent and why

A showcase piece, not a compliance audit. It answers the question "is the UK
National Lottery Lotto actually random?" with visible statistical rigour and
creativity, to impress upper management and technical colleagues. No decision
depends on the output. The interest comes from the quality and variety of the
detective work, not from the verdict, which will almost certainly be "yes, it
looks random".

## 2. Goals, non-goals, anti-goals

### Goals

- A polished Marimo notebook, richly annotated and fully self-contained: a
  technical reader can follow every method and decision without the author
  present. Full statistical vocabulary is fine here (p-values, FDR, ROPE).
- A professional HTML report and a PDF report for management: plain English,
  intuition only, no formulas or jargon. Same numbers as the notebook,
  always, because both are generated from the same engine outputs.
- A statistical engine that is era-aware (see section 5) and extensible to
  EuroMillions, Thunderball and international games by configuration, not by
  code changes.
- Reproducibility: one command regenerates everything from a pinned data
  snapshot, deterministically.

### Non-goals

- Predicting numbers or betting advice.
- Auditing the integrity of any organisation; this is a methods showcase.
- Compliance or regulatory use.
- Live data feeds; the analysis runs against a pinned snapshot.

### Anti-goals (failure even if it "works")

1. A statistical error that a colleague could spot.
2. A chart that disagrees with the underlying numbers.
3. Ugly charts.

## 3. Constraints and principles

- Python 3.13, uv, pytest, Ruff. Dependencies already present: numpy, pandas,
  scipy, altair, matplotlib, marimo. Approved new dependencies: `jinja2`,
  `weasyprint`, `vl-convert-python`. No further additions without approval.
- LiteLLM and Context7 are development-time tools only. No LLM calls in the
  build path of the notebook or reports; all outputs deterministic.
- **Simplicity is a first-class requirement.** The code must be as simple as
  possible to follow. Concretely:
  - Flat package layout, functions over classes.
  - State lives in frozen dataclasses; draws live in a pandas DataFrame.
  - No abstract base classes, no plugin registries, no dependency injection,
    no callbacks, no metaprogramming.
  - Configuration is typed Python (frozen dataclasses in one module), not
    YAML.
  - Every module does one thing and can be read top to bottom in minutes.
- Type hints on public interfaces. Ruff clean. pytest for everything important.

## 4. Audiences and deliverables

| Audience | Deliverable | Register |
|---|---|---|
| Management | HTML report, PDF report | Plain English, intuition, attractive charts |
| Technical colleagues | Marimo notebook | Full statistics, annotated, self-contained |

The notebook is the authoring and exploration surface; the reports are
rendered by the same chart functions so notebook and reports can never
disagree.

## 5. Domain facts (verified 2026-10-21)

UK Lotto has three statistical eras. Pooling across era boundaries is
invalid; every analysis is per era, and per machine where relevant.

| Era | Dates | Format | Approx. draws |
|---|---|---|---|
| 1 | 1994-11-19 to 2015-10-07 | 6 of 49 + Bonus, one set per draw | ~2,200 |
| 2 | 2015-10-10 to 2026-06-06 | 6 of 59 + Bonus, one set per draw | ~1,100 |
| 3 | 2026-06-10 onwards | 6 of 59 + Bonus, **two rounds per draw**, two separate machines | ~45 draws (~90 rounds) |

Notes:

- Era 1 was Saturday-only until the Wednesday draw was added in February
  1997; draw cadence is a data property, not an analysis concern, but the
  era split must use draw dates, not assumptions.
- The Bonus Ball is drawn from the remaining balls after the main six, so it
  is marginally uniform but not independent of the main draw. It is analysed
  separately and this subtlety is stated in the notebook.
- Era 3's two rounds are drawn on two separate machines back to back. This
  gives a built-in machine-vs-machine homogeneity check and doubles the
  current-era sample, which is still small: power there is low and the
  write-up must say so honestly.
- Draw counts above are approximate; exact counts come from the snapshot.

Future games (EuroMillions, Thunderball) are configuration only: ball pool,
picks, bonus pool, rounds per draw, era boundaries. Their exact era dates are
verified at adoption time, not assumed now (EuroMillions changed matrix again
in September 2026).

## 6. Scientific methodology

Pre-registered battery, run per era and per machine. Per-test alpha 0.01 with
Benjamini-Hochberg FDR control across the whole battery within an era.
Each notebook chapter: intuition, chart, the maths, verdict.

1. **Frequency uniformity.** Chi-squared goodness of fit on main balls, and
   separately on the Bonus Ball. Detects biased balls.
2. **Pair co-occurrence.** All C(n,2) pairs against expectation. Cells are
   too sparse for asymptotic chi-squared, so p-values come from the Monte
   Carlo engine. Detects correlated selection.
3. **Runs tests.** Wald-Wolfowitz on odd/even and high/low sequences.
   Detects dependence across draws.
4. **Serial independence.** Autocorrelation of draw features (sum, range,
   odd-count) at lags 1..k. Detects machine drift over time.
5. **Gap analysis.** Waits between appearances of each number against the
   geometric distribution. Detects non-memorylessness; also powers the
   "overdue numbers are a myth" narrative.
6. **Order statistics.** Min, max and spread of sorted draws against their
   theoretical distributions. Detects structural oddities.
7. **Spectral analysis.** Periodogram of each number's appearance indicator
   series. Detects cyclic rigging; the creative angle few expect.
8. **Bayesian layer.** Dirichlet-multinomial posterior over ball
   probabilities (conjugate, closed form; no MCMC, no new dependencies).
   Credible intervals per ball and a ROPE-based "practically fair"
   conclusion. ROPE: each ball's probability within 1/n x (1 +/- 5%);
   the 5% is the designer's call and is stated as such.
9. **Monte Carlo engine.** Two jobs:
   - Null reference distributions for every statistic above, with a fixed
     seed.
   - The counterfactual gallery: simulated rigged lotteries (weighted balls,
     sticky pairs, cyclic bias) showing what each test catches at what
     sample size. This is the power analysis chapter: "how biased would a
     ball have to be before we would notice?"

Stated limitations, written into both notebook and report:

- Sample sizes bound detectable bias; era 3 especially (~90 rounds).
- Tests find aggregate bias, not per-draw predictability.
- With ~20+ statistics per era, flukes are expected; hence pre-registration
  and FDR control.

## 7. Technical architecture

```
src/poc_lottery_randomness_auditor/
  games.py         # GameSpec, Era frozen dataclasses; LOTTO definition
  data.py          # load snapshot, verify checksums, validate schema
  engine/
    __init__.py
    frequencies.py # frequency uniformity
    pairs.py       # pair co-occurrence
    runs.py        # runs tests
    serial.py      # autocorrelation
    gaps.py        # gap analysis
    order_stats.py # order statistics
    spectral.py    # periodograms
    bayesian.py    # Dirichlet-multinomial posterior
    simulate.py    # null distributions, rigged-lottery generator, power
    battery.py     # runs the explicit list of tests, returns TestResults
  charts.py        # one function per figure; inputs are engine outputs only
  theme.py         # the single Altair theme
  report.py        # Jinja2 HTML render, WeasyPrint PDF
notebooks/
  lotto_audit.py   # marimo notebook; zero statistical logic, narrative only
scripts/
  fetch_data.py    # downloads sources, reconciles, writes snapshot
data/
  snapshots/       # pinned CSVs + SHA256SUMS
reports/           # generated HTML and PDF
tests/             # mirrors src, plus fixtures
```

Key decisions:

- **One chart code path.** `charts.py` serves notebook and both reports.
  Static export via vl-convert for the PDF. Matplotlib stays available but
  unused unless a chart genuinely needs it.
- **Typed results.** Each engine function takes a DataFrame plus the era's
  GameSpec and returns a frozen `TestResult` (name, era, statistic, p-value
  or posterior summary, verdict, chart payload). `battery.py` is an explicit
  list of calls; reading it shows the whole analysis at a glance.
- **No caching or pipeline framework.** One `audit()` function runs data
  load, battery, charts, report render, end to end.
- **Reproducibility.** Snapshot pinned with a SHA-256 manifest, verified on
  load. Fixed seeds for all simulation. `uv.lock` pins dependencies.
  No network access outside `scripts/fetch_data.py`.

## 8. Data acquisition plan

- **Primary source:** official National Lottery draw history download from
  national-lottery.co.uk. The public page shows only the last 180 days, so
  the first task of phase 1 is verifying what the download endpoint returns
  and capturing the full history from it.
- **Cross-validation source:** one independent full-history dataset (the
  Kaggle "UK National Lottery results 1994-2026" dataset or lottery.co.uk).
  Every draw reconciled field by field against the primary source.
  Discrepancies are listed in a reconciliation report; never silently
  resolved.
- **Snapshot:** reconciled draws written to `data/snapshots/` as CSV with a
  SHA256SUMS manifest. The snapshot is committed; fetching is a manual,
  documented, repeatable step via `scripts/fetch_data.py`.
- **Schema:** draw number, date, era, round (1/2, single-round eras use 1),
  six main numbers, bonus ball. Sorted and uniqueness-checked.

## 9. Validation strategy

Maps to the three anti-goals.

- **Statistical errors:**
  - Known-answer tests: hand-computed cases for every engine function.
  - Planted-anomaly tests: every detector must fire on synthetic rigged data
    (weighted balls, sticky pairs, cyclic bias) and must not fire
    systematically on simulated fair data.
  - Cross-checks against scipy reference implementations where they exist.
- **Incorrect charts:**
  - Charts accept only engine output objects; tests assert chart inputs
    equal engine results for fixed fixtures.
  - No hand-built figures anywhere.
- **Ugly charts:**
  - One bespoke Altair theme in `theme.py`, designed before the first
    figure, applied everywhere.
- **Whole-system:**
  - Reproducibility audit in the final phase: fresh environment, `uv sync`,
    one command, byte-identical outputs except embedded timestamps.
  - `uv run pytest` and `uv run ruff check .` green at the end of every phase.

## 10. Phased implementation plan

Each phase lands tested and lint-clean. No phase starts before the previous
one passes review.

1. **Data layer.** Verify the official download, fetch both sources,
   reconcile, pin the snapshot, `GameSpec` for Lotto with era boundaries,
   schema validation.
2. **Engine core.** Frequencies, pairs, runs, serial, gaps, order stats,
   plus the simulation harness with fixed seeds.
3. **Engine advanced.** Bayesian layer, spectral analysis, power analysis.
4. **Presentation.** Altair theme, chart library, HTML report, PDF report.
5. **Notebook.** Full narrative build-out, chapter per method.
6. **Polish.** Reproducibility audit, documentation, final review.

## 11. What is left to the builder

- The theme's exact palette and typography, subject to the "not ugly" bar.
- Notebook copy and the plain-English phrasings in the report, subject to
  the registers in section 4.
- CSV column naming, decided when the real download format is known.
- Exact lag count and similar minor parameter choices, stated in code
  comments with reasons.
- Whether Kaggle or lottery.co.uk is the cross-validation source, decided by
   which is cleaner to obtain during phase 1.
