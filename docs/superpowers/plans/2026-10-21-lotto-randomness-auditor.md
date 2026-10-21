# Lottery Randomness Auditor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible statistical audit of UK Lotto draw history with a Marimo notebook and HTML/PDF reports, per the approved design.

**Architecture:** Flat package: `games.py` (era config), `data.py` (snapshot load/validate), `engine/` (one module per test family plus `simulate.py` Monte Carlo and `battery.py` orchestration), `charts.py` + `theme.py` (single chart code path), `report.py` (Jinja2 HTML, WeasyPrint PDF), `audit.py` (one-command regeneration). Notebook contains narrative only.

**Tech Stack:** Python 3.13, uv, pandas, numpy, scipy, altair 6, marimo, jinja2, weasyprint, vl-convert-python, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-10-21-lotto-randomness-auditor-design.md`

## Global Constraints

- Python >= 3.13; dependencies managed only via `uv add`. New deps allowed: `jinja2`, `weasyprint`, `vl-convert-python`. Nothing else without approval.
- Simplicity: flat layout, functions over classes, frozen dataclasses only, no ABCs, no plugin registries, no DI, no YAML config.
- No network access outside `scripts/fetch_data.py`. No LLM calls anywhere in the build path.
- All randomness via `numpy.random.Generator` created from explicit seeds; master seed `20261021`.
- Snapshot data is committed under `data/snapshots/` with a `SHA256SUMS` manifest, verified on every load.
- British English and no em dashes in all copy (notebook, reports, docs).
- `uv run pytest` and `uv run ruff check .` must pass at the end of every task.
- Tests never touch the network; parsers are tested against committed fixtures.

## Review Focus

1. Bonus ball leaking into main-ball analyses would invalidate uniformity claims; expected behaviour: bonus analysed only as its own subject. Pinned in Task 5.
2. Era 3 draws missing round 2, or carrying round values outside {1,2}, must fail loudly at load time, not silently skew the homogeneity check. Pinned in Task 2.
3. Dependent p-values make plain Benjamini-Hochberg attackable; expected behaviour: Benjamini-Yekutieli reported alongside as sensitivity. Pinned in Task 13.
4. Hidden non-determinism (unseeded RNG, set iteration order, wall-clock timestamps) would break the reproducibility claim; expected behaviour: identical HTML on repeat runs with the same seed and fixed clock. Pinned in Tasks 13 and 18.
5. Upstream source format drift must produce a descriptive parse error at fetch time, never a malformed snapshot. Pinned in Task 3.

---

### Task 1: Game definitions and Lotto eras (`games.py`)

**Files:**
- Create: `src/poc_lottery_randomness_auditor/games.py`
- Test: `tests/test_games.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `@dataclass(frozen=True) class Era`: fields `name: str, start: date, end: date | None, pool_size: int, picks: int, has_bonus: bool, rounds_per_draw: int`.
  - `@dataclass(frozen=True) class GameSpec`: field `name: str, eras: tuple[Era, ...]`; method `era_for_date(self, d: date) -> Era` raising `ValueError` when no era contains `d`.
  - Constant `LOTTO: GameSpec`.

- [ ] **Step 1: Write the failing test**

```python
from datetime import date
import pytest
from poc_lottery_randomness_auditor.games import LOTTO

def test_era_boundaries():
    assert LOTTO.era_for_date(date(1994, 11, 19)).pool_size == 49
    assert LOTTO.era_for_date(date(2015, 10, 7)).pool_size == 49
    assert LOTTO.era_for_date(date(2015, 10, 10)).pool_size == 59
    assert LOTTO.era_for_date(date(2026, 6, 6)).rounds_per_draw == 1
    era3 = LOTTO.era_for_date(date(2026, 6, 10))
    assert era3.rounds_per_draw == 2 and era3.end is None

def test_dates_outside_eras_raise():
    with pytest.raises(ValueError):
        LOTTO.era_for_date(date(1994, 11, 18))
    with pytest.raises(ValueError):
        LOTTO.era_for_date(date(2015, 10, 8))  # gap day between eras
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_games.py -v`
Expected: FAIL, `ModuleNotFoundError` or missing attribute.

- [ ] **Step 3: Implement `games.py`**

Three eras: `("era-49", 1994-11-19, 2015-10-07, 49, 6, True, 1)`, `("era-59", 2015-10-10, 2026-06-06, 59, 6, True, 1)`, `("era-59-two-round", 2026-06-10, None, 59, 6, True, 2)`. `era_for_date` linear-scans the tuple.

- [ ] **Step 4: Run tests, then ruff**

Run: `uv run pytest tests/test_games.py -v && uv run ruff check .`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/poc_lottery_randomness_auditor/games.py tests/test_games.py
git commit -m "feat: add GameSpec and Lotto era definitions"
```

---

### Task 2: Snapshot loading and validation (`data.py`)

**Files:**
- Create: `src/poc_lottery_randomness_auditor/data.py`
- Create: `tests/fixtures/snapshot_mini/draws.csv` and `tests/fixtures/snapshot_mini/SHA256SUMS`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: `GameSpec`, `Era` from Task 1.
- Produces:
  - `COLUMNS: tuple[str, ...]` = `("draw_no", "date", "round", "n1", "n2", "n3", "n4", "n5", "n6", "bonus", "machine", "ball_set")`.
  - `class ValidationError(Exception)`, `class ChecksumError(Exception)`.
  - `verify_checksums(snapshot_dir: Path) -> None` raising `ChecksumError` on mismatch.
  - `load_draws(snapshot_dir: Path, spec: GameSpec) -> pd.DataFrame`: verifies checksums, reads `draws.csv` (parse `date` as datetime date), validates, appends an `era` column (era name), returns sorted by `(date, round)`.
  - `validate_draws(df: pd.DataFrame, spec: GameSpec) -> None` raising `ValidationError` naming the offending `draw_no`.

- [ ] **Step 1: Write the failing tests**

```python
def test_loads_mini_snapshot_with_eras(mini_snapshot):
    df = load_draws(mini_snapshot, LOTTO)
    assert list(df["era"].unique()) == ["era-49", "era-59", "era-59-two-round"]

def test_checksum_tampering_raises(mini_snapshot_tampered):
    with pytest.raises(ChecksumError):
        load_draws(mini_snapshot_tampered, LOTTO)

def test_duplicate_ball_in_draw_raises(df_with_n1_equals_n2):
    with pytest.raises(ValidationError, match="2001"):
        validate_draws(df_with_n1_equals_n2, LOTTO)

def test_ball_outside_era_pool_raises(df_with_50_in_era49):
    with pytest.raises(ValidationError):
        validate_draws(df_with_50_in_era49, LOTTO)

def test_era3_requires_both_rounds(df_era3_missing_round2):
    with pytest.raises(ValidationError, match="round"):
        validate_draws(df_era3_missing_round2, LOTTO)
```

Fixtures: `snapshot_mini` holds ~9 draws (3 per era, era 3 as round pairs); tampered/dodgy variants built by test fixtures copying and editing rows. Validation also checks: main six strictly increasing per row, bonus not among main six, numbers within 1..`pool_size`, `round` values exactly `{1..rounds_per_draw}` per draw in that era, unique `(draw_no, round)`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_data.py -v`
Expected: FAIL, module missing.

- [ ] **Step 3: Implement `data.py`**

Checksums via `hashlib.sha256` compared against `SHA256SUMS` lines (`"<hex>  <filename>"`). Validation as a sequence of explicit checks with row context in messages. Era assignment via `spec.era_for_date` vectorised through `df["date"].map`.

- [ ] **Step 4: Run tests, then ruff** — expect PASS.

- [ ] **Step 5: Commit**

```bash
git add src/poc_lottery_randomness_auditor/data.py tests/test_data.py tests/fixtures/
git commit -m "feat: add snapshot loading, checksums and draw validation"
```

---

### Task 3: Data acquisition (`scripts/fetch_data.py`, reconciliation, source recon)

**Files:**
- Create: `scripts/fetch_data.py`
- Create: `docs/data-sources.md`
- Create: `tests/fixtures/official_sample.txt` (first ~15 lines of the real official download, saved during Step 1)
- Modify: `src/poc_lottery_randomness_auditor/data.py` (add reconcile + parser)
- Test: `tests/test_reconcile.py`, `tests/test_parse_official.py`

**Interfaces:**
- Consumes: `COLUMNS`, `validate_draws` from Task 2.
- Produces:
  - `parse_official(text: str) -> pd.DataFrame` (columns = `COLUMNS` minus era) raising `ValueError("unexpected official draw history format: ...")` on unrecognised input.
  - `@dataclass(frozen=True) class Mismatch`: `draw_no: int, round: int, field: str, primary: str, secondary: str`.
  - `@dataclass(frozen=True) class ReconciliationReport`: `matched: int, only_primary: tuple[tuple[int, int], ...], only_secondary: tuple[tuple[int, int], ...], mismatched: tuple[Mismatch, ...]`.
  - `reconcile(primary: pd.DataFrame, secondary: pd.DataFrame) -> ReconciliationReport` keyed on `(draw_no, round)`.

- [ ] **Step 1: Verify the official source (manual recon, checkable result)**

Open `https://www.national-lottery.co.uk/results/lotto/draw-history`, follow "Download draw history", record in `docs/data-sources.md`: the download URL, format, earliest draw covered, whether machine/ball-set IDs are present, and the chosen cross-validation source (Kaggle dataset or lottery.co.uk, whichever is cleaner to obtain). If the official download lacks full history, record that and promote the cross-validation source to primary with the official site used for spot checks. Save the first ~15 lines as `tests/fixtures/official_sample.txt`.
Expected: `docs/data-sources.md` exists and answers all five points.

- [ ] **Step 2: Write the failing tests**

```python
def test_parse_official_fixture():
    df = parse_official(Path("tests/fixtures/official_sample.txt").read_text())
    assert len(df) >= 2 and list(df.columns) == list(COLUMNS)

def test_parse_official_rejects_drift():
    with pytest.raises(ValueError, match="unexpected official draw history format"):
        parse_official("draw,number\n1,2\n")

def test_reconcile_clean(primary_df, secondary_df):
    report = reconcile(primary_df, secondary_df)
    assert report.matched == len(primary_df)
    assert not report.mismatched and not report.only_primary and not report.only_secondary

def test_reconcile_flags_mismatch_and_missing(primary_df, secondary_df_edited):
    report = reconcile(primary_df, secondary_df_edited)
    assert report.mismatched[0].field == "n3"
    assert report.only_secondary == [(9999, 1)]
```

- [ ] **Step 3: Run tests to verify they fail** — expect FAIL on import.

- [ ] **Step 4: Implement parser and `reconcile`**

Parser maps the real format recorded in Step 1 to `COLUMNS`, normalising date to ISO. `reconcile` outer-joins on `(draw_no, round)` and diffs shared fields.

- [ ] **Step 5: Run tests, then ruff** — expect PASS.

- [ ] **Step 6: Implement `scripts/fetch_data.py` and run it for real**

Thin script: fetch primary, fetch secondary, `reconcile`, print the report, write `data/snapshots/lotto/draws.csv` + `SHA256SUMS` only when `mismatched` is empty (otherwise exit 1 and print the discrepancy table). Then run it, inspect the output, and commit the snapshot.
Run: `uv run python scripts/fetch_data.py`
Expected: reconciliation report printed with zero mismatches (or each mismatch investigated and resolved with a note in `docs/data-sources.md`), snapshot written.

- [ ] **Step 7: Commit**

```bash
git add scripts/fetch_data.py docs/data-sources.md data/snapshots/ tests/test_reconcile.py tests/test_parse_official.py tests/fixtures/official_sample.txt src/poc_lottery_randomness_auditor/data.py
git commit -m "feat: add data acquisition, reconciliation and pinned Lotto snapshot"
```

---

### Task 4: Engine foundations (`engine/types.py`, `engine/simulate.py`)

**Files:**
- Create: `src/poc_lottery_randomness_auditor/engine/__init__.py`, `src/poc_lottery_randomness_auditor/engine/types.py`, `src/poc_lottery_randomness_auditor/engine/simulate.py`
- Test: `tests/engine/test_simulate.py`

**Interfaces:**
- Consumes: `Era` from Task 1.
- Produces:
  - `@dataclass(frozen=True) class TestResult`: `family: str, era: str, subject: str, statistic: float, p_value: float | None, method: str, details: dict`.
  - `MASTER_SEED: int = 20261021`, `DEFAULT_REPLICATES: int = 10_000`.
  - `fair_draws(era: Era, n_draws: int, rng: np.random.Generator) -> np.ndarray` shape `(n_draws, era.picks)`, rows sorted, values in `1..pool_size`.
  - `weighted_draws(era, weights: np.ndarray, n_draws, rng) -> np.ndarray`, `sticky_pair_draws(era, pair: tuple[int, int], boost: float, n_draws, rng) -> np.ndarray`, `cyclic_draws(era, favoured: int, period: int, n_draws, rng) -> np.ndarray` (same shape contract).
  - `null_distribution(stat: Callable[[np.ndarray], float], era: Era, n_draws: int, replicates: int, rng: np.random.Generator) -> np.ndarray`.
  - `mc_p_value(observed: float, null: np.ndarray) -> float` = `(1 + count(null >= observed)) / (1 + len(null))`.

- [ ] **Step 1: Write the failing tests**

```python
def test_fair_draws_contract():
    rng = np.random.default_rng(1)
    draws = fair_draws(LOTTO.eras[1], 100, rng)
    assert draws.shape == (100, 6)
    assert draws.min() >= 1 and draws.max() <= 59
    assert all(len(set(row)) == 6 for row in draws)

def test_seeded_reproducibility():
    a = fair_draws(LOTTO.eras[1], 50, np.random.default_rng(7))
    b = fair_draws(LOTTO.eras[1], 50, np.random.default_rng(7))
    assert np.array_equal(a, b)

def test_weighted_draws_detectably_bias():
    w = np.full(59, 1.0); w[0] = 10.0
    draws = weighted_draws(LOTTO.eras[1], w, 4000, np.random.default_rng(3))
    p = scipy.stats.chisquare(np.bincount(draws.ravel(), minlength=60)[1:]).pvalue
    assert p < 1e-6

def test_mc_p_value_bounds():
    null = np.arange(1000.0)
    assert mc_p_value(9999.0, null) == pytest.approx(1 / 1001)
    assert mc_p_value(-1.0, null) == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests to verify they fail** — expect FAIL.

- [ ] **Step 3: Implement `types.py` and `simulate.py`**

Sampling without replacement via `rng.choice(pool, size=(n, picks), replace=False)` only if it supports weights per row cleanly; otherwise a documented per-row loop or `rng.random` argsort trick (`(-log(u)/w).argsort` for weighted sampling without replacement). Sticky pairs: sample with the pair forced in with probability `boost`, else fair. Cyclic: on every `period`-th draw, force `favoured` into the draw.

- [ ] **Step 4: Run tests, then ruff** — expect PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat: add engine types and Monte Carlo simulation harness"`.

---

### Task 5: Frequency uniformity (`engine/frequencies.py`)

**Files:**
- Create: `src/poc_lottery_randomness_auditor/engine/frequencies.py`
- Test: `tests/engine/test_frequencies.py`

**Interfaces:**
- Consumes: `TestResult` (Task 4), `Era` (Task 1).
- Produces:
  - `chi2_uniform(counts: np.ndarray) -> tuple[float, float]` (statistic, p-value).
  - `frequency_test(draws_df: pd.DataFrame, era: Era, subject: str) -> TestResult` where `subject` is `"main"` or `"bonus"`; `details` carries `{"counts": list[int], "expected": float}`.
  - `homogeneity_test(counts_a: np.ndarray, counts_b: np.ndarray, era: Era, subject: str) -> TestResult` (two-sample chi-squared, for era 3 round-vs-round or machine-vs-machine).

- [ ] **Step 1: Write the failing tests**

```python
def test_chi2_uniform_known_answer():
    stat, p = chi2_uniform(np.full(49, 120))
    assert stat == pytest.approx(0.0) and p == pytest.approx(1.0)

def test_chi2_matches_scipy():
    counts = np.array([17, 23, 11, 29, 20])
    stat, p = chi2_uniform(counts)
    ref = scipy.stats.chisquare(counts)
    assert stat == pytest.approx(ref.statistic, rel=1e-12)
    assert p == pytest.approx(ref.pvalue, rel=1e-12)

def test_bonus_never_leaks_into_main(era2_draws_df):
    r1 = frequency_test(era2_draws_df, LOTTO.eras[1], "main")
    shuffled = era2_draws_df.assign(bonus=np.random.default_rng(0).permutation(era2_draws_df["bonus"]))
    r2 = frequency_test(shuffled, LOTTO.eras[1], "main")
    assert r1.statistic == r2.statistic

def test_planted_weight_bias_fires():
    w = np.full(59, 1.0); w[4] = 4.0
    draws = weighted_draws(LOTTO.eras[1], w, 5000, np.random.default_rng(11))
    df = draws_to_df(draws)  # test helper building COLUMNS-shaped df
    assert frequency_test(df, LOTTO.eras[1], "main").p_value < 0.01
```

- [ ] **Step 2: Run tests to verify they fail** — expect FAIL.

- [ ] **Step 3: Implement `frequencies.py`**

Thin wrappers over `scipy.stats.chisquare` with counts built only from `n1..n6` for `main`, only from `bonus` for `bonus`. Homogeneity via `scipy.stats.chi2_contingency` on the two count vectors.

- [ ] **Step 4: Run tests, then ruff** — expect PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat: add frequency uniformity and homogeneity tests"`.

---

### Task 6: Pair co-occurrence (`engine/pairs.py`)

**Files:**
- Create: `src/poc_lottery_randomness_auditor/engine/pairs.py`
- Test: `tests/engine/test_pairs.py`

**Interfaces:**
- Consumes: `fair_draws`, `null_distribution`, `mc_p_value`, `DEFAULT_REPLICATES` (Task 4); `TestResult`; `Era`.
- Produces:
  - `pair_counts(draws_df: pd.DataFrame, era: Era) -> np.ndarray` symmetric `(pool, pool)` zero-diagonal counts over `n1..n6`.
  - `expected_pair_count(n_draws: int, era: Era) -> float` = `n_draws * comb(pool - 2, picks - 2) / comb(pool, picks)`.
  - `pairs_omnibus(counts: np.ndarray, expected: float) -> float` = max absolute standardised residual over pairs.
  - `pairs_test(draws_df, era, rng, replicates: int = DEFAULT_REPLICATES) -> TestResult` (Monte Carlo p-value on the omnibus).

- [ ] **Step 1: Write the failing tests**

```python
def test_expected_pair_count_exact():
    assert expected_pair_count(100, LOTTO.eras[1]) == pytest.approx(100 * comb(57, 4) / comb(59, 6))

def test_omnibus_zero_at_exact_expectation():
    era = LOTTO.eras[1]
    counts = np.full((59, 59), 10.0); np.fill_diagonal(counts, 0.0)
    assert pairs_omnibus(counts, 10.0) == pytest.approx(0.0)

def test_planted_sticky_pair_fires():
    draws = sticky_pair_draws(LOTTO.eras[1], (7, 42), 0.3, 3000, np.random.default_rng(5))
    r = pairs_test(draws_to_df(draws), LOTTO.eras[1], np.random.default_rng(6), replicates=1000)
    assert r.p_value < 0.01

def test_fair_seed_does_not_fire():
    draws = fair_draws(LOTTO.eras[1], 3000, np.random.default_rng(8))
    r = pairs_test(draws_to_df(draws), LOTTO.eras[1], np.random.default_rng(9), replicates=1000)
    assert r.p_value > 0.01
```

- [ ] **Step 2: Run tests to verify they fail** — expect FAIL.

- [ ] **Step 3: Implement `pairs.py`**

Omnibus statistic fed to `null_distribution` over `fair_draws`; standardised residual `(obs - exp) / sqrt(exp)`; details carry the upper-triangle pair list for the heatmap.

- [ ] **Step 4: Run tests, then ruff** — expect PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat: add pair co-occurrence omnibus test"`.

---

### Task 7: Runs tests (`engine/runs.py`)

**Files:**
- Create: `src/poc_lottery_randomness_auditor/engine/runs.py`
- Test: `tests/engine/test_runs.py`

**Interfaces:**
- Consumes: `mc_p_value`, `DEFAULT_REPLICATES`; `TestResult`; `Era`.
- Produces:
  - `sum_parity_bits(draws_df: pd.DataFrame) -> np.ndarray` (bool per draw: sum of main six is odd).
  - `sum_high_bits(draws_df: pd.DataFrame, era: Era) -> np.ndarray` (sum above the era's median draw sum `era.picks * (era.pool_size + 1) / 2`).
  - `runs_z(bits: np.ndarray) -> tuple[int, float]` (observed runs, z vs independence).
  - `runs_test(bits: np.ndarray, rng, replicates: int = DEFAULT_REPLICATES) -> TestResult` with p-value from permutation of the bits.

- [ ] **Step 1: Write the failing tests**

```python
def test_runs_z_alternating_and_blocked():
    alternating = np.arange(50) % 2
    n_runs, z = runs_z(alternating)
    assert n_runs == 50 and z > 5
    blocked = np.array([0] * 25 + [1] * 25)
    n_runs_b, z_b = runs_z(blocked)
    assert n_runs_b == 2 and z_b < -5

def test_planted_alternation_fires():
    r = runs_test(np.arange(400) % 2, np.random.default_rng(2), replicates=2000)
    assert r.p_value < 0.01

def test_fair_bits_do_not_fire():
    bits = np.random.default_rng(4).integers(0, 2, 400)
    r = runs_test(bits, np.random.default_rng(5), replicates=2000)
    assert r.p_value > 0.01
```

- [ ] **Step 2: Run tests to verify they fail** — expect FAIL.

- [ ] **Step 3: Implement `runs.py`**

`runs_z` uses the standard Wald-Wolfowitz mean/variance; p-value from permuted-bit run-count null (`|z|`-style two-sided via the run-count distribution, documented in a comment). Bits helpers document the encoding choice (per-draw features, time-ordered).

- [ ] **Step 4: Run tests, then ruff** — expect PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat: add runs tests on draw sum parity and level"`.

---

### Task 8: Serial independence (`engine/serial.py`)

**Files:**
- Create: `src/poc_lottery_randomness_auditor/engine/serial.py`
- Test: `tests/engine/test_serial.py`

**Interfaces:**
- Consumes: `fair_draws`, `null_distribution`, `mc_p_value`, `DEFAULT_REPLICATES`; `TestResult`; `Era`.
- Produces:
  - `FEATURES: tuple[str, ...] = ("sum", "range", "odd_count")`.
  - `draw_features(draws_df: pd.DataFrame) -> pd.DataFrame` (one column per feature, draw order).
  - `acf_max(features: pd.DataFrame, lags: int = 10) -> float` (max absolute autocorrelation across features and lags 1..`lags`).
  - `serial_test(draws_df, era, rng, replicates: int = DEFAULT_REPLICATES, lags: int = 10) -> TestResult` (Monte Carlo null via `fair_draws`).

- [ ] **Step 1: Write the failing tests**

```python
def test_draw_features_values(tiny_draws_df):
    feats = draw_features(tiny_draws_df)
    assert feats.loc[0, "sum"] == int(tiny_draws_df.loc[0, [f"n{i}" for i in range(1, 7)]].sum())

def test_acf_max_white_noise_small():
    rng = np.random.default_rng(12)
    feats = pd.DataFrame({"sum": rng.normal(size=500), "range": rng.normal(size=500), "odd_count": rng.normal(size=500)})
    assert acf_max(feats) < 0.2

def test_planted_drift_fires():
    t = np.arange(800)
    draws = drifting_draws(t)  # helper: fair draws plus slowly rising floor on ball values
    r = serial_test(draws_to_df(draws), LOTTO.eras[1], np.random.default_rng(13), replicates=1000)
    assert r.p_value < 0.01
```

- [ ] **Step 2: Run tests to verify they fail** — expect FAIL.

- [ ] **Step 3: Implement `serial.py`**

Autocorrelation via `np.correlate` on demeaned series, normalised; details carry the per-feature acf vectors for the chart.

- [ ] **Step 4: Run tests, then ruff** — expect PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat: add serial independence analysis"`.

---

### Task 9: Gap analysis (`engine/gaps.py`)

**Files:**
- Create: `src/poc_lottery_randomness_auditor/engine/gaps.py`
- Test: `tests/engine/test_gaps.py`

**Interfaces:**
- Consumes: Task 4 harness; `TestResult`; `Era`.
- Produces:
  - `gap_ks(waits: np.ndarray, p: float) -> float` (KS distance of observed waits vs `scipy.stats.geom(p)`).
  - `gaps_omnibus(draws_df: pd.DataFrame, era: Era) -> float` (max per-number `gap_ks`, `p = era.picks / era.pool_size`).
  - `gaps_test(draws_df, era, rng, replicates: int = DEFAULT_REPLICATES) -> TestResult`.
  - `overdue_table(draws_df: pd.DataFrame, era: Era, top: int = 10) -> pd.DataFrame` (number, draws since last seen, expected wait; exploratory, labelled as such in `details`).

- [ ] **Step 1: Write the failing tests**

```python
def test_gap_ks_matches_scipy():
    waits = np.array([3, 1, 8, 2, 11, 4, 5])
    assert gap_ks(waits, 6 / 59) == pytest.approx(
        scipy.stats.kstest(waits, scipy.stats.geom(6 / 59).cdf).statistic)

def test_suppressed_number_fires():
    w = np.full(59, 1.0); w[13] = 0.0
    draws = weighted_draws(LOTTO.eras[1], w, 4000, np.random.default_rng(21))
    r = gaps_test(draws_to_df(draws), LOTTO.eras[1], np.random.default_rng(22), replicates=1000)
    assert r.p_value < 0.01

def test_overdue_table_correct(tiny_draws_df):
    table = overdue_table(tiny_draws_df, LOTTO.eras[1], top=5)
    assert table["draws_since_seen"].is_monotonic_decreasing
```

- [ ] **Step 2: Run tests to verify they fail** — expect FAIL.

- [ ] **Step 3: Implement `gaps.py`**

Waits per number = differences between consecutive appearance indices; omnibus to Monte Carlo null; details carry per-number KS values and the overdue table.

- [ ] **Step 4: Run tests, then ruff** — expect PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat: add gap analysis and overdue table"`.

---

### Task 10: Order statistics (`engine/order_stats.py`)

**Files:**
- Create: `src/poc_lottery_randomness_auditor/engine/order_stats.py`
- Test: `tests/engine/test_order_stats.py`

**Interfaces:**
- Consumes: Task 4 harness; `TestResult`; `Era`.
- Produces:
  - `min_pmf(era: Era) -> np.ndarray` where entry `m` (1-based) is `comb(pool - m, picks - 1) / comb(pool, picks)`... derived as `P(min = m) = comb(pool - m, picks - 1) / comb(pool, picks)`; verified against the identity `P(min >= m) = comb(pool - m + 1, picks) / comb(pool, picks)` in tests.
  - `order_stats_ks(draws_df: pd.DataFrame, era: Era) -> float` (max KS over {min, max, spread} against exact or Monte Carlo reference CDFs).
  - `order_stats_test(draws_df, era, rng, replicates: int = DEFAULT_REPLICATES) -> TestResult`.

- [ ] **Step 1: Write the failing tests**

```python
def test_min_pmf_known_answer():
    pmf = min_pmf(LOTTO.eras[0])
    assert pmf[1] == pytest.approx(6 / 49)          # P(min = 1)
    assert pmf.sum() == pytest.approx(1.0)

def test_low_bias_fires():
    w = np.linspace(2.0, 0.5, 59)  # low numbers favoured
    draws = weighted_draws(LOTTO.eras[1], w, 4000, np.random.default_rng(31))
    r = order_stats_test(draws_to_df(draws), LOTTO.eras[1], np.random.default_rng(32), replicates=1000)
    assert r.p_value < 0.01
```

- [ ] **Step 2: Run tests to verify they fail** — expect FAIL.

- [ ] **Step 3: Implement `order_stats.py`**

Exact min/max CDFs from the combinatorial identity; spread reference via the Monte Carlo harness; details carry empirical vs reference CDF points for the chart.

- [ ] **Step 4: Run tests, then ruff** — expect PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat: add order statistics analysis"`.

---

### Task 11: Spectral analysis (`engine/spectral.py`)

**Files:**
- Create: `src/poc_lottery_randomness_auditor/engine/spectral.py`
- Test: `tests/engine/test_spectral.py`

**Interfaces:**
- Consumes: Task 4 harness; `TestResult`; `Era`.
- Produces:
  - `periodogram(bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]` (frequencies, power; `np.fft.rfft` on the demeaned indicator series).
  - `spectral_omnibus(draws_df: pd.DataFrame, era: Era) -> float` (max standardised peak across all numbers and frequencies).
  - `spectral_test(draws_df, era, rng, replicates: int = DEFAULT_REPLICATES) -> TestResult`.

- [ ] **Step 1: Write the failing tests**

```python
def test_periodogram_finds_known_period():
    bits = (np.arange(500) % 5 == 0).astype(float)
    freqs, power = periodogram(bits)
    assert freqs[np.argmax(power)] == pytest.approx(1 / 5, abs=1 / 500)

def test_cyclic_rig_fires():
    draws = cyclic_draws(LOTTO.eras[1], favoured=7, period=10, n_draws=2000, rng=np.random.default_rng(41))
    r = spectral_test(draws_to_df(draws), LOTTO.eras[1], np.random.default_rng(42), replicates=500)
    assert r.p_value < 0.01
```

- [ ] **Step 2: Run tests to verify they fail** — expect FAIL.

- [ ] **Step 3: Implement `spectral.py`**

Omnibus = max peak-to-median ratio over numbers; Monte Carlo null via `fair_draws`; details carry the per-number peak table for the chart.

- [ ] **Step 4: Run tests, then ruff** — expect PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat: add spectral analysis"`.

---

### Task 12: Bayesian layer (`engine/bayesian.py`)

**Files:**
- Create: `src/poc_lottery_randomness_auditor/engine/bayesian.py`
- Test: `tests/engine/test_bayesian.py`

**Interfaces:**
- Consumes: `TestResult`; `Era`.
- Produces:
  - `ROPE_DELTA: float = 0.05`.
  - `posterior_intervals(counts: np.ndarray, credibility: float = 0.95) -> np.ndarray` shape `(len(counts), 2)`: Beta marginal quantiles under a symmetric Dirichlet(1) posterior (`scipy.stats.beta(1 + c_i, 1 + total - c_i).ppf`).
  - `certified_delta(counts: np.ndarray, credibility: float = 0.95) -> float`: smallest delta such that every interval lies inside `uniform * (1 ± delta)`.
  - `bayesian_uniformity(draws_df: pd.DataFrame, era: Era) -> TestResult` with `method="bayesian"`, `p_value=None`, `statistic=certified_delta(...)`, details carrying intervals and `ROPE_DELTA`.

- [ ] **Step 1: Write the failing tests**

```python
def test_posterior_interval_matches_scipy():
    counts = np.array([5, 5, 5, 5])
    lo, hi = posterior_intervals(counts, 0.95)[0]
    ref = scipy.stats.beta(1 + 5, 1 + 20 - 5)
    assert lo == pytest.approx(ref.ppf(0.025)) and hi == pytest.approx(ref.ppf(0.975))

def test_certified_delta_small_for_huge_uniform_sample():
    counts = np.full(59, 100_000)
    assert certified_delta(counts) < 0.05

def test_certified_delta_large_for_skewed_sample():
    counts = np.full(59, 100); counts[0] = 400
    assert certified_delta(counts) > 0.5
```

- [ ] **Step 2: Run tests to verify they fail** — expect FAIL.

- [ ] **Step 3: Implement `bayesian.py`**

Closed-form conjugate updates only; a module docstring states the model (symmetric Dirichlet(1) prior, multinomial likelihood) and the reading of `certified_delta` ("every ball is within delta% of its fair share, at 95% credibility").

- [ ] **Step 4: Run tests, then ruff** — expect PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat: add Bayesian uniformity layer"`.

---

### Task 13: Battery orchestration and multiple-comparison control (`engine/battery.py`)

**Files:**
- Create: `src/poc_lottery_randomness_auditor/engine/battery.py`
- Test: `tests/engine/test_battery.py`

**Interfaces:**
- Consumes: every `*_test` function from Tasks 5-12; `validate_draws`-shaped frames (Task 2); `MASTER_SEED`, `DEFAULT_REPLICATES`.
- Produces:
  - `benjamini_hochberg(p_values: Sequence[float], q: float = 0.05) -> list[bool]`.
  - `benjamini_yekutieli(p_values: Sequence[float], q: float = 0.05) -> list[bool]`.
  - `@dataclass(frozen=True) class BatteryResult`: `results: tuple[TestResult, ...], rejections_bh: tuple[str, ...], rejections_by: tuple[str, ...]` where rejection keys are `"{era}:{family}:{subject}"`.
  - `run_battery(draws: pd.DataFrame, spec: GameSpec, seed: int = MASTER_SEED, replicates: int = DEFAULT_REPLICATES) -> BatteryResult`.

- [ ] **Step 1: Write the failing tests**

```python
def test_bh_known_answer():
    assert benjamini_hochberg([0.001, 0.02, 0.5]) == [True, True, False]

def test_by_known_answer():
    assert benjamini_yekutieli([0.001, 0.02, 0.5]) == [True, False, False]

def test_battery_flags_planted_pair(era2_sticky_df):
    result = run_battery(era2_sticky_df, LOTTO, seed=1, replicates=500)
    assert any("pairs" in key for key in result.rejections_bh)

def test_battery_deterministic(era2_sticky_df):
    r1 = run_battery(era2_sticky_df, LOTTO, seed=1, replicates=500)
    r2 = run_battery(era2_sticky_df, LOTTO, seed=1, replicates=500)
    assert [r.p_value for r in r1.results] == [r.p_value for r in r2.results]
```

- [ ] **Step 2: Run tests to verify they fail** — expect FAIL.

- [ ] **Step 3: Implement `battery.py`**

An explicit, readable sequence: per era, split draws; per subject (`main`, `bonus`; era 3 also splits by `round` and adds `homogeneity_test`); append each family's result; Bayesian appended with `p_value=None` and excluded from FDR. BH and BY applied per era across non-None p-values. All RNG from one `np.random.default_rng(seed)` spawned per family via `rng.spawn` or sequential `default_rng(seed + index)`; document the scheme in a comment.

- [ ] **Step 4: Run tests, then ruff** — expect PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat: add battery orchestration with BH and BY control"`.

---

### Task 14: Power analysis (`engine/power.py`)

**Files:**
- Create: `src/poc_lottery_randomness_auditor/engine/power.py`
- Test: `tests/engine/test_power.py`

**Interfaces:**
- Consumes: `weighted_draws`, `sticky_pair_draws`, `cyclic_draws` (Task 4); family test functions (Tasks 5, 6, 11 as the showcased detectors).
- Produces:
  - `@dataclass(frozen=True) class PowerScenario`: `family: str, bias_kind: str, level: float, n_draws: int`.
  - `power_grid(scenarios: list[PowerScenario], era: Era, replicates: int, seed: int) -> pd.DataFrame` columns `("family", "bias_kind", "level", "n_draws", "power")`.
  - `DEFAULT_SCENARIOS: list[PowerScenario]` (frequency vs weighted, pairs vs sticky, spectral vs cyclic; levels and sizes chosen so a full run stays within the five-minute budget).

- [ ] **Step 1: Write the failing tests**

```python
def test_power_high_for_strong_bias():
    scenarios = [PowerScenario("frequency", "weighted", level=4.0, n_draws=3000)]
    df = power_grid(scenarios, LOTTO.eras[1], replicates=100, seed=7)
    assert df.loc[0, "power"] > 0.9

def test_power_near_alpha_for_null_bias():
    scenarios = [PowerScenario("frequency", "weighted", level=1.0, n_draws=3000)]
    df = power_grid(scenarios, LOTTO.eras[1], replicates=100, seed=7)
    assert df.loc[0, "power"] < 0.2
```

- [ ] **Step 2: Run tests to verify they fail** — expect FAIL.

- [ ] **Step 3: Implement `power.py`**

For each scenario: generate `replicates` rigged datasets, run the family's statistic, compare against the cached fair null (replicates shared across levels for speed, documented); `power` = fraction of p-values below 0.01.

- [ ] **Step 4: Run tests, then ruff** — expect PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat: add power analysis grid"`.

---

### Task 15: Visual theme (`theme.py`)

**Files:**
- Create: `src/poc_lottery_randomness_auditor/theme.py`
- Test: `tests/test_theme.py`

**Interfaces:**
- Consumes: altair.
- Produces:
  - `THEME: dict` (Altair theme config: font, palette of >= 6 colourblind-safe categorical colours, grid, background, title/label sizing).
  - `register_theme() -> None` registering and enabling theme `"lottery_audit"`.

- [ ] **Step 1: Write the failing test**

```python
def test_theme_registers_and_enables():
    register_theme()
    assert alt.themes.active == "lottery_audit"
    assert "config" in THEME
    assert len(THEME["config"]["range"]["category"]) >= 6
```

- [ ] **Step 2: Run test to verify it fails** — expect FAIL.

- [ ] **Step 3: Implement `theme.py`**

One `THEME` dict; palette from a colourblind-safe family (for example Okabe-Ito); typography and grid chosen here once, applied everywhere.

- [ ] **Step 4: Run tests, then ruff** — expect PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat: add shared Altair theme"`.

---

### Task 16: Chart library (`charts.py`)

**Files:**
- Create: `src/poc_lottery_randomness_auditor/charts.py`
- Test: `tests/test_charts.py`

**Interfaces:**
- Consumes: `TestResult` (Task 4), power `DataFrame` (Task 14), `register_theme` (Task 15).
- Produces one function per figure, each `-> alt.Chart`:
  - `frequency_chart(r: TestResult)`, `pairs_chart(r)`, `runs_chart(r)`, `acf_chart(r)`, `gaps_chart(r)`, `order_stats_chart(r)`, `spectral_chart(r)`, `posterior_chart(r)`, `power_chart(df: pd.DataFrame)`.
  - `CHARTS_BY_FAMILY: dict[str, Callable[[TestResult], alt.Chart]]` explicit mapping for report/notebook use.

- [ ] **Step 1: Write the failing tests**

```python
def test_frequency_chart_embeds_engine_values():
    r = make_frequency_result()  # fixture: counts [10, 20, 30], expected 20
    spec = frequency_chart(r).to_dict()
    values = spec["layer"][0]["data"]["values"]
    assert [v["count"] for v in values] == [10, 20, 30]

def test_all_charts_render_valid_specs():
    for name, fn in CHARTS_BY_FAMILY.items():
        spec = fn(make_result_for(name)).to_dict()  # fixture per family
        assert "mark" in spec or "layer" in spec

def test_power_chart_valid(power_df):
    assert power_chart(power_df).to_dict()["mark"] in {"rect", "bar", "line"}
```

- [ ] **Step 2: Run tests to verify they fail** — expect FAIL.

- [ ] **Step 3: Implement `charts.py`**

Each function reads only its `TestResult.details` payload (no recomputation). `register_theme()` called at module import. Charts: counts-vs-expected bars with reference line; pair heatmap; run-count trace; acf stems with MC bands; per-number KS bars plus overdue table data; empirical vs reference CDFs; per-number peak scatter with MC threshold; forest plot of credible intervals against the ROPE band; power heatmap.

- [ ] **Step 4: Run tests, then ruff** — expect PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat: add chart library on the shared theme"`.

---

### Task 17: Report pipeline (`report.py` + dependencies)

**Files:**
- Create: `src/poc_lottery_randomness_auditor/report.py`, `src/poc_lottery_randomness_auditor/templates/report.html.j2`
- Modify: `pyproject.toml` (via `uv add jinja2 weasyprint vl-convert-python`)
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `BatteryResult` (Task 13), `CHARTS_BY_FAMILY` (Task 16), power `DataFrame` (Task 14).
- Produces:
  - `@dataclass(frozen=True) class ReportMeta`: `title: str, generated_at: datetime, snapshot_sha256: str, n_draws: int, n_eras: int`.
  - `chart_svg(chart: alt.Chart) -> str` via `vl_convert.vegalite_to_svg`.
  - `render_html(battery: BatteryResult, power: pd.DataFrame, meta: ReportMeta) -> str`.
  - `write_pdf(html: str, out_path: Path) -> None` via WeasyPrint.

- [ ] **Step 1: Add dependencies**

Run: `uv add jinja2 weasyprint vl-convert-python`
Expected: all three in `pyproject.toml` and installed.

- [ ] **Step 2: Write the failing tests**

```python
def test_chart_svg_outputs_svg():
    svg = chart_svg(frequency_chart(make_frequency_result()))
    assert svg.startswith("<svg") or "<svg" in svg[:200]

def test_render_html_contains_verdicts_and_meta(battery_result, power_df):
    html = render_html(battery_result, power_df, make_meta())
    assert make_meta().snapshot_sha256 in html
    assert "era-59-two-round" in html
    assert html.count("<svg") >= 1

def test_render_html_deterministic(battery_result, power_df):
    assert render_html(battery_result, power_df, make_meta()) == render_html(battery_result, power_df, make_meta())

def test_write_pdf_produces_pdf(tmp_path):
    write_pdf("<html><body><h1>Audit</h1></body></html>", tmp_path / "r.pdf")
    assert (tmp_path / "r.pdf").read_bytes().startswith(b"%PDF-")
```

- [ ] **Step 3: Run tests to verify they fail** — expect FAIL (`ModuleNotFoundError` until Step 1 is done).

- [ ] **Step 4: Implement `report.py` and the template**

Jinja2 `Environment` with `autoescape=True`, template loaded from the package `templates/` directory. Report sections in plain English: headline verdict per era, one section per family with its SVG and a two-sentence intuition, power-analysis section, limitations, provenance block (snapshot hash, generation time, seed). No statistical jargon: p-values rendered as "a fair lottery would produce a result this extreme about 1 time in N".

- [ ] **Step 5: Run tests, then ruff** — expect PASS.

- [ ] **Step 6: Commit** — `git commit -m "feat: add HTML and PDF report pipeline"`.

---

### Task 18: One-command orchestration (`audit.py`, CLI)

**Files:**
- Create: `src/poc_lottery_randomness_auditor/audit.py`
- Modify: `src/poc_lottery_randomness_auditor/__init__.py` (add `main()`)
- Test: `tests/test_audit.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `run(snapshot_dir: Path, out_dir: Path, seed: int = MASTER_SEED, now: datetime | None = None) -> Path` (returns HTML path; writes `report.html` and `report.pdf` into `out_dir`).
  - `main() -> None` in `__init__.py`: argparse with `--snapshot`, `--out`, `--seed`, `--now` (ISO format; the hook the reproducibility check uses).

- [ ] **Step 1: Write the failing test**

```python
def test_end_to_end_deterministic(tmp_path, synthetic_snapshot):
    a = run(synthetic_snapshot, tmp_path / "a", seed=1, now=datetime(2026, 10, 21, 12, 0))
    b = run(synthetic_snapshot, tmp_path / "b", seed=1, now=datetime(2026, 10, 21, 12, 0))
    assert a.read_bytes() == b.read_bytes()
    assert (tmp_path / "a" / "report.pdf").read_bytes().startswith(b"%PDF-")
```

`synthetic_snapshot` fixture: a small valid snapshot generated by `fair_draws` (committed as CSV + SHA256SUMS under `tests/fixtures/snapshot_synth/`).

- [ ] **Step 2: Run test to verify it fails** — expect FAIL.

- [ ] **Step 3: Implement `audit.py` and `main()`**

Straight-line code: `load_draws` -> `run_battery` -> `power_grid(DEFAULT_SCENARIOS)` -> charts -> `render_html` -> `write_pdf`. `now=None` uses `datetime.now(timezone.utc)`.

- [ ] **Step 4: Run test, then the real thing**

Run: `uv run pytest tests/test_audit.py -v` then `uv run poc-lottery-randomness-auditor --snapshot data/snapshots/lotto --out reports`
Expected: PASS; `reports/report.html` and `reports/report.pdf` generated from the real snapshot.

- [ ] **Step 5: Commit** — `git commit -m "feat: add one-command audit orchestration and CLI"`.

---

### Task 19: Marimo notebook (`notebooks/lotto_audit.py`)

**Files:**
- Create: `notebooks/lotto_audit.py` (replaces the empty placeholder)
- Modify: none in `src/`

**Interfaces:**
- Consumes: public package API only (`games.LOTTO`, `data.load_draws`, `engine.battery.run_battery`, `engine.power`, `charts`, `theme`).

- [ ] **Step 1: Write the notebook**

Chapters, one per method in spec section 6 order, each: intuition paragraph, its `charts.py` figure, the maths in a `mo.md` callout, verdict line. Opening chapter: the question, the data provenance, the three eras table. Era-3 chapter includes the round-vs-round (or machine-vs-machine, per what the snapshot contains) homogeneity check. Closing chapters: rigged-lottery gallery, power analysis, limitations, Bayesian "certified within delta%" summary. An `mo.ui.dropdown` era selector drives per-era chapters through marimo reactivity. Zero statistical computation in the notebook: every number comes from `run_battery` or `engine.power`.

- [ ] **Step 2: Smoke-run the notebook headless**

Run: `uv run marimo export html notebooks/lotto_audit.py -o /private/var/folders/z1/sp4_dgd16_92vgm8wsp3vpsc0000gn/T/opencode/lotto_audit_smoke.html`
Expected: exit 0, non-trivial HTML written, no cell errors.

- [ ] **Step 3: Review the exported page for the three anti-goals**

Read the export: annotations complete enough to stand alone (open it as a stranger), charts on-theme, numbers match `reports/report.html` for the same era.
Expected: no mismatches; fixes folded into this task.

- [ ] **Step 4: Commit** — `git commit -m "feat: add annotated Lotto audit notebook"`.

---

### Task 20: Reproducibility audit and documentation

**Files:**
- Create: `scripts/verify_repro.sh`
- Modify: `README.md`, `AGENTS.md`, `docs/data-sources.md`

**Interfaces:**
- Consumes: the CLI from Task 18.

- [ ] **Step 1: Write `scripts/verify_repro.sh`**

Runs the CLI twice with the same `--seed` and `--now` into two temp directories and `diff`s the HTML (byte-identical expected) and reports PDF equality on the same machine. Exits non-zero on difference.

- [ ] **Step 2: Run it**

Run: `bash scripts/verify_repro.sh`
Expected: exit 0. If the PDF differs byte-wise on the same machine, find and fix the nondeterminism (fonts/metadata) before continuing.

- [ ] **Step 3: Fresh-environment check**

Run: `uv sync --frozen && uv run pytest && bash scripts/verify_repro.sh`
Expected: all pass from a clean sync.

- [ ] **Step 4: Update documentation**

README: project purpose, quickstart (`uv sync`, fetch data, run audit, open notebook), updated structure section. `AGENTS.md`: add a short project section (what this POC is, key commands, where the design and plan live). `docs/data-sources.md`: final snapshot hash and draw counts.

- [ ] **Step 5: Final gate**

Run: `uv run pytest && uv run ruff check . && uv run ruff format --check .`
Expected: all green.

- [ ] **Step 6: Commit** — `git commit -m "docs: add reproducibility check and project documentation"`.

---

## Self-review notes

- Spec coverage: sections 5-10 of the spec map to Tasks 1-3 (data), 4-12 (methodology families), 13 (FDR/BY), 14 (power), 15-17 (presentation), 18 (orchestration), 19 (notebook), 20 (reproducibility). Era-3 homogeneity rides Tasks 5 and 13.
- Statistical tests in pytest use fixed seeds and strong planted signals to avoid flakiness; marginal cases belong to the Monte Carlo chapter, not CI.
- `draws_to_df` and `drifting_draws` test helpers are defined once in `tests/engine/conftest.py` during Task 5 and reused.
