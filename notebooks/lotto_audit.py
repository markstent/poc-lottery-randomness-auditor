import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo

    from poc_lottery_randomness_auditor import charts, theme
    from poc_lottery_randomness_auditor.data import load_draws
    from poc_lottery_randomness_auditor.engine import power as power_engine
    from poc_lottery_randomness_auditor.engine.battery import run_battery
    from poc_lottery_randomness_auditor.games import LOTTO

    theme.register_theme()
    seed = 20261021
    battery_replicates = 500
    power_replicates = 200
    draws = load_draws(Path("data/snapshots/lotto"), LOTTO)
    battery = run_battery(draws, LOTTO, seed=seed, replicates=battery_replicates)
    power_by_era = {
        era.name: power_engine.power_grid(
            power_engine.DEFAULT_SCENARIOS,
            era,
            power_replicates,
            seed,
        )
        for era in LOTTO.eras
        if (draws["era"] == era.name).any()
    }
    mo.md("Snapshot, battery and power grid loaded.")


@app.cell
def _(LOTTO, draws, mo):
    era_rows = [
        {
            "era": era.name,
            "start": era.start.isoformat(),
            "end": era.end.isoformat() if era.end else "ongoing",
            "pool": era.pool_size,
            "picks": era.picks,
            "bonus": era.has_bonus,
            "rounds": era.rounds_per_draw,
        }
        for era in LOTTO.eras
    ]
    mo.vstack(
        [
            mo.md("# UK Lotto randomness audit"),
            mo.md(
                "Can we tell, from the published winning numbers alone, whether "
                "Lotto draws behave like a fair random process? This notebook "
                "checks the pinned snapshot in `data/snapshots/lotto` and keeps "
                "every statistical number inside the audited engine."
            ),
            mo.md(
                f"The snapshot contains {len(draws)} draw rounds. The game has "
                "three rule eras; the two-round era is checked for round-to-round "
                "consistency as well as overall fairness."
            ),
            mo.ui.table(era_rows, pagination=False, selection=None),
        ]
    )


@app.cell
def _(LOTTO, mo):
    era_selector = mo.ui.dropdown(
        options=[era.name for era in LOTTO.eras],
        value="era-59",
        label="Era",
    )
    era_selector


@app.cell
def _(battery, era_selector, mo):
    era_name = era_selector.value
    by_key = {
        (result.family, result.subject): result
        for result in battery.results
        if result.era == era_name
    }
    mo.md(f"## Era: {era_name}")


@app.cell
def _(battery, mo):
    rejections = set(battery.rejections_bh)

    def verdict(result):
        if result is None:
            return "Verdict: not run for this era."
        key = f"{result.era}:{result.family}:{result.subject}"
        if key in rejections:
            return "Verdict: flagged after false-discovery-rate control."
        if result.p_value is None:
            return "Verdict: Bayesian summary; see the certified-delta chapter."
        times = max(1, round(1.0 / result.p_value))
        return (
            "Verdict: not flagged. A fair lottery would produce something this "
            f"extreme about 1 time in {times}."
        )

    mo.md("")


@app.cell
def _(by_key, charts, mo, verdict):
    def chapter(title, family, subject, intuition, maths):
        result = by_key.get((family, subject))
        blocks = [mo.md(f"## {title}"), mo.md(intuition)]
        if result is not None and family in charts.CHARTS_BY_FAMILY:
            blocks.append(charts.CHARTS_BY_FAMILY[family](result))
        blocks.append(mo.callout(mo.md(maths), kind="info", title="Maths"))
        blocks.append(mo.md(verdict(result)))
        return mo.vstack(blocks)

    bonus_verdict = verdict(by_key.get(("frequency", "bonus")))
    runs_high_verdict = verdict(by_key.get(("runs", "sum-high")))
    homogeneity_main = verdict(by_key.get(("homogeneity", "main-round1-vs-round2")))
    homogeneity_bonus = verdict(by_key.get(("homogeneity", "bonus-round1-vs-round2")))

    mo.vstack(
        [
            chapter(
                "Frequency",
                "frequency",
                "main",
                "If the machine is fair, each ball should appear about equally often. This chart compares the observed counts with the fair line.",
                "For counts $c_i$ over $N$ balls, the statistic is $\\sum_i (c_i - N/6)^2 / (N/6)$, compared with a chi-square reference.",
            ),
            mo.md(f"Bonus-ball check: {bonus_verdict}"),
            chapter(
                "Round-vs-round homogeneity",
                "homogeneity",
                "main-round1-vs-round2",
                "In the two-round era, the two rounds drawn on the same night should come from the same distribution.",
                "Two count vectors are compared in a 2 by 59 contingency table; categories never seen in either round are dropped before the chi-square test.",
            ),
            mo.md(f"Main balls: {homogeneity_main}"),
            mo.md(f"Bonus balls: {homogeneity_bonus}"),
            chapter(
                "Pairs",
                "pairs",
                "main",
                "Some balls could be sticky as a pair even if every ball alone looks fair. This heatmap looks for pairs that travel together too often.",
                "All 1,711 unordered pairs are compared with the fair pair count and summarised by one max-deviation statistic against a Monte Carlo null.",
            ),
            chapter(
                "Runs",
                "runs",
                "sum-parity",
                "A fair sequence should not alternate too regularly or sit in long streaks. This chapter checks the sequence of draw-sum parities.",
                "The number of runs is converted to a z-score and compared with permutations of the same bit sequence.",
            ),
            mo.md(f"High-sum check: {runs_high_verdict}"),
            chapter(
                "Serial dependence",
                "serial",
                "main-features",
                "One draw should not remember the previous draw. This chapter checks autocorrelation in simple draw features at several lags.",
                "For each feature, the autocorrelation at lags 1 to 10 is compared with the spread seen in fair simulated histories.",
            ),
            chapter(
                "Gaps",
                "gaps",
                "main",
                "For each ball, the waiting time between appearances should look geometric. Long droughts and suspiciously regular spacing both distort this.",
                "Each ball's empirical gap distribution is compared with the geometric reference by a Kolmogorov-Smirnov distance.",
            ),
            chapter(
                "Order statistics",
                "order-stats",
                "main",
                "The smallest ball, largest ball and spread of a draw have known fair distributions even though the six balls are sorted.",
                "Empirical CDFs for the minimum, maximum and range are compared with the exact order-statistic reference CDFs.",
            ),
            chapter(
                "Spectral cycles",
                "spectral",
                "main",
                "A hidden cycle would make some balls appear with a fixed period. This chapter scans the appearance indicator of every ball for dominant frequencies.",
                "A periodogram is computed for each ball and the largest peak-to-median ratio is compared with fair simulated draws.",
            ),
            chapter(
                "Bayesian uniformity",
                "bayesian",
                "main",
                "Instead of asking only whether the data are surprising, this estimates each ball's probability directly with a symmetric Dirichlet prior.",
                "A Dirichlet(1) prior gives Beta marginal posteriors; the certified delta is the smallest band containing every 95% interval around the fair share.",
            ),
        ]
    )


@app.cell
def _(by_key, charts, era_name, mo, power_by_era, verdict):
    bayesian = by_key.get(("bayesian", "main"))
    certified = (
        f"At 95% credibility, every ball is within {bayesian.statistic:.1%} of its fair share."
        if bayesian is not None
        else "No Bayesian result was run for this era."
    )
    power_frame = power_by_era.get(era_name)
    power_chart = (
        charts.power_chart(power_frame)
        if power_frame is not None
        else mo.md("No power grid for this era.")
    )
    mo.vstack(
        [
            mo.md("## Rigged-lottery gallery"),
            mo.md(
                "The power analysis plants three cheat styles: one overweighted ball, one sticky pair, and one cyclic ball. The gallery is the set of power cells below; it shows which cheats this battery can actually catch."
            ),
            mo.md("## Power analysis"),
            power_chart,
            mo.md(
                "A quiet fairness report is only meaningful where these planted cheats are detected with high power."
            ),
            mo.md("## Bayesian certified summary"),
            mo.md(certified),
            mo.md(verdict(bayesian)),
            mo.md("## Limitations"),
            mo.md(
                "This audit tests the recorded numbers, not the physical machine, ball sets or draw procedure. A pass means these checks found no convincing deviation in this snapshot; it is not a guarantee about future draws."
            ),
        ]
    )


if __name__ == "__main__":
    app.run()
