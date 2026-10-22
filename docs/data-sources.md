# Lotto data sources

Recon date: 2026-10-21.

## Official source

- Page: `https://www.national-lottery.co.uk/results/lotto/draw-history`
- Download URL: `https://api-dfe.national-lottery.co.uk/draw-game/results/6/download?interval=ONE_EIGHTY`
- Format: CSV with header `DrawDate,Round,Ball 1,Ball 2,Ball 3,Ball 4,Ball 5,Ball 6,Bonus Ball,Guaranteed Millionaire Maker,Ball Set,Machine,DrawNumber`.
- Earliest draw covered by the download on 2026-10-21: 2026-04-25 (draw 3166). The public page and download are explicitly the previous 180 days only, so this is not the full 1994 to 2026 history.
- Machine and ball-set IDs: present in the 180 day download (`Machine` values such as `Lotto4` and `Lotto 4`; `Ball Set` values such as `L3`). Era 3 rows use `Round` 1 and 2; earlier rows in the same download use `Round` 1.

## Cross-validation and full-history source

Chosen source: Kaggle dataset `deltaromeo/uk-national-lottery-results-1994-2023`, file `lohi.csv` from `https://www.kaggle.com/api/v1/datasets/download/deltaromeo/uk-national-lottery-results-1994-2023`. The same archive also contains `drawn.csv` in draw order; `lohi.csv` is used because its main six are already low to high.

Why: it is a direct CSV download, needs no credentials, and includes machine and ball-set columns. It covers draw 1 (1994-11-19) to draw 3168 (2026-05-02). It is stale for draws after 2026-05-02, including the start of era 3.

Decision: use Kaggle `lohi.csv` as the primary source for draws up to 2026-05-02, use the official 180 day CSV as the primary source for draws after 2026-05-02, and reconcile the overlap (2026-04-25 to 2026-05-02). Winning numbers, bonus, date and round must match exactly. Machine and ball-set are compared where the naming is comparable; legacy machine names such as `Lancelot` are not always identical to the current official `LottoN` labels, so any such mismatch is listed here rather than silently edited. This follows the design fallback because the official download does not cover full history, while still covering era 3.

Reconciliation result on 2026-10-21: overlap draws 3166, 3167 and 3168 matched exactly for date, round, the six main numbers and bonus after normalising machine whitespace and writing numeric ball sets as `L<n>`. One ID mismatch remains documented: draw 3168 has Kaggle machine `Lancelot` and official machine `Lotto6`. The snapshot keeps the primary Kaggle row for draws up to 2026-05-02; machine IDs before era 3 are not used for the era 3 machine homogeneity checks, and the official recent rows carry the current `LottoN` labels from 2026-05-06 onward.

Snapshot written by `scripts/fetch_data.py` on 2026-10-21: `data/snapshots/lotto/draws.csv`, 3256 rows, draws 1 to 3217, dates 1994-11-19 to 2026-10-21, sha256 `f75080537ca6bfe750b5db254fb60b577ff61798ce2a6b55a986d1dc4fce6f91`. Era row counts: era-49 2065, era-59 1113, era-59-two-round 78.

Independent spot check: lottery.co.uk archive pages (`https://www.lottery.co.uk/lotto/results/archive-YYYY`) agree with the winning numbers shown for sampled 1994, 2026 era 2 and 2026 era 3 dates, but they are HTML year pages and do not expose machine or ball-set IDs, so they are not the primary source.
