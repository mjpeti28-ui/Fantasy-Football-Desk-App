# PFF Premium Stats vs v2 rest-of-season forecasts (2021-2025)

Run on 2026-10-16. Code: `tools/fetch_pff_stats.py` (fetch, cached under gitignored `.cache/pff/`) and
`backtest/pff_stats.py` (backtest). Aggregate numbers only; no PFF rows are committed.

## Setup
- Rows: `usage_rows_{2021..2025}` checkpoints W=3..13, est_ppg > 0, future_played >= 3.
- PFF: league-wide `/v1/facet/{receiving,rushing,passing}/summary` with `week=1,...,W-1` (season to date)
  and `week=W-3,...,W-1` (last 3). With a week list the API returns one aggregated row per player.
- Leakage check: mean WR/TE season-to-date routes go 41 (W=3), 107.6 (W=8), 162.6 (W=13); the last-3 window
  holds at about 50. The filter restricts the data as intended.
- Model: LOSO OLS per position, weighted by future_played (as in `ecr_signal.py`). Missing PFF values
  (player didn't play before W) are filled with the position median.
- Bases: `est_ppg` (v2) and `live` (est_ppg blended with rank-matched FantasyPros ECR at 1/3).
- Metrics: rank corr = mean Spearman within season-week-position; MAE = future_played-weighted, in ppg.

## ID match (normalized name + position, ties broken by team; nflverse gsis to espn)
Rows with est_ppg > 5 and >= 3 future games: QB 94.3%, RB 97.9%, TE 98.2%, WR 98.6%.
The unmatched QBs are backups with no snaps in weeks 1-12 (Brissett, Flacco, Rudolph, Mullens 2023, etc.), so PFF has
nothing to match them to. Over all est_ppg > 0 rows the rate is lower (33-42% for est <= 5) because most of those
players never played.

## Baselines (rank corr / MAE)
| pos | rows | est_ppg | live |
|---|---|---|---|
| QB | all | 0.705 / 3.81 | 0.717 / 3.80 |
| RB | all | 0.823 / 2.49 | 0.830 / 2.43 |
| WR | all | 0.803 / 2.38 | 0.812 / 2.33 |
| TE | all | 0.807 / 1.66 | 0.816 / 1.63 |
| QB | est > 5 | 0.586 / 3.72 | 0.601 / 3.71 |
| RB | est > 5 | 0.688 / 3.14 | 0.707 / 3.04 |
| WR | est > 5 | 0.728 / 2.83 | 0.742 / 2.76 |
| TE | est > 5 | 0.636 / 2.39 | 0.651 / 2.38 |

For scale, adding ECR (est_ppg to live) is worth +0.008 to +0.019 in rank corr.

## Deltas when each group is added to the base (est > 5 rows; d_rc > 0 and d_MAE < 0 are improvements)
Seasons up = number of held-out seasons (of 5) where rank corr improved.

| pos | group | d_rc vs est | seasons up | d_MAE vs est | d_rc vs live | seasons up | d_MAE vs live |
|---|---|---|---|---|---|---|---|
| WR | grade (offense) | +0.0004 | 2 | -0.007 | -0.0008 | 2 | -0.004 |
| WR | route grade | +0.0002 | 2 | -0.005 | -0.0012 | 2 | -0.003 |
| WR | YPRR | -0.0003 | 3 | -0.004 | -0.0012 | 2 | -0.003 |
| WR | targets/route | +0.0008 | 3 | -0.003 | -0.0008 | 3 | -0.001 |
| WR | route share (STD+L3) | +0.0029 | 5 | -0.004 | +0.0020 | 4 | -0.002 |
| WR | target share (STD+L3) | -0.0013 | 1 | +0.001 | -0.0011 | 2 | +0.003 |
| WR | all | -0.0014 | 2 | +0.001 | -0.0020 | 3 | +0.004 |
| TE | grade | +0.0064 | 3 | -0.018 | +0.0075 | 3 | -0.028 |
| TE | route grade | -0.0077 | 3 | +0.002 | -0.0066 | 3 | -0.008 |
| TE | YPRR | -0.0032 | 2 | +0.007 | -0.0001 | 4 | +0.003 |
| TE | targets/route | -0.0073 | 1 | +0.006 | -0.0047 | 2 | +0.005 |
| TE | route share | +0.0003 | 4 | -0.019 | -0.0038 | 4 | -0.012 |
| TE | target share | +0.0014 | 3 | +0.004 | +0.0002 | 4 | +0.006 |
| TE | all | +0.0088 | 3 | -0.046 | +0.0029 | 3 | -0.043 |
| RB | grade | -0.0029 | 2 | +0.016 | -0.0030 | 2 | +0.011 |
| RB | run grade | -0.0031 | 2 | +0.014 | -0.0020 | 3 | +0.011 |
| RB | elusive rating | +0.0020 | 4 | +0.006 | +0.0003 | 3 | +0.005 |
| RB | YAC/att | +0.0036 | 3 | -0.001 | +0.0009 | 2 | +0.007 |
| RB | rush share | +0.0021 | 3 | -0.009 | -0.0010 | 3 | +0.006 |
| RB | target share | -0.0026 | 3 | +0.016 | -0.0023 | 3 | +0.015 |
| RB | YPRR / route share | -0.0005 / -0.0026 | 2 / 1 | +0.004 / +0.011 | -0.0009 / -0.0035 | 2 / 1 | +0.002 / +0.019 |
| RB | all | -0.0067 | 1 | +0.050 | -0.0100 | 1 | +0.073 |
| QB | grade | +0.0001 | 2 | -0.006 | -0.0019 | 2 | +0.007 |
| QB | pass grade | -0.0009 | 2 | +0.012 | -0.0023 | 2 | +0.023 |
| QB | BTT / TWP rates | -0.0051 | 2 | +0.018 | +0.0005 | 3 | +0.014 |
| QB | ADOT | -0.0011 | 3 | +0.013 | -0.0066 | 1 | +0.019 |
| QB | rush att/game | -0.0114 | 2 | -0.085 | -0.0083 | 2 | -0.074 |
| QB | dropbacks/game | -0.0218 | 1 | +0.044 | -0.0338 | 1 | +0.049 |
| QB | all | -0.0333 | 2 | -0.032 | -0.0440 | 2 | +0.001 |

Over all est_ppg > 0 rows the picture is the same: every single-group mean delta is within +-0.01 rank corr and
+-0.06 ppg MAE. No group beats baseline in 4 of 5 seasons with a positive mean, except QB ADOT on est_ppg
(+0.004), so the "best together" combination never fired. Full tables: `pff_stats_results_{all,est_gt5}.csv`
(written by the script, not committed).

## Per-season consistency
Per-season rank-corr deltas mostly run +-0.01 (WR/RB) and +-0.05 (QB/TE). The few positive means come from one or two
seasons. For example, TE grade's +0.006 is +0.031, -0.050, +0.031, -0.016, +0.036 by season. The one group that is
positive every season, WR route share on est_ppg, gains +0.003 (live: +0.002, 4/5 seasons). That is about a fifth of
what ECR adds and well inside the season-to-season spread. QB rushing attempts cut MAE by 0.07-0.08 ppg in 3 of 5
seasons but hurt rank corr, so they shift QB levels a little without ordering QBs better.

## Verdict
Nothing here is worth adding to valuation.py. PFF grades and efficiency rates (YPRR, TPRR, elusive, YAC/att,
BTT/TWP, ADOT) add nothing beyond v2's usage-based estimate, with or without ECR. The volume shares (route, target,
rush) are mostly already in v2 through nflverse snaps, targets and opportunity. WR route share is the only
consistent sign, and at +0.002-0.003 rank corr it is too small to justify a new data dependency.
Caveats: rates are raw, not shrunk toward a position mean, and missing values are median-imputed. Shrinkage might
help the grades slightly, but deltas this close to zero make a real gain unlikely.
