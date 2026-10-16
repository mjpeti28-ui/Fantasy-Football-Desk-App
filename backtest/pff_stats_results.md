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

# Follow-up: low-history rows and a gradient-boosted check (`backtest/pff_stats_followup.py`)

Same rows, folds, metrics and leakage guard as above (features from weeks < W; future_played >= 3; est_ppg > 0).
d_rc = change in mean per-season within-week Spearman; "up" = held-out seasons (of 5) where it improved.
Reference scale for a real gain: ECR on top of est_ppg, +0.008 to +0.019.

## 1. Low-history subsets
Rows: n <= 3 games observed before W (QB 1499, RB 2742, WR 4033, TE 2257); W <= 5 (QB 919, RB 2102, WR 3115,
TE 1780); rookies, draft_year == season (QB 352, RB 1157, WR 1675, TE 817). Each cell is OLS fit either on the
subset only ("subset") or on all rows and scored on the subset ("all"). Base = live blend (est_ppg + 1/3 ECR);
est_ppg-base results are in history/backtest/multi/pff_followup_subsets.csv and tell the same story.

| subset | fit | pos | base rc | all PFF d_rc (up) | best group | best d_rc (up) | groups > 0 |
|---|---|---|---|---|---|---|---|
| n<=3 | all | QB | 0.537 | -0.055 (3) | rush_pg | +0.003 (3) | 1/6 |
| n<=3 | all | RB | 0.591 | -0.012 (2) | tgt_share | +0.000 (3) | 1/8 |
| n<=3 | all | WR | 0.561 | -0.003 (1) | route_share | **+0.019 (5)** | 1/7 |
| n<=3 | all | TE | 0.540 | -0.102 (1) | tgt_share | -0.006 (1) | 0/7 |
| n<=3 | subset | QB | 0.539 | -0.064 (2) | rush_pg | -0.005 (1) | 0/6 |
| n<=3 | subset | RB | 0.591 | -0.016 (3) | tgt_share | +0.000 (2) | 1/8 |
| n<=3 | subset | WR | 0.561 | -0.003 (2) | route_share | **+0.015 (4)** | 2/7 |
| n<=3 | subset | TE | 0.540 | -0.050 (1) | yprr_l3 | -0.005 (3) | 0/7 |
| W<=5 | all | QB | 0.732 | -0.014 (1) | grade | +0.006 (3) | 4/6 |
| W<=5 | all | RB | 0.809 | -0.003 (1) | grade | +0.001 (4) | 2/8 |
| W<=5 | all | WR | 0.812 | -0.003 (1) | route_share | -0.000 (3) | 0/7 |
| W<=5 | all | TE | 0.808 | -0.016 (0) | yprr | -0.001 (2) | 0/7 |
| W<=5 | subset | QB | 0.733 | -0.017 (1) | grade | +0.002 (3) | 2/6 |
| W<=5 | subset | RB | 0.809 | -0.004 (1) | grade | +0.001 (3) | 1/8 |
| W<=5 | subset | WR | 0.812 | -0.004 (1) | route_share | -0.001 (3) | 0/7 |
| W<=5 | subset | TE | 0.808 | -0.012 (2) | tgt_share | +0.001 (4) | 1/7 |
| rookie | all | QB | 0.596 | -0.037 (3) | grade | +0.035 (2) | 6/6 |
| rookie | all | RB | 0.751 | +0.000 (2) | yprr | +0.004 (2) | 3/8 |
| rookie | all | WR | 0.799 | -0.010 (1) | tgt_share | -0.000 (2) | 0/7 |
| rookie | all | TE | 0.692 | -0.012 (2) | tgt_share | +0.004 (4) | 3/7 |
| rookie | subset | QB | 0.596 | -0.207 (0) | pass_grade | +0.046 (2) | 5/6 |
| rookie | subset | RB | 0.751 | -0.020 (1) | yprr | +0.003 (3) | 1/8 |
| rookie | subset | WR | 0.799 | -0.007 (2) | grade | +0.003 (3) | 2/7 |
| rookie | subset | TE | 0.692 | -0.028 (2) | yprr | +0.002 (3) | 0/7 |

WR route share on n <= 3 rows, by season (live, fit all): +0.016 +0.006 +0.027 +0.034 +0.013. Rookie QB gains are
one season (+0.178 in 2022) on ~70 rows a year; noise. Everything else sits inside +-0.01 or is negative.

## 2. Gradient boosting (HistGradientBoostingRegressor per position)
max_depth 3, lr 0.05, min_samples_leaf 50, l2 1.0; iterations (<= 400, floor 20) chosen on the last training season,
then refit on all training seasons; weights = future_played; NaNs passed through natively. (a) est_ppg + ECR rank,
(b) (a) + all position-relevant PFF features (season-to-date and last-3). "live" = LOSO OLS on the live blend.

| rows | pos | a - live d_rc (up) | b - a d_rc (up) | b - live d_rc (up) | b - live d_mae |
|---|---|---|---|---|---|
| all | QB | +0.009 (3) | -0.008 (1) | +0.001 (2) | +0.047 |
| all | RB | -0.004 (1) | -0.004 (1) | -0.008 (1) | +0.081 |
| all | WR | -0.001 (2) | +0.001 (4) | +0.000 (2) | +0.063 |
| all | TE | -0.003 (2) | **+0.012 (5)** | **+0.009 (4)** | +0.009 |
| n<=3 | QB | +0.023 (3) | +0.002 (2) | +0.025 (3) | +0.109 |
| n<=3 | RB | +0.006 (3) | -0.032 (1) | -0.026 (2) | +0.128 |
| n<=3 | WR | +0.015 (4) | +0.002 (2) | +0.018 (3) | +0.108 |
| n<=3 | TE | -0.033 (1) | +0.056 (4) | **+0.024 (5)** | +0.015 |
| W<=5 | QB | +0.016 (2) | -0.014 (1) | +0.002 (2) | +0.219 |
| W<=5 | RB | -0.006 (1) | -0.004 (1) | -0.009 (2) | +0.135 |
| W<=5 | WR | +0.000 (2) | +0.001 (3) | +0.001 (2) | +0.100 |
| W<=5 | TE | -0.002 (2) | +0.015 (4) | +0.014 (4) | +0.036 |
| rookie | QB | +0.014 (4) | +0.005 (4) | +0.019 (3) | -0.061 |
| rookie | RB | -0.008 (1) | +0.007 (3) | -0.000 (3) | +0.021 |
| rookie | WR | -0.006 (1) | -0.001 (3) | -0.007 (1) | +0.163 |
| rookie | TE | -0.001 (3) | +0.005 (3) | +0.004 (3) | +0.065 |

TE (b) - (a), all rows, by season: +0.012 +0.003 +0.017 +0.017 +0.012. TE (b) - live, n <= 3: +0.024 +0.028 +0.025
+0.029 +0.013. GBM MAE is worse than the linear live blend almost everywhere (it orders a bit differently, levels
are less calibrated).

Held-out permutation importance in (b), mean drop in rank corr (folds > 0 of 5): TE route_share_l3 0.024 (5),
TE route_share 0.014 (5), QB adot 0.010 (5), RB rush_share 0.008 (5), TE rec_grade 0.008 (5), WR route_share_l3
0.005 (5), QB pass_grade_off 0.005 (4), WR tgt_share 0.004 (4). Full list: pff_followup_gbm_importance.csv.

## Follow-up verdict
Two things reach the ECR scale, and both are route share for pass catchers:
- WR route share on low-history rows (n <= 3), linear: +0.015 to +0.019, 4-5 of 5 seasons.
- TE with PFF features in a GBM: +0.009 over live overall (4/5), +0.024 on n <= 3 (5/5), driven by route share.
Grades, efficiency rates, QB and RB features, early-season and rookie subsets: noise. Caveat: this scanned ~400
cells, so a couple of 4-5/5 results are expected by chance; route share holding up in both the linear and the GBM
test, and in the original run, makes it the one signal worth a targeted test (route share as a low-n WR/TE term,
ideally from a free source), not PFF as a whole.
