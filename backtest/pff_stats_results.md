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

## Exploration: outlier flags, stability, ML residual models, archetypes (backtest/pff_explore.py)
Run on 2026-10-16. Player-level 2026 lists and charts live only in the gitignored `.cache/pff/report/index.html`.
Shares are per game played (player per-game volume / team per-game volume), so missed games don't dilute them.

### A. Do the 2026 list flags predict future_ppg - live? (2021-25, checkpoints W=5-13, future_played >= 3)
Usage-expected ppg = GBM on usage only (WR/TE: route share, target share, TPRR, targets/g, aDOT, air yards/g; RB: carries/g, rush share,
targets/g, route share, target share), LOSO by season. Flags use pooled per-position deciles. CIs are a player-season cluster bootstrap.

| flag                                | pos   |   rows |   players |   fut-est |   fut-live |   vs unflagged | vs unflagged CI   |   seasons right sign | label                   |
|:------------------------------------|:------|-------:|----------:|----------:|-----------:|---------------:|:------------------|---------------------:|:------------------------|
| usage>production (res bottom 10%)   | RB    |    236 |        74 |    -0.899 |     -0.873 |         -0.612 | [-1.66, +0.44]    |                    2 | descriptive only        |
| usage>production (res bottom 10%)   | WR    |    455 |       132 |    -0.218 |     -0.222 |          0.036 | [-0.63, +0.71]    |                    2 | descriptive only        |
| usage>production (res bottom 10%)   | TE    |    151 |        50 |     0.426 |      0.421 |          0.337 | [-0.51, +1.11]    |                    2 | descriptive only        |
| production>usage (res top 10%)      | RB    |    252 |        69 |    -0.679 |     -0.703 |         -0.427 | [-1.34, +0.41]    |                    4 | descriptive only        |
| production>usage (res top 10%)      | WR    |    450 |       126 |    -0.271 |     -0.311 |         -0.062 | [-0.78, +0.67]    |                    3 | descriptive only        |
| production>usage (res top 10%)      | TE    |    171 |        46 |    -0.465 |     -0.471 |         -0.657 | [-1.48, +0.07]    |                    4 | descriptive only        |
| role riser (L3 - STD share top 10%) | RB    |    255 |       117 |     0.179 |     -0.054 |          0.304 | [-0.47, +1.04]    |                    4 | descriptive only        |
| role riser (L3 - STD share top 10%) | WR    |    441 |       186 |     0.346 |      0.240 |          0.541 | [+0.01, +1.09]    |                    5 | historically predictive |
| role riser (L3 - STD share top 10%) | TE    |    168 |        77 |     0.705 |      0.571 |          0.544 | [-0.35, +1.45]    |                    3 | descriptive only        |
| role faller (bottom 10%)            | RB    |    236 |       108 |    -0.813 |     -0.659 |         -0.369 | [-1.12, +0.39]    |                    3 | descriptive only        |
| role faller (bottom 10%)            | WR    |    426 |       187 |    -0.582 |     -0.660 |         -0.455 | [-1.07, +0.16]    |                    3 | descriptive only        |
| role faller (bottom 10%)            | TE    |    153 |        71 |     0.056 |      0.026 |         -0.065 | [-0.91, +0.82]    |                    2 | descriptive only        |
| efficiency high (z top 10%)         | RB    |    243 |        62 |    -0.182 |     -0.362 |         -0.045 | [-1.17, +1.09]    |                    2 | descriptive only        |
| efficiency high (z top 10%)         | WR    |    466 |       105 |    -0.153 |     -0.225 |          0.033 | [-0.90, +0.91]    |                    2 | descriptive only        |
| efficiency high (z top 10%)         | TE    |    166 |        38 |    -0.057 |     -0.138 |         -0.282 | [-1.81, +1.36]    |                    2 | descriptive only        |
| efficiency low (z bottom 10%)       | RB    |    197 |        61 |     0.116 |      0.199 |          0.561 | [-0.68, +1.80]    |                    2 | descriptive only        |
| efficiency low (z bottom 10%)       | WR    |    431 |       117 |    -0.048 |     -0.030 |          0.247 | [-0.48, +0.98]    |                    2 | descriptive only        |
| efficiency low (z bottom 10%)       | TE    |    157 |        44 |    -0.218 |     -0.146 |         -0.285 | [-1.16, +0.59]    |                    3 | descriptive only        |

Continuous: weighted slope of (future - live) on the usage residual:

| pos   |   slope | CI               |   res_sd |
|:------|--------:|:-----------------|---------:|
| RB    |  -0.036 | [-0.142, +0.076] |    2.720 |
| TE    |  -0.114 | [-0.265, +0.029] |    2.089 |
| WR    |   0.011 | [-0.088, +0.109] |    2.370 |

Only WR role risers clear zero, and only barely (1 of 18 tests; chance level). Every list is descriptive only.

### B. Stability (Spearman, mean over 2021-25): weeks 1-6 vs 7-12, adjacent 3-week blocks, and links to fantasy ppg

| pos   | stat        |      n |   split_half |   block3 |   same_ppg |   next_ppg |
|:------|:------------|-------:|-------------:|---------:|-----------:|-----------:|
| QB    | rush_pg     | 28.200 |        0.766 |    0.647 |      0.465 |      0.342 |
| QB    | ttt         | 28.200 |        0.651 |    0.609 |      0.163 |      0.115 |
| QB    | scr_pg      | 28.200 |        0.601 |    0.500 |      0.393 |      0.238 |
| QB    | ppg         | 28.200 |        0.435 |    0.363 |      1.000 |      0.435 |
| QB    | ypa_pass    | 28.200 |        0.377 |    0.228 |      0.555 |      0.258 |
| QB    | pass_grade  | 28.200 |        0.353 |    0.196 |      0.543 |      0.366 |
| QB    | sack_pct    | 28.200 |        0.346 |    0.214 |     -0.283 |     -0.255 |
| QB    | db_pg       | 28.200 |        0.318 |    0.283 |      0.299 |      0.050 |
| QB    | adot_pass   | 28.200 |        0.297 |    0.302 |      0.133 |      0.079 |
| QB    | btt         | 28.200 |        0.284 |    0.266 |      0.376 |      0.236 |
| QB    | acc         | 28.200 |        0.239 |    0.066 |      0.386 |      0.215 |
| QB    | twp         | 28.200 |        0.067 |    0.049 |     -0.259 |     -0.199 |
| RB    | rush_share  | 44.600 |        0.681 |    0.713 |      0.775 |      0.516 |
| RB    | rush_pg     | 44.600 |        0.629 |    0.585 |      0.804 |      0.517 |
| RB    | targets_pg  | 44.600 |        0.613 |    0.542 |      0.612 |      0.396 |
| RB    | route_share | 44.600 |        0.586 |    0.650 |      0.661 |      0.382 |
| RB    | ppg         | 44.600 |        0.558 |    0.494 |      1.000 |      0.558 |
| RB    | elusive     | 44.600 |        0.300 |    0.162 |      0.314 |      0.154 |
| RB    | rush_grade  | 44.600 |        0.282 |    0.235 |      0.545 |      0.257 |
| RB    | rush_td_pg  | 44.600 |        0.280 |    0.246 |      0.661 |      0.271 |
| RB    | yco         | 44.600 |        0.189 |    0.155 |      0.326 |      0.167 |
| RB    | breakaway   | 44.600 |        0.155 |    0.116 |      0.332 |      0.126 |
| RB    | ypa         | 44.600 |        0.125 |    0.163 |      0.449 |      0.140 |
| TE    | inline_rate | 30.600 |        0.787 |    0.765 |     -0.136 |     -0.050 |
| TE    | slot_rate   | 30.600 |        0.776 |    0.707 |      0.121 |      0.043 |
| TE    | tgt_share   | 30.600 |        0.669 |    0.552 |      0.831 |      0.547 |
| TE    | route_share | 30.600 |        0.657 |    0.672 |      0.591 |      0.465 |
| TE    | targets_pg  | 30.600 |        0.615 |    0.521 |      0.837 |      0.507 |
| TE    | ppg         | 30.600 |        0.479 |    0.371 |      1.000 |      0.479 |
| TE    | adot        | 30.600 |        0.456 |    0.331 |      0.220 |      0.083 |
| TE    | tprr        | 30.600 |        0.384 |    0.327 |      0.690 |      0.332 |
| TE    | yprr        | 30.600 |        0.373 |    0.276 |      0.819 |      0.390 |
| TE    | yac_rec     | 30.600 |        0.350 |    0.109 |      0.231 |      0.081 |
| TE    | rec_grade   | 30.600 |        0.310 |    0.263 |      0.710 |      0.374 |
| TE    | route_grade | 30.600 |        0.299 |    0.217 |      0.776 |      0.359 |
| TE    | rec_td_pg   | 30.600 |        0.116 |    0.081 |      0.640 |      0.180 |
| TE    | drop_rate   | 30.600 |        0.028 |   -0.035 |     -0.023 |      0.079 |
| WR    | slot_rate   | 84.000 |        0.881 |    0.842 |     -0.144 |     -0.024 |
| WR    | tgt_share   | 84.000 |        0.751 |    0.688 |      0.875 |      0.678 |
| WR    | targets_pg  | 84.000 |        0.728 |    0.635 |      0.895 |      0.683 |
| WR    | ppg         | 84.000 |        0.672 |    0.529 |      1.000 |      0.672 |
| WR    | route_share | 84.000 |        0.654 |    0.678 |      0.707 |      0.479 |
| WR    | tprr        | 84.000 |        0.639 |    0.513 |      0.716 |      0.613 |
| WR    | adot        | 84.000 |        0.558 |    0.394 |     -0.004 |     -0.066 |
| WR    | rec_grade   | 84.000 |        0.549 |    0.364 |      0.816 |      0.582 |
| WR    | route_grade | 84.000 |        0.539 |    0.368 |      0.821 |      0.582 |
| WR    | yprr        | 84.000 |        0.497 |    0.380 |      0.802 |      0.535 |
| WR    | yac_rec     | 84.000 |        0.371 |    0.237 |      0.247 |      0.174 |
| WR    | rec_td_pg   | 84.000 |        0.338 |    0.169 |      0.748 |      0.476 |
| WR    | drop_rate   | 84.000 |        0.062 |    0.031 |     -0.064 |     -0.022 |

### C. ML on future_ppg - live (LOSO by season, hyperparameters on inner LOSO folds). d_rc vs live; ECR's gain was +0.008 to +0.019
`*_base` = same model with only live, est_ppg, ppg-to-date, games, week (no PFF), as a control.

| pos   | rows   | model      |   base_rc |   d_rc |   rc_up |   d_mae |   mae_down | d_rc_by_season                     |
|:------|:-------|:-----------|----------:|-------:|--------:|--------:|-----------:|:-----------------------------------|
| QB    | all    | ridge      |     0.717 | -0.033 |       2 |  -0.095 |          3 | +0.010 +0.004 -0.083 -0.039 -0.057 |
| QB    | est>5  | ridge      |     0.600 | -0.033 |       1 |   0.037 |          3 | -0.035 +0.024 -0.083 -0.045 -0.028 |
| QB    | all    | lasso      |     0.717 | -0.004 |       3 |  -0.150 |          3 | +0.008 +0.004 +0.010 -0.038 -0.006 |
| QB    | est>5  | lasso      |     0.600 | -0.008 |       2 |   0.001 |          3 | -0.022 +0.021 +0.010 -0.038 -0.011 |
| QB    | all    | gbm        |     0.717 | -0.034 |       1 |  -0.085 |          3 | -0.011 +0.018 -0.133 -0.015 -0.031 |
| QB    | est>5  | gbm        |     0.600 | -0.032 |       1 |   0.058 |          3 | -0.018 +0.045 -0.133 -0.006 -0.049 |
| QB    | all    | rf         |     0.717 | -0.032 |       2 |  -0.109 |          3 | +0.004 +0.003 -0.100 -0.031 -0.033 |
| QB    | est>5  | rf         |     0.600 | -0.023 |       1 |   0.044 |          3 | -0.004 +0.053 -0.100 -0.022 -0.044 |
| QB    | all    | mlp        |     0.717 | -0.121 |       0 |   0.264 |          2 | -0.027 -0.063 -0.270 -0.148 -0.098 |
| QB    | est>5  | mlp        |     0.600 | -0.134 |       0 |   0.420 |          0 | -0.065 -0.051 -0.270 -0.138 -0.148 |
| RB    | all    | ridge      |     0.830 | -0.001 |       2 |   0.059 |          1 | -0.002 +0.004 -0.001 +0.001 -0.005 |
| RB    | est>5  | ridge      |     0.707 | -0.003 |       3 |  -0.010 |          3 | -0.015 +0.002 -0.008 +0.008 +0.000 |
| RB    | all    | lasso      |     0.830 |  0.000 |       3 |   0.047 |          1 | -0.002 +0.001 +0.001 +0.001 -0.001 |
| RB    | est>5  | lasso      |     0.707 | -0.002 |       3 |  -0.014 |          3 | -0.011 +0.003 -0.005 +0.003 +0.001 |
| RB    | all    | gbm        |     0.830 |  0.000 |       4 |   0.037 |          2 | +0.002 +0.002 +0.000 +0.002 -0.005 |
| RB    | est>5  | gbm        |     0.707 | -0.002 |       2 |  -0.016 |          4 | -0.008 +0.000 -0.006 +0.010 -0.008 |
| RB    | all    | rf         |     0.830 |  0.000 |       3 |   0.051 |          1 | +0.004 -0.002 +0.003 +0.002 -0.006 |
| RB    | est>5  | rf         |     0.707 | -0.007 |       2 |   0.024 |          3 | -0.008 -0.006 +0.001 +0.008 -0.032 |
| RB    | all    | mlp        |     0.830 | -0.015 |       0 |   0.177 |          0 | -0.020 -0.008 -0.024 -0.005 -0.019 |
| RB    | est>5  | mlp        |     0.707 | -0.047 |       0 |   0.205 |          1 | -0.066 -0.006 -0.071 -0.043 -0.047 |
| WR    | all    | ridge      |     0.812 |  0.000 |       3 |   0.023 |          2 | -0.001 +0.001 +0.003 -0.005 +0.003 |
| WR    | est>5  | ridge      |     0.742 |  0.001 |       3 |  -0.005 |          3 | +0.001 +0.000 +0.007 -0.001 -0.001 |
| WR    | all    | lasso      |     0.812 | -0.000 |       3 |   0.028 |          2 | -0.001 +0.001 +0.002 -0.004 +0.001 |
| WR    | est>5  | lasso      |     0.742 |  0.000 |       2 |   0.008 |          3 | +0.001 -0.000 +0.002 -0.001 -0.002 |
| WR    | all    | gbm        |     0.812 |  0.000 |       2 |   0.030 |          2 | +0.001 +0.002 -0.000 -0.002 -0.000 |
| WR    | est>5  | gbm        |     0.742 | -0.000 |       3 |   0.002 |          4 | +0.005 +0.003 -0.002 +0.000 -0.006 |
| WR    | all    | rf         |     0.812 |  0.001 |       4 |   0.035 |          2 | +0.003 +0.003 -0.001 +0.001 +0.001 |
| WR    | est>5  | rf         |     0.742 | -0.002 |       2 |   0.007 |          4 | +0.003 +0.002 -0.005 -0.000 -0.011 |
| WR    | all    | mlp        |     0.812 | -0.019 |       0 |   0.147 |          0 | -0.016 -0.027 -0.025 -0.023 -0.005 |
| WR    | est>5  | mlp        |     0.742 | -0.019 |       0 |   0.109 |          0 | -0.004 -0.030 -0.021 -0.035 -0.002 |
| TE    | all    | ridge      |     0.816 | -0.015 |       0 |   0.004 |          2 | -0.009 -0.007 -0.017 -0.038 -0.005 |
| TE    | est>5  | ridge      |     0.651 |  0.025 |       5 |  -0.097 |          5 | +0.026 +0.035 +0.032 +0.004 +0.030 |
| TE    | all    | lasso      |     0.816 | -0.013 |       0 |   0.002 |          3 | -0.009 -0.006 -0.013 -0.034 -0.006 |
| TE    | est>5  | lasso      |     0.651 |  0.024 |       5 |  -0.094 |          5 | +0.025 +0.035 +0.032 +0.005 +0.022 |
| TE    | all    | gbm        |     0.816 |  0.003 |       4 |   0.018 |          2 | +0.000 +0.003 +0.007 -0.004 +0.007 |
| TE    | est>5  | gbm        |     0.651 |  0.000 |       3 |  -0.004 |          1 | +0.004 -0.002 +0.023 -0.034 +0.009 |
| TE    | all    | rf         |     0.816 |  0.006 |       4 |   0.022 |          2 | +0.000 +0.012 +0.014 -0.002 +0.007 |
| TE    | est>5  | rf         |     0.651 | -0.009 |       3 |   0.022 |          1 | +0.009 -0.010 +0.005 -0.049 +0.000 |
| TE    | all    | mlp        |     0.816 | -0.022 |       1 |   0.097 |          0 | -0.040 +0.010 -0.030 -0.031 -0.020 |
| TE    | est>5  | mlp        |     0.651 | -0.022 |       1 |   0.045 |          1 | -0.011 -0.019 -0.024 -0.064 +0.006 |
| QB    | all    | ridge_base |     0.717 | -0.005 |       3 |  -0.145 |          3 | +0.007 +0.014 +0.002 -0.019 -0.031 |
| QB    | est>5  | ridge_base |     0.600 | -0.008 |       2 |   0.023 |          3 | -0.012 +0.016 +0.002 -0.006 -0.037 |
| QB    | all    | gbm_base   |     0.717 | -0.024 |       1 |  -0.117 |          3 | -0.008 +0.014 -0.092 -0.026 -0.010 |
| QB    | est>5  | gbm_base   |     0.600 | -0.024 |       1 |   0.055 |          3 | -0.011 +0.034 -0.092 -0.045 -0.008 |
| RB    | all    | ridge_base |     0.830 | -0.000 |       3 |   0.059 |          1 | +0.001 -0.004 +0.002 +0.001 -0.000 |
| RB    | est>5  | ridge_base |     0.707 |  0.000 |       3 |  -0.021 |          4 | +0.001 -0.002 -0.000 +0.002 +0.001 |
| RB    | all    | gbm_base   |     0.830 |  0.001 |       3 |   0.055 |          1 | -0.000 -0.001 +0.001 +0.000 +0.004 |
| RB    | est>5  | gbm_base   |     0.707 | -0.004 |       0 |   0.006 |          4 | -0.007 -0.004 -0.005 -0.002 -0.000 |
| WR    | all    | ridge_base |     0.812 | -0.001 |       2 |   0.023 |          2 | -0.001 +0.001 -0.000 -0.003 +0.001 |
| WR    | est>5  | ridge_base |     0.742 | -0.002 |       1 |   0.002 |          3 | +0.003 -0.005 -0.001 -0.004 -0.003 |
| WR    | all    | gbm_base   |     0.812 | -0.000 |       3 |   0.034 |          2 | +0.001 +0.003 -0.002 -0.004 +0.001 |
| WR    | est>5  | gbm_base   |     0.742 | -0.000 |       2 |   0.011 |          3 | +0.001 +0.001 -0.001 -0.001 -0.002 |
| TE    | all    | ridge_base |     0.816 | -0.001 |       2 |   0.044 |          1 | -0.002 +0.001 -0.001 -0.005 +0.001 |
| TE    | est>5  | ridge_base |     0.651 | -0.001 |       3 |   0.027 |          1 | +0.001 +0.003 -0.001 -0.011 +0.002 |
| TE    | all    | gbm_base   |     0.816 |  0.004 |       5 |   0.034 |          1 | +0.002 +0.007 +0.005 +0.004 +0.002 |
| TE    | est>5  | gbm_base   |     0.651 | -0.009 |       1 |   0.051 |          1 | -0.004 +0.003 -0.005 -0.009 -0.030 |

TE est>5 ridge/lasso (+0.025, 5/5 seasons) is the one gain over the control. Lasso keeps rec grade (+), TD rate (-) and TPRR (-).
It loses -0.015 over all TE rows and swings -0.04 to +0.04 by season when trained on est>5 only, so it's not ready to deploy.
MLP is worst everywhere (early stopping validates on rows from the same players, so it overfits). Torch with player embeddings was skipped:
under LOSO-by-season a player embedding learns from that player's later seasons, which is leakage the Desk couldn't use live.

### C. Archetypes (PCA 85% variance then k-means, k by silhouette 3-6; fit on 2021-25 weeks 1-12; profile = top centroid z-scores)

| pos   |   cluster |   k |   silhouette |   hist_n | profile                                                          |   hist_ppg_wk1_12 |   hist_ppg_wk13_17 |   late_minus_early |
|:------|----------:|----:|-------------:|---------:|:-----------------------------------------------------------------|------------------:|-------------------:|-------------------:|
| WR    |         0 |   3 |        0.288 |      234 | tprr -0.7, tgt_share -0.7, rec_grade -0.6, yprr -0.6             |             5.773 |              5.781 |             -0.069 |
| WR    |         1 |   3 |        0.288 |      230 | tgt_share +1.0, rec_grade +0.9, tprr +0.9, yprr +0.9             |            14.130 |             13.534 |             -0.704 |
| WR    |         2 |   3 |        0.288 |      139 | slot_rate +1.3, adot -1.0, rec_td_pg -0.6, yac_rec +0.5          |             6.748 |              6.431 |             -0.406 |
| TE    |         0 |   3 |        0.244 |       64 | yprr +1.1, rec_grade +1.0, tprr +1.0, tgt_share +1.0             |            11.513 |             11.740 |              0.189 |
| TE    |         1 |   3 |        0.244 |       88 | tgt_share -0.7, slot_rate -0.7, inline_rate +0.7, yprr -0.7      |             5.440 |              5.684 |              0.222 |
| TE    |         2 |   3 |        0.244 |       61 | slot_rate +1.1, inline_rate -1.0, yac_rec -0.5, adot +0.3        |             8.034 |              8.141 |              0.217 |
| RB    |         0 |   3 |        0.298 |      134 | rush_share +0.9, rush_pg +0.9, route_share +0.7, targets_pg +0.6 |            15.219 |             14.227 |             -0.984 |
| RB    |         1 |   3 |        0.298 |       39 | elusive +1.5, yco +1.5, ypa +1.5, breakaway +1.3                 |            11.136 |              9.721 |             -1.757 |
| RB    |         2 |   3 |        0.298 |      149 | rush_pg -0.7, rush_share -0.6, rush_grade -0.6, route_share -0.6 |             7.006 |              7.573 |              0.417 |
| QB    |         0 |   5 |        0.204 |       53 | pass_grade +1.1, btt +0.7, db_pg +0.7, sack_pct -0.6             |            18.551 |             16.815 |             -1.808 |
| QB    |         1 |   5 |        0.204 |       70 | btt -0.6, adot_pass -0.5, rush_pg -0.4, pass_grade -0.4          |            13.400 |             11.894 |             -1.645 |
| QB    |         2 |   5 |        0.204 |       27 | adot_pass +1.4, acc -1.0, btt +0.8, db_pg -0.7                   |            13.369 |             13.865 |              0.523 |
| QB    |         3 |   5 |        0.204 |       14 | twp +2.1, pass_grade -1.8, acc -1.3, btt -1.2                    |             8.339 |              7.257 |             -0.782 |
| QB    |         4 |   5 |        0.204 |       32 | rush_pg +1.5, ttt +1.5, sack_pct +0.8, twp -0.5                  |            17.987 |             17.174 |             -1.887 |
