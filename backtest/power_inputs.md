# Power inputs: does any new input improve Desk Power?

Target ROS PF/g. Strength = leak-free 3-wk smoothed next-week projection lineup (byes filled). No projection after week W+1 used. Diffs are candidate minus baseline, per league averaged over checkpoints for 'pooled'; 95% CI = league bootstrap.

## Sleeper, clean set (1000 leagues, 3996 league-checkpoints, 45430 team rows)

Baseline blend = LOSO OLS on z(strength 3-wk smoothed), z(PF/g), fit per checkpoint.

| cp | baseline Spearman | baseline pairwise | baseline z-weights (strength, PF/g) |
|---|---|---|---|
| wk4 | 0.470 | 0.679 | +0.365, +0.196 |
| wk6 | 0.479 | 0.683 | +0.286, +0.294 |
| wk8 | 0.498 | 0.690 | +0.322, +0.265 |
| wk10 | 0.465 | 0.677 | +0.258, +0.296 |

## 1. PF split: projected PF/g + over/under replaces PF/g

### 1. split

features: st + proj_pf_g + over. proj_pf_g = mean of started players' Sleeper projections (weeks 1..W); over = PF/g - proj_pf_g.

| cp | leagues | Spearman diff vs base [95% CI] | pairwise diff vs base [95% CI] |
|---|---|---|---|
| pooled | 1000 | +0.0071 [+0.0023,+0.0118] | +0.0029 [+0.0010,+0.0048] |
| wk4 | 1000 | +0.0118 [+0.0051,+0.0186] | +0.0055 [+0.0027,+0.0084] |
| wk6 | 1000 | +0.0125 [+0.0047,+0.0203] | +0.0047 [+0.0014,+0.0077] |
| wk8 | 1000 | +0.0008 [-0.0048,+0.0067] | +0.0000 [-0.0025,+0.0024] |
| wk10 | 996 | +0.0033 [-0.0027,+0.0095] | +0.0013 [-0.0010,+0.0038] |

Fitted z-weights (all seasons) by cp: wk4: st +0.242, proj_pf_g +0.285, over +0.095; wk6: st +0.183, proj_pf_g +0.353, over +0.118; wk8: st +0.255, proj_pf_g +0.288, over +0.101; wk10: st +0.204, proj_pf_g +0.311, over +0.092

Luck-heavy teams: rank under split minus rank under baseline (positive = split ranks them lower).

| group | n | median rank chg | mean rank chg | mean over | mean ROS rank minus base rank |
|---|---|---|---|---|---|
| over +10..+14 | 1634 | +0.0 | +0.75 | +11.8 | +1.66 |
| over >= +12 | 2003 | +0.0 | +1.00 | +16.9 | +1.64 |
| over <= -12 | 10403 | +0.0 | -0.48 | -18.1 | -1.00 |

## 2. Activity

### 2a. same-season adds/wk to date

features: st + pf_g + adds_wk. waiver + FA adds, legs 1..W, per week.

| cp | leagues | Spearman diff vs base [95% CI] | pairwise diff vs base [95% CI] |
|---|---|---|---|
| pooled | 1000 | +0.0060 [+0.0031,+0.0091] | +0.0021 [+0.0008,+0.0034] |
| wk4 | 1000 | +0.0071 [+0.0027,+0.0116] | +0.0029 [+0.0009,+0.0049] |
| wk6 | 1000 | +0.0081 [+0.0034,+0.0129] | +0.0020 [-0.0001,+0.0042] |
| wk8 | 1000 | +0.0051 [+0.0006,+0.0095] | +0.0021 [-0.0001,+0.0042] |
| wk10 | 996 | +0.0038 [-0.0001,+0.0077] | +0.0012 [-0.0005,+0.0030] |

Fitted z-weights (all seasons) by cp: wk4: st +0.364, pf_g +0.198, adds_wk +0.083; wk6: st +0.286, pf_g +0.294, adds_wk +0.094; wk8: st +0.325, pf_g +0.260, adds_wk +0.097; wk10: st +0.259, pf_g +0.290, adds_wk +0.068

### 2b. prior-season adds/wk (subset with user's prior season: 161 leagues, 6488 rows)

features: st + pf_g + prior_adds. baseline refit on the same subset; teams without a prior dropped.

| cp | leagues | Spearman diff vs base [95% CI] | pairwise diff vs base [95% CI] |
|---|---|---|---|
| pooled | 161 | +0.0114 [+0.0021,+0.0215] | +0.0042 [-0.0002,+0.0092] |
| wk4 | 161 | +0.0192 [+0.0029,+0.0355] | +0.0071 [+0.0001,+0.0141] |
| wk6 | 161 | +0.0197 [+0.0031,+0.0361] | +0.0078 [+0.0004,+0.0154] |
| wk8 | 161 | +0.0051 [-0.0081,+0.0181] | +0.0020 [-0.0043,+0.0083] |
| wk10 | 161 | +0.0016 [-0.0104,+0.0137] | -0.0002 [-0.0056,+0.0052] |

Fitted z-weights (all seasons) by cp: wk4: st +0.400, pf_g +0.177, prior_adds +0.122; wk6: st +0.356, pf_g +0.237, prior_adds +0.134; wk8: st +0.361, pf_g +0.233, prior_adds +0.120; wk10: st +0.300, pf_g +0.236, prior_adds +0.105

## 3. VORP

Starter VORP vs strength (same week, unsmoothed): identical within-league ranking in 14.5% of league-checkpoints; mean within-league Spearman 0.9332. Not identical because flex slots subtract the replacement level of whichever position fills them.

### 3a. starter VORP (added)

features: st + pf_g + vorp. 

| cp | leagues | Spearman diff vs base [95% CI] | pairwise diff vs base [95% CI] |
|---|---|---|---|
| pooled | 1000 | -0.0003 [-0.0010,+0.0005] | -0.0001 [-0.0004,+0.0002] |
| wk4 | 1000 | -0.0004 [-0.0023,+0.0015] | -0.0001 [-0.0010,+0.0007] |
| wk6 | 1000 | +0.0001 [-0.0009,+0.0011] | -0.0000 [-0.0005,+0.0004] |
| wk8 | 1000 | -0.0006 [-0.0023,+0.0009] | -0.0003 [-0.0009,+0.0004] |
| wk10 | 996 | -0.0001 [-0.0010,+0.0008] | +0.0000 [-0.0004,+0.0005] |

Fitted z-weights (all seasons) by cp: wk4: st +0.269, pf_g +0.196, vorp +0.099; wk6: st +0.246, pf_g +0.294, vorp +0.041; wk8: st +0.264, pf_g +0.264, vorp +0.061; wk10: st +0.244, pf_g +0.295, vorp +0.014

### 3a'. starter VORP (replaces strength)

features: vorp + pf_g. 

| cp | leagues | Spearman diff vs base [95% CI] | pairwise diff vs base [95% CI] |
|---|---|---|---|
| pooled | 1000 | -0.0049 [-0.0073,-0.0023] | -0.0024 [-0.0034,-0.0013] |
| wk4 | 1000 | -0.0036 [-0.0078,+0.0012] | -0.0014 [-0.0033,+0.0004] |
| wk6 | 1000 | -0.0022 [-0.0060,+0.0016] | -0.0009 [-0.0026,+0.0007] |
| wk8 | 1000 | -0.0085 [-0.0130,-0.0042] | -0.0044 [-0.0063,-0.0026] |
| wk10 | 996 | -0.0053 [-0.0092,-0.0013] | -0.0027 [-0.0045,-0.0010] |

Fitted z-weights (all seasons) by cp: wk4: vorp +0.353, pf_g +0.209; wk6: vorp +0.270, pf_g +0.309; wk8: vorp +0.295, pf_g +0.292; wk10: vorp +0.226, pf_g +0.325

### 3b. full-roster VORP, bench above replacement x0.15 (added)

features: st + pf_g + vorp_full. 

| cp | leagues | Spearman diff vs base [95% CI] | pairwise diff vs base [95% CI] |
|---|---|---|---|
| pooled | 1000 | +0.0004 [-0.0007,+0.0014] | +0.0000 [-0.0004,+0.0005] |
| wk4 | 1000 | +0.0006 [-0.0018,+0.0030] | +0.0004 [-0.0007,+0.0014] |
| wk6 | 1000 | +0.0010 [-0.0007,+0.0028] | +0.0002 [-0.0006,+0.0010] |
| wk8 | 1000 | +0.0000 [-0.0021,+0.0021] | -0.0003 [-0.0012,+0.0006] |
| wk10 | 996 | -0.0000 [-0.0015,+0.0014] | -0.0001 [-0.0008,+0.0005] |

Fitted z-weights (all seasons) by cp: wk4: st +0.237, pf_g +0.196, vorp_full +0.133; wk6: st +0.210, pf_g +0.293, vorp_full +0.080; wk8: st +0.232, pf_g +0.263, vorp_full +0.095; wk10: st +0.217, pf_g +0.294, vorp_full +0.044

### 3b'. full-roster VORP (replaces strength)

features: vorp_full + pf_g. 

| cp | leagues | Spearman diff vs base [95% CI] | pairwise diff vs base [95% CI] |
|---|---|---|---|
| pooled | 1000 | -0.0025 [-0.0051,+0.0001] | -0.0012 [-0.0023,-0.0002] |
| wk4 | 1000 | -0.0015 [-0.0062,+0.0029] | -0.0006 [-0.0025,+0.0014] |
| wk6 | 1000 | +0.0000 [-0.0038,+0.0039] | +0.0003 [-0.0014,+0.0020] |
| wk8 | 1000 | -0.0049 [-0.0094,-0.0002] | -0.0026 [-0.0045,-0.0006] |
| wk10 | 996 | -0.0037 [-0.0078,+0.0002] | -0.0022 [-0.0038,-0.0004] |

Fitted z-weights (all seasons) by cp: wk4: vorp_full +0.354, pf_g +0.209; wk6: vorp_full +0.273, pf_g +0.307; wk8: vorp_full +0.300, pf_g +0.288; wk10: vorp_full +0.232, pf_g +0.321

### 3c. scarcity-scaled VORP (added)

features: st + pf_g + scar. each starter (proj - best FA at pos) x sd(starter proj at pos in league) / mean sd over positions.

| cp | leagues | Spearman diff vs base [95% CI] | pairwise diff vs base [95% CI] |
|---|---|---|---|
| pooled | 1000 | -0.0003 [-0.0013,+0.0008] | -0.0001 [-0.0006,+0.0004] |
| wk4 | 1000 | +0.0005 [-0.0025,+0.0034] | +0.0004 [-0.0010,+0.0017] |
| wk6 | 1000 | +0.0001 [-0.0018,+0.0020] | -0.0001 [-0.0009,+0.0007] |
| wk8 | 1000 | -0.0018 [-0.0039,+0.0003] | -0.0006 [-0.0015,+0.0004] |
| wk10 | 996 | +0.0001 [-0.0010,+0.0011] | -0.0001 [-0.0006,+0.0004] |

Fitted z-weights (all seasons) by cp: wk4: st +0.199, pf_g +0.197, scar +0.172; wk6: st +0.212, pf_g +0.294, scar +0.078; wk8: st +0.254, pf_g +0.264, scar +0.072; wk10: st +0.238, pf_g +0.295, scar +0.022

### 3c'. scarcity-scaled VORP (replaces strength)

features: scar + pf_g. 

| cp | leagues | Spearman diff vs base [95% CI] | pairwise diff vs base [95% CI] |
|---|---|---|---|
| pooled | 1000 | -0.0040 [-0.0068,-0.0013] | -0.0022 [-0.0033,-0.0010] |
| wk4 | 1000 | +0.0004 [-0.0046,+0.0057] | -0.0005 [-0.0027,+0.0018] |
| wk6 | 1000 | -0.0015 [-0.0059,+0.0029] | -0.0007 [-0.0026,+0.0011] |
| wk8 | 1000 | -0.0098 [-0.0149,-0.0046] | -0.0049 [-0.0070,-0.0028] |
| wk10 | 996 | -0.0052 [-0.0094,-0.0010] | -0.0027 [-0.0045,-0.0009] |

Fitted z-weights (all seasons) by cp: wk4: scar +0.353, pf_g +0.214; wk6: scar +0.266, pf_g +0.316; wk8: scar +0.285, pf_g +0.302; wk10: scar +0.217, pf_g +0.334

## 4. Elite market share

### 4. elite share (added)

features: st + pf_g + elite. share of league's top-12 QB / 24 RB / 24 WR / 12 TE projected pts (all rostered players) owned.

| cp | leagues | Spearman diff vs base [95% CI] | pairwise diff vs base [95% CI] |
|---|---|---|---|
| pooled | 1000 | -0.0027 [-0.0039,-0.0015] | -0.0012 [-0.0017,-0.0006] |
| wk4 | 1000 | -0.0023 [-0.0045,-0.0001] | -0.0011 [-0.0022,-0.0001] |
| wk6 | 1000 | -0.0040 [-0.0065,-0.0016] | -0.0017 [-0.0028,-0.0005] |
| wk8 | 1000 | -0.0036 [-0.0056,-0.0017] | -0.0014 [-0.0023,-0.0005] |
| wk10 | 996 | -0.0009 [-0.0033,+0.0014] | -0.0005 [-0.0016,+0.0005] |

Fitted z-weights (all seasons) by cp: wk4: st +0.346, pf_g +0.195, elite +0.026; wk6: st +0.270, pf_g +0.293, elite +0.021; wk8: st +0.310, pf_g +0.264, elite +0.015; wk10: st +0.233, pf_g +0.294, elite +0.032

## 5. Max's ESPN league, 2021-25 (12 teams x 5 seasons per cp; n is tiny, CIs bootstrap the 5 seasons)

Sleeper-fit weights transferred (not refit) so nothing is tuned on this league; LOSO refit on 48 rows/cp in the last column.

| test | cp | Spearman (transfer) | diff vs base | pairwise diff | LOSO-refit Spearman diff |
|---|---|---|---|---|---|
| base | 4 | 0.271 | - | - | 0.306 (base) |
| base | 6 | 0.323 | - | - | 0.284 (base) |
| base | 8 | 0.269 | - | - | 0.199 (base) |
| base | 10 | 0.382 | - | - | 0.399 (base) |
| base | all | 0.311 | - | - | 0.297 (base) |
| 1 split | 4 | 0.344 | +0.0727 [+0.0084,+0.1552] | +0.0394 [+0.0182,+0.0667] | -0.0490 [-0.1399,+0.0406] |
| 1 split | 6 | 0.345 | +0.0224 [-0.1245,+0.1538] | +0.0030 [-0.0697,+0.0636] | +0.0420 [-0.0937,+0.1664] |
| 1 split | 8 | 0.302 | +0.0336 [-0.0364,+0.1063] | +0.0121 [-0.0212,+0.0455] | -0.0741 [-0.1692,+0.0112] |
| 1 split | 10 | 0.442 | +0.0601 [-0.0322,+0.1846] | +0.0242 [-0.0121,+0.0758] | -0.0601 [-0.1510,+0.0154] |
| 1 split | all | 0.358 | +0.0472 [-0.0035,+0.0986] | +0.0197 [-0.0046,+0.0424] | -0.0353 [-0.0895,+0.0213] |
| 3a vorp+ | 4 | 0.266 | -0.0056 [-0.0182,+0.0028] | -0.0061 [-0.0182,+0.0061] | -0.0517 [-0.2126,+0.0545] |
| 3a vorp+ | 6 | 0.320 | -0.0028 [-0.0224,+0.0168] | +0.0030 [-0.0061,+0.0121] | +0.0182 [-0.0448,+0.0671] |
| 3a vorp+ | 8 | 0.259 | -0.0098 [-0.0406,+0.0098] | -0.0000 [-0.0152,+0.0121] | -0.1455 [-0.3091,-0.0280] |
| 3a vorp+ | 10 | 0.366 | -0.0154 [-0.0280,-0.0042] | -0.0152 [-0.0242,-0.0061] | +0.0168 [-0.0601,+0.0937] |
| 3a vorp+ | all | 0.303 | -0.0084 [-0.0182,+0.0003] | -0.0045 [-0.0114,+0.0023] | -0.0406 [-0.1052,+0.0168] |
| 3c scar+ | 4 | 0.271 | +0.0000 [-0.0140,+0.0196] | +0.0061 [-0.0061,+0.0152] | -0.0294 [-0.0657,+0.0042] |
| 3c scar+ | 6 | 0.320 | -0.0028 [-0.0601,+0.0685] | +0.0030 [-0.0242,+0.0303] | -0.1301 [-0.4070,+0.0657] |
| 3c scar+ | 8 | 0.284 | +0.0154 [-0.0238,+0.0671] | +0.0030 [-0.0242,+0.0303] | -0.0140 [-0.0350,+0.0112] |
| 3c scar+ | 10 | 0.354 | -0.0280 [-0.0671,+0.0028] | -0.0152 [-0.0424,+0.0061] | -0.0350 [-0.0811,+0.0084] |
| 3c scar+ | all | 0.307 | -0.0038 [-0.0276,+0.0199] | -0.0008 [-0.0129,+0.0106] | -0.0521 [-0.1255,+0.0021] |

3a/3c transfer = Sleeper baseline weights with VORP/scarcity in strength's place. Over/under mean +1.6, sd 10.1 pts/wk (ESPN projections, league scoring).

### Desk LOSO ledger vs ESPN projections as the strength input

Desk = history/ledger/<y>/wk<W+1>_eq0_players.csv xp_w<W+1> (bye -> est_ppg); ledger only covers weeks 3-8, so checkpoints 4, 6, 7 (7 stands in for 8). Same rosters, same 3-wk smoothing. Blend = Sleeper-fit baseline weights for the nearest cp.

| cp | n seasons | Spearman strength ESPN | Desk | blend ESPN | blend Desk | pairwise blend ESPN | Desk |
|---|---|---|---|---|---|---|---|
| 4 | 5 | 0.333 | 0.222 | 0.271 | 0.232 | 0.591 | 0.597 |
| 6 | 5 | 0.290 | 0.269 | 0.323 | 0.331 | 0.627 | 0.624 |
| 7 | 5 | 0.373 | 0.341 | 0.361 | 0.368 | 0.645 | 0.648 |

Desk minus ESPN strength Spearman over all season-cps (n=15): -0.0545 [-0.1301,+0.0196]


## 6. 'Everything' model ceiling test (power_ml.py; 1000 clean leagues, 45430 team rows, 122 features)

Features: top-9 starters by next-week projection x (pos, next-week proj, ppg to date, games, last-3 ppg, proj change 1 and 3 wks, target+carry share of team, snap share) + team PF/g, projected PF/g, over/under, adds/wk, strength, bench top-3 proj (raw and within-league z) + league size + cp. Target = within-league z of ROS PF/g. LOSO by season, checkpoints pooled.

| model | cp | Spearman | diff vs baseline blend | pairwise diff vs blend | diff vs blend+split | pairwise diff vs split |
|---|---|---|---|---|---|---|
| baseline blend | pooled | 0.478 | +0.0000 [+0.0000,+0.0000] | +0.0000 [+0.0000,+0.0000] | -0.0071 [-0.0122,-0.0021] | -0.0029 [-0.0048,-0.0009] |
| baseline blend | wk4 | 0.470 | +0.0000 [+0.0000,+0.0000] | +0.0000 [+0.0000,+0.0000] | -0.0118 [-0.0185,-0.0050] | -0.0055 [-0.0082,-0.0026] |
| baseline blend | wk6 | 0.479 | +0.0000 [+0.0000,+0.0000] | +0.0000 [+0.0000,+0.0000] | -0.0125 [-0.0201,-0.0046] | -0.0047 [-0.0079,-0.0014] |
| baseline blend | wk8 | 0.498 | +0.0000 [+0.0000,+0.0000] | +0.0000 [+0.0000,+0.0000] | -0.0008 [-0.0065,+0.0046] | -0.0000 [-0.0025,+0.0024] |
| baseline blend | wk10 | 0.465 | +0.0000 [+0.0000,+0.0000] | +0.0000 [+0.0000,+0.0000] | -0.0033 [-0.0090,+0.0030] | -0.0013 [-0.0038,+0.0012] |
| blend+split | pooled | 0.485 | +0.0071 [+0.0021,+0.0121] | +0.0029 [+0.0009,+0.0048] | +0.0000 [+0.0000,+0.0000] | +0.0000 [+0.0000,+0.0000] |
| blend+split | wk4 | 0.482 | +0.0118 [+0.0052,+0.0189] | +0.0055 [+0.0027,+0.0084] | +0.0000 [+0.0000,+0.0000] | +0.0000 [+0.0000,+0.0000] |
| blend+split | wk6 | 0.492 | +0.0125 [+0.0047,+0.0201] | +0.0047 [+0.0013,+0.0082] | +0.0000 [+0.0000,+0.0000] | +0.0000 [+0.0000,+0.0000] |
| blend+split | wk8 | 0.499 | +0.0008 [-0.0044,+0.0063] | +0.0000 [-0.0022,+0.0023] | +0.0000 [+0.0000,+0.0000] | +0.0000 [+0.0000,+0.0000] |
| blend+split | wk10 | 0.469 | +0.0033 [-0.0028,+0.0098] | +0.0013 [-0.0011,+0.0039] | +0.0000 [+0.0000,+0.0000] | +0.0000 [+0.0000,+0.0000] |
| GBM | pooled | 0.470 | -0.0086 [-0.0153,-0.0019] | -0.0038 [-0.0065,-0.0010] | -0.0157 [-0.0214,-0.0096] | -0.0067 [-0.0090,-0.0041] |
| GBM | wk4 | 0.448 | -0.0225 [-0.0327,-0.0118] | -0.0097 [-0.0142,-0.0052] | -0.0343 [-0.0448,-0.0246] | -0.0152 [-0.0195,-0.0109] |
| GBM | wk6 | 0.479 | -0.0005 [-0.0103,+0.0101] | -0.0003 [-0.0047,+0.0041] | -0.0130 [-0.0222,-0.0035] | -0.0050 [-0.0091,-0.0010] |
| GBM | wk8 | 0.494 | -0.0038 [-0.0148,+0.0063] | -0.0017 [-0.0060,+0.0026] | -0.0047 [-0.0134,+0.0040] | -0.0018 [-0.0057,+0.0021] |
| GBM | wk10 | 0.458 | -0.0076 [-0.0184,+0.0033] | -0.0036 [-0.0078,+0.0007] | -0.0109 [-0.0201,-0.0018] | -0.0049 [-0.0087,-0.0010] |
| MLP | pooled | 0.454 | -0.0241 [-0.0312,-0.0172] | -0.0105 [-0.0135,-0.0076] | -0.0312 [-0.0379,-0.0246] | -0.0134 [-0.0161,-0.0107] |
| MLP | wk4 | 0.437 | -0.0335 [-0.0442,-0.0232] | -0.0147 [-0.0190,-0.0101] | -0.0453 [-0.0561,-0.0346] | -0.0202 [-0.0247,-0.0158] |
| MLP | wk6 | 0.468 | -0.0110 [-0.0222,+0.0000] | -0.0054 [-0.0101,-0.0008] | -0.0235 [-0.0340,-0.0130] | -0.0101 [-0.0143,-0.0058] |
| MLP | wk8 | 0.467 | -0.0306 [-0.0409,-0.0195] | -0.0124 [-0.0169,-0.0080] | -0.0314 [-0.0419,-0.0211] | -0.0124 [-0.0169,-0.0080] |
| MLP | wk10 | 0.443 | -0.0218 [-0.0336,-0.0105] | -0.0096 [-0.0143,-0.0047] | -0.0251 [-0.0358,-0.0145] | -0.0109 [-0.0156,-0.0063] |
| pairwise GBM | pooled | 0.490 | +0.0123 [+0.0058,+0.0188] | +0.0050 [+0.0024,+0.0076] | +0.0052 [+0.0003,+0.0099] | +0.0021 [-0.0000,+0.0041] |
| pairwise GBM | wk4 | 0.477 | +0.0072 [-0.0018,+0.0157] | +0.0031 [-0.0005,+0.0068] | -0.0046 [-0.0121,+0.0028] | -0.0024 [-0.0056,+0.0008] |
| pairwise GBM | wk6 | 0.501 | +0.0218 [+0.0126,+0.0313] | +0.0089 [+0.0049,+0.0130] | +0.0093 [+0.0024,+0.0167] | +0.0042 [+0.0011,+0.0073] |
| pairwise GBM | wk8 | 0.507 | +0.0089 [-0.0003,+0.0173] | +0.0037 [-0.0001,+0.0073] | +0.0080 [+0.0006,+0.0154] | +0.0036 [+0.0005,+0.0069] |
| pairwise GBM | wk10 | 0.476 | +0.0111 [+0.0019,+0.0206] | +0.0040 [+0.0004,+0.0075] | +0.0078 [+0.0009,+0.0147] | +0.0027 [-0.0003,+0.0057] |
| blend+split + GBM (avg of z) | pooled | 0.487 | +0.0091 [+0.0036,+0.0150] | +0.0036 [+0.0013,+0.0058] | +0.0020 [-0.0013,+0.0054] | +0.0007 [-0.0007,+0.0022] |
| blend+split + GBM (avg of z) | wk4 | 0.478 | +0.0080 [-0.0004,+0.0161] | +0.0033 [-0.0002,+0.0067] | -0.0038 [-0.0101,+0.0024] | -0.0022 [-0.0049,+0.0005] |
| blend+split + GBM (avg of z) | wk6 | 0.496 | +0.0164 [+0.0084,+0.0248] | +0.0064 [+0.0029,+0.0101] | +0.0040 [-0.0017,+0.0096] | +0.0018 [-0.0008,+0.0044] |
| blend+split + GBM (avg of z) | wk8 | 0.505 | +0.0072 [-0.0002,+0.0144] | +0.0023 [-0.0007,+0.0053] | +0.0063 [+0.0008,+0.0122] | +0.0023 [-0.0003,+0.0046] |
| blend+split + GBM (avg of z) | wk10 | 0.470 | +0.0048 [-0.0033,+0.0127] | +0.0022 [-0.0011,+0.0054] | +0.0016 [-0.0044,+0.0072] | +0.0009 [-0.0014,+0.0032] |

Leakage check, GBM trained on target shuffled within league: Spearman vs real target -0.0140 [-0.0247,-0.0031] (should be ~0).

GBM permutation importance (drop in held-out R^2, mean over seasons), top 10: lz_proj_pf_g 0.0674, lz_pf_g 0.0561, lz_st 0.0475, lz_adds_wk 0.0072, s5_proj 0.0069, s2_snap_sh 0.0047, s0_last3 0.0047, s1_last3 0.0034, s0_d1 0.0029, s2_last3 0.0029

