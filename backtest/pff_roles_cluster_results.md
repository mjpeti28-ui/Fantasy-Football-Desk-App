# PFF features for Roles style clustering (aggregates only)

`python backtest/pff_roles_cluster.py [--comps]`. Cache only, 0 PFF API calls.

**Setup.** Each row is one player-season, 2021-25, weeks 1-12. Rows pass the real-volume filter in `pff_explore.eligible` (at least 6 games) and must be matched to an ESPN ID. That gives WR 549, RB 299, TE 202 and QB 174 rows. Every feature set is fit on the same rows with a full-covariance GMM, pooled across seasons, choosing k by BIC over 2-6. The early label applies the same scaler and GMM to the w01-02 window and takes the hard label. The year-to-year (yoy) check compares w01-12 labels in consecutive seasons. The ARI column compares against the labels from the shipped `roles_model.json`. The kappa SE is a bootstrap over rows with the labels held fixed, so it understates the real SE.

**Baseline caveat.** The shipped benchmarks (63-78% early, 65-81% yoy) use the noise-aware classifier at 3 games, the full 17-week season and 2016-25 data. On this protocol (hard label after 2 games, compared with weeks 1-12, 2021-25 only) the current features score lower. Compare within the table below, not against those numbers.

| pos | set | k (BIC k) | early acc | early κ ± se | yoy same | yoy κ ± se | ARI vs shipped |
|---|---|---|---|---|---|---|---|
| WR | current | 3 (3) | .54 | .27 ± .03 | .64 | .34 ± .05 | .27 |
| WR | current + PFF | 3 (3) | .62 | .42 ± .03 | .76 | .56 ± .04 | .10 |
| WR | PFF only | 3 (3) | .65 | .47 ± .03 | .71 | .47 ± .04 | .09 |
| TE | current | 2 (2) | .66 | .31 ± .07 | .73 | .45 ± .09 | .27 |
| TE | current | 3 | .50 | .18 ± .06 | .60 | .36 ± .08 | .28 |
| TE | current + PFF | 2 (2) | .78 | .50 ± .07 | .75 | .50 ± .08 | .17 |
| TE | current + PFF | 3 | .70 | .40 ± .06 | .64 | .38 ± .08 | .27 |
| TE | PFF only | 3 | .75 | .47 ± .06 | .77 | .57 ± .07 | .19 |
| RB | current | 3 (3) | .67 | .41 ± .05 | .66 | .33 ± .07 | .74 |
| RB | current + PFF | 3 | .74 | .51 ± .05 | .69 | .22 ± .08 | .50 |
| RB | current + PFF | 2 (2) | .83 | .37 ± .08 | .89 | .26 ± .13 | .15 |
| RB | PFF only | 3 | .57 | .31 ± .05 | .53 | .21 ± .06 | .04 |
| QB | current | 2 (2) | .81 | .54 ± .07 | .96 | .88 ± .06 | .51 |
| QB | current | 3 | .69 | .53 ± .05 | .73 | .58 ± .07 | .56 |
| QB | current + PFF | 3 (3) | .68 | .50 ± .06 | .68 | .49 ± .08 | .32 |

Feature sets:

| pos | current features | PFF features added |
|---|---|---|
| WR | adot, behind_r, deep_r | slot_rate, route_rate |
| TE | tgt_per_snap, adot, behind_r | inline_rate, slot_rate, route_rate |
| RB | rec_mix, pd_ratio | route_rate, targets per route, zone share of zone+gap runs |
| QB | des_rush_g, scr_rate, p_adot | time to throw, PFF scramble rate |

Two PFF columns are left out because they are nearly redundant: wide_rate ≈ 1 - slot - inline, and pass_block_rate ≈ 1 - route_rate.

## What the new groups split

These are centroids for current + PFF at the k shown.

**WR, k=3.** Slot alignment becomes the main axis. The shipped "Field stretcher" label absorbs most outside receivers, and PFF splits them into two groups:

| suggested name | share | slot % | aDOT | deep % | behind-LOS % | PPG |
|---|---|---|---|---|---|---|
| Slot / underneath | 34% | 62 | 8.4 | 11 | 17 | 9.0 |
| Outside primary | 56% | 25 | 11.8 | 18 | 9 | 10.4 |
| Outside vertical | 11% | 35 | 15.1 | 31 | 13 | 6.0 |

route_rate is about 93% for every WR group and adds nothing. A one-off run without it moved early κ to .49 and yoy κ to .50, both within noise.

**TE, k=2.** This is the inline vs move split the current features can't see:

| suggested name | share | inline % | slot % | route rate | tgt/snap | PPG |
|---|---|---|---|---|---|---|
| Inline / Y | 62% | 50 | 35 | 85% | .089 | 7.1 |
| Move / slot | 38% | 29 | 52 | 90% | .125 | 9.6 |

BIC prefers k=2 on the current features too, and the shipped third TE group is not stable on this protocol (early 50%). route_rate does help TE: dropping it lowered yoy κ to .38.

**RB.** PFF adds little. Route rate is about 77% in every group. Zone share mostly reflects team scheme, so the PFF-only clusters are scheme groups (ARI .04 vs shipped). current + PFF at k=3 reproduces the shipped early-down / all-purpose / passing-down groups, with early κ up (.41 to .51) but yoy κ down (.33 to .22).

**QB.** Time to throw tracks mobility (2.6s for pocket, 3.06s for dual-threat) and adds no new information. The current 3-feature set at k=2 (dual-threat vs everyone else) is already 96% stable year to year. The third shipped group is a soft split among the non-runners.

## Comps backtest (2021-25 rows, LOSO)

I added the 3 PFF-only cluster probabilities from the w01-(W-1) window to the comp distance:

| setting | RB MAE / ρ | WR MAE / ρ | TE MAE / ρ | QB MAE / ρ |
|---|---|---|---|---|
| base | 2.978 / .790 | 2.680 / .778 | 1.853 / .814 | 3.766 / .520 |
| + PFF probs, weight 0.5 | 2.975 / .786 | 2.706 / .773 | 1.873 / .808 | 3.756 / .519 |
| + PFF probs, weight 1.0 | 3.084 / .770 | 2.759 / .766 | 1.968 / .788 | 3.911 / .428 |

The probabilities are neutral to harmful, so they should stay out of the comps.

## Verdict (opinion)

- **WR:** a real stability gain from adding slot_rate. Early κ goes from .27 to .42-.49 and yoy κ from .34 to .50-.56, well beyond the SE. This is a different partition (ARI ~.1), not a refinement of the current one.
- **TE:** a likely early-label gain (κ .31 to .50 at k=2). yoy is within noise. Two groups, inline vs move, look better than the current three.
- **RB and QB:** keep the current features. PFF does not beat them on stability.
- **Not worth it for comps.**
- **Cost of shipping:** live labels would need weekly PFF pulls (one receiving/summary call per week for WR/TE).
- **Sample size:** TE has only 109 yoy pairs and QB 100. GMMs are fit on pooled seasons, including the test season (unsupervised, so this is a minor leak).
