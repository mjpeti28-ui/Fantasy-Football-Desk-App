# What to ship: recommendations from the 2026 systems review

Written 2026-10-05. Everything numeric below comes from backtests on 2021-2025 (leave-one-season-out forecasts at weekly checkpoints W=3..13) unless marked otherwise. Opinions are labelled **[opinion]**. Items I did not test are labelled **[untested]**. Function and file names in the "how" sections are from a read of the repo that skipped `streamlit_app.py`, `api/`, `trading.py` and `playoffs.py` internals, so verify names before editing.

## TL;DR

The point forecast (v2 valuation) is at diminishing returns. Every usage, xFP, profile and ML feature set was neutral or harmful against `est_ppg` alone. The remaining value is in how results are presented and tracked.

Ship, in this order:

1. **Trade tool prints the calibrated "direction-right" probability and calls edges under 10 noise.** Highest leverage.
2. **Waiver output gets a hit-probability column** from a thin `est_ppg`-only lookup (no new features).
3. **Trade tool shows the change in playoff and seed odds** from `season_sim`, not just points. Stop emphasizing title odds.
4. **Ledger scores the probabilities** (hit probability, bust flag) so the ML layer gets a clean forward test.
5. **One-line "why" and an error band** on every waiver/trade row.
6. **Cleanup:** drop the ESPN z-term from waivers, quarantine v1 after checking call paths.

Do not ship: usage/xFP/ML features into valuation, a logistic layer on all predictions, a "trust the sample after week X" rule, or title-odds precision work.

---

## Evidence base (so the recommendations can be checked)

| Finding | Number | Source test |
|---|---|---|
| Added feature sets vs `est_ppg` alone (within-group rank corr) | -0.001 to -0.012 | LOSO regression, OLS/ridge/boosted, 5 sets |
| Shuffled-feature placebo | -0.004 | noise floor |
| Trade-direction accuracy from added features | -0.1 to -0.4 pts | random cross-position rostered pairs |
| Bust prediction AUC, `est_ppg` only vs with features | .608 vs .628 | classifier; no decision gain |
| Waiver replay, v2 alone | +1.17 [-1.27, +3.73] | not significant; ESPN z-term did not improve it |
| Trade calibration | realized = 0.86-0.94 x predicted | trade backtest |
| Direction-right by edge bin (<5 / 5-10 / 10-20 / 20-40 / 40-80 / 80+) | 47 / 57 / 61 / 69 / 82 / 96 % | trade backtest |
| Season-sim title odds | Brier skill <= .07 | near lottery |
| Season-sim seeding/playoff odds | signal after about week 6 | project findings |
| Model vs current ppg projected forward, rolling target RMSE | 3.64 vs 5.75 (W3), 3.91 vs 4.58 (W8), 4.22 vs 4.69 (W13) | all 5 seasons favor the model at every W |
| Same, fixed target (ppg in weeks 14-17) | gap 1.66 (W3) down to 0.38-0.45 (W12-13), 90% band above zero throughout | `history/backtest/figs/realtime_accuracy.png` |
| Best blend weight on current ppg | 0 to 0.15 | RMSE gain about 0.01-0.02 |

---

## Ship now

### 1. Calibrated trade confidence (trade_value.py)

**Why.** Edges under 10 points are 47-57% direction-right, which is a coin flip. The tool currently reports a number that reads as more precise than it is. Realized gain is also 0.86-0.94 of predicted, so the raw gain is mildly optimistic.

**How.**
- In the `evaluate` and `find` output, add two columns: calibrated gain (raw x about 0.9) and P(direction right) from the bin table above.
- Label edges under 10 as "noise" and sort `find` results so they sit below the cutoff.
- Store the bin table in one constants block with its source date so it is easy to refresh after each season.

**Acceptance.** Re-run the trade backtest with the displayed probabilities and confirm each bin is within a few points of its label. Re-check yearly.

**Effort [guess].** 1-2 hours. **Risk.** Low, display only.

### 2. Waiver hit-probability column (tools/waivers.py)

**Why.** "WR30-ish median, X% chance of 14+ ppg" is easier to act on than a point estimate, and it is the one ML-flavored thing the tests did not reject, because it uses no new information.

**How.**
- Fit P(ROS ppg >= threshold | `est_ppg`, position) as a thin logistic or isotonic lookup on `ml_rows.csv` (`backtest/ml_signal.py clf` already has the machinery). Inputs are `est_ppg` and position only.
- Thresholds: the positional starter line and a "boom" line. Choose them per position from the league's own replacement and starter levels.
- Print the probability next to the point estimate. Do not rank by it.

**Acceptance [untested, must pass before shipping].** Leave-one-season-out reliability check: ECE small and no bin off by more than a few points. I have not computed ECE for this exact lookup. If it fails, show the point estimate with the error band from item 5 instead.

**Effort [guess].** 2-3 hours. **Risk.** Low, display only. The failure mode is overconfident-looking percentages, hence the acceptance gate.

### 3. Season-odds deltas in the trade tool (trade_value.py + season_sim.py)

**Why.** Seed and playoff odds carry real signal after about week 6; title odds do not (Brier skill <= .07). Points alone hide whether a trade changes your playoff path.

**How.**
- For a proposed trade, run `season_sim.simulate` on pre- and post-trade rosters with common random numbers and print delta playoff odds, delta top-2 seed odds and delta median seed.
- In `tools/season_odds.py`, keep title odds but add a one-line caveat, and show playoff/seed odds first.
- Suppress the delta before week 6, or print it with "early-season, low signal".

**Acceptance.** Sim runtime stays under what you will tolerate per trade; deltas smaller than Monte Carlo noise print as "about 0". **[opinion]** This is the most decision-relevant addition after item 1.

**Effort [guess].** Half a day. **Risk.** Medium: easy to over-read small deltas.

### 4. Score probabilities in the ledger (ledger.py)

**Why.** The ML layer was tested only retrospectively. The ledger already scores point forecasts and flags calibration outside 0.85-1.15 without applying it. Logging probabilities gives a real forward test instead of another backtest argument.

**How.**
- Each weekly run, log the hit probability from item 2 and a bust flag per player alongside the existing forecast rows.
- Score with Brier and a reliability table at season end; reuse the scorecard path.
- Decide in the offseason whether the bust signal (AUC .608 to .628) earns any wiring. Until then it is tracking only.

**Effort [guess].** 2-3 hours. **Risk.** Very low.

### 5. "Why" line and error band on every row

**Why.** Cheap, and it makes outputs checkable.

**How.**
- Why line from components `valuation.py` already computes: ours vs ESPN vs ECR gap, availability discount in ppg, waiver-replacement scale.
- Error band: typical forecast error is roughly 3.6-4.2 ppg RMSE on remaining-season ppg across weeks 3-13 (rolling target). Show it as a plain "typical miss +/- about 4 ppg". Do not show a per-week narrowing curve for the rolling target: that curve rises with W only because the window shrinks. On a fixed target, error falls from 5.23 to 4.77 over W3 to W13.
- Any points shown must use league scoring from `scoring.py`. I got this wrong earlier in this chat by quoting game points on FantasyPros scoring, which understated the league's scoring.

**Effort [guess].** 2-3 hours. **Risk.** Low.

---

## Do next season / track only

### Bust probability

AUC rose from .608 to .628 with usage and profile features, but no decision metric improved. Keep it in the ledger (item 4) and revisit with a season of forward data. **[opinion]** I expect it to stay a curiosity.

### Availability model upgrades (avail_model.py) **[untested]**

Availability was one of the few real historical gains and is already logistic. Candidate inputs are practice-report status, current-week designation and days since last game. I have no result for any of them. Treat as an experiment with the same LOSO and placebo harness, not a ship item.

### Concentration display for single-player write-ups **[untested as a predictor]**

In the Boston case study about 56% of his points came from three plays. That is a useful interpretability flag ("points concentrated in N plays") but I did not test whether it predicts anything. Display-only if built; do not feed it into valuation.

---

## Cleanup

- **Remove or demote the ESPN z-term in `tools/waivers.py`.** Replay showed no gain over v2 alone (+1.17 [-1.27, +3.73] for v2, not significant). Simplification, not a loss.
- **Quarantine v1 after checking call paths.** `trading.py` (1158 lines), `main.py`, much of `api/models.py` and the epw/power_*/simulation family look superseded by v2. **[opinion]** Two valuation paths mean two different numbers for the same player, which is an interpretability problem on its own. I did not trace what `streamlit_app.py` imports, so confirm nothing in the weekly flow calls them before moving anything.
- **`playoffs.py` (759 lines) vs `season_sim.py`.** I suspect the latter supersedes it. Verify before removing.

---

## Do not ship

| Idea | Why not |
|---|---|
| Usage, xFP, snap, profile or ML features in valuation, trade or waivers | Neutral to harmful in every test (-0.001 to -0.012 rank corr; placebo -0.004). |
| Logistic layer on every prediction | Same evidence; adds apparent confidence without information. |
| "Trust the sample after week X" rule | No crossover exists. Model beats current ppg at every checkpoint W3-W13 in all 5 seasons. |
| Blending current ppg into the model | Best weight 0-0.15, RMSE gain about 0.01-0.02. |
| Title-odds precision work | Brier skill <= .07. |

Where judgment can still beat the model **[opinion, untested]**: a single player whose role has clearly changed (injury to a teammate, coaching or scheme shift). The backtest averages over thousands of ordinary cases and says nothing about those.

---

## Suggested order and checks

1. Item 1 (trade confidence) and the cleanup of the ESPN z-term. Same session.
2. Item 2 (waiver probability), only if the ECE check passes.
3. Item 3 (season-odds deltas) once week 6+ data exists.
4. Items 4 and 5 alongside, since they only add logging and text.
5. After the season: score the ledger probabilities, decide on bust flag and availability experiments.

Before shipping anything: run `python3 tools/health_check.py`, then rerun the relevant backtest to confirm the displayed numbers still match.

## Limits of this evidence

- Five seasons; a few hundred players per checkpoint, so late-week results are the least settled.
- The fixed-target bands resample players and ignore cross-player correlation inside a season.
- The crossover test used plain average ppg as the naive baseline. A smarter naive baseline was not tested.
- Everything is ppg-when-playing; availability is evaluated separately and was not part of the crossover test.
- Backtest forecasts reuse fixed parameters fit on other seasons. In-season re-training was not tested.
