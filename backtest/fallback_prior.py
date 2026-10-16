"""When ESPN has no season projection for a player (all of 2023; some players other years),
is last season's ppg a better prior than the current fallback (mean of ESPN weekly
projections published so far)?

Rows: player_rows.csv (rostered, checkpoints 3-10), players with no ESPN season projection
and >= 4 games last season. Priors compared inside the real forecast formula (fitted k,
prior scale, ESPN weight from the other seasons). Metrics: rank corr of rest-of-season ppg
(within season-week-position), MAE, bias; early (3-5) vs later checkpoints; 2023 vs other.
  python3 backtest/fallback_prior.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import valuation as v  # noqa: E402
from backtest import multi_season as ms  # noqa: E402
from backtest.prev_season import prev_features  # noqa: E402

OUT = ROOT / "history" / "backtest" / "multi"


def est(rows, prior_col, p):
    k = rows["position"].map(p.k)
    prior = rows[prior_col] * rows["position"].map(p.prior_scale)
    ours = (k * prior + rows["n"] * rows["obs_ppg"]) / (k + rows["n"])
    w = rows["position"].map(p.espn_weight)
    return np.where(rows["espn_now"] > 0, w * rows["espn_now"] + (1 - w) * ours, ours)


if __name__ == "__main__":
    rows = pd.read_csv(OUT / "player_rows.csv")
    has_proj = {}
    for y in ms.SEASONS:
        s = v.load_season(y)
        has_proj[y] = {i for i, t in s.prior_total.items() if t and t > 0}
    rows["no_season_proj"] = [i not in has_proj[y] for y, i in zip(rows["season"], rows["espn_id"])]
    rows = rows.merge(prev_features(), on=["season", "espn_id"], how="left")
    x = rows[rows["no_season_proj"] & (rows["prev_g"] >= 4) & (rows["future_played"] >= 3) & rows["position"].isin(["QB", "RB", "WR", "TE"])].copy()
    # prior_ppg_raw already holds the fallback (mean ESPN weekly projections so far); last season is unscaled ppg
    x["prior_last"] = x["prev_ppg"] / x["position"].map(lambda p: 1.0)
    x["prior_mix"] = 0.5 * x["prior_ppg_raw"] + 0.5 * x["prev_ppg"]
    print("rows", len(x), "by season", x["season"].value_counts().sort_index().to_dict())
    res = []
    for held in ms.SEASONS:
        te = x[x["season"] == held]
        if te.empty:
            continue
        p = ms.fit_params(rows[rows["season"] != held])
        for name, col in (("fallback (ESPN weekly mean)", "prior_ppg_raw"), ("last season", "prior_last"), ("50/50", "prior_mix")):
            x.loc[te.index, name] = est(te, col, p)
    x["span"] = np.where(x["week"] <= 5, "wk3-5", "wk6-10")
    x["grp"] = np.where(x["season"] == 2023, "2023", "other")
    names = ["fallback (ESPN weekly mean)", "last season", "50/50"]
    out = []
    for (grp, span), g in x.groupby(["grp", "span"]):
        rec = {"group": grp, "span": span, "n": len(g)}
        for n in names:
            rec["rho " + n] = np.mean([ms.spearman(h[n], h["future_ppg"]) for _, h in g.groupby(["season", "week", "position"]) if len(h) > 5])
            rec["MAE " + n] = float((g[n] - g["future_ppg"]).abs().mean())
        out.append(rec)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(out).round(3).to_string(index=False))
    rng = np.random.default_rng(0)
    for n in names[1:]:
        per = [ms.spearman(h[n], h["future_ppg"]) - ms.spearman(h[names[0]], h["future_ppg"]) for _, h in x.groupby(["season", "week", "position"]) if len(h) > 5]
        per = np.array(per)
        bs = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(2000)]
        mae = (x[n] - x["future_ppg"]).abs().mean() - (x[names[0]] - x["future_ppg"]).abs().mean()
        print(f"{n}: rank corr vs fallback {per.mean():+.4f} [{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}], MAE change {mae:+.3f}")
