"""Simple average vs fitted weights across forecast sources (the FFA finding, tested here).

Sources per checkpoint (usage_rows_*.csv, LOSO; ECR from last Friday's archived scrape):
  A = ours     : preseason prior shrunk toward observed ppg (no ESPN weekly, no experts)
  B = ESPN     : ESPN's weekly projection for week W (as a per-game rate)
  C = experts  : FantasyPros ROS consensus rank, rank-matched onto the mean of A and B
Combinations: each alone; current two-stage fitted blend; equal thirds; equal halves of
each pair; median of three; fitted three-way weights (LOSO grid). Metrics: within-week rank
corr and MAE of rest-of-season ppg (players with >= 3 later games), rank corr of ROS totals
(x v2 availability), bias, per-season results, and stability of fitted weights across folds.
  python3 backtest/source_averaging.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import nflverse_data as nd  # noqa: E402
from backtest import multi_season as ms  # noqa: E402

OUT = ROOT / "history" / "backtest" / "multi"
SKILL = ["QB", "RB", "WR", "TE"]


def load() -> pd.DataFrame:
    d = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob("usage_rows_*.csv"))], ignore_index=True)
    d = d[d["position"].isin(SKILL) & (d["ours_ppg"] > 0)]
    E = nd.ecr_table().dropna(subset=["espn_id"])
    E = E[E["page_type"].isin(["redraft-qb", "redraft-rb", "redraft-wr", "redraft-te"])]
    E["espn_id"] = E["espn_id"].astype("int64")
    E = E.groupby(["season", "week", "espn_id"])["ecr"].min().reset_index()
    E["week"] += 1  # last Friday's scrape is what's known at checkpoint W
    d = d.merge(E, on=["season", "week", "espn_id"], how="inner")
    d["A"] = d["ours_ppg"]
    d["B"] = d["espn_now"].where(d["espn_now"] > 0, d["ours_ppg"])  # bye/zero -> fall back to ours
    d["AB"] = (d["A"] + d["B"]) / 2
    d["C"] = np.nan
    for _, g in d.groupby(["season", "week", "position"]):
        vals = np.sort(g["AB"].to_numpy())[::-1]
        order = g["ecr"].rank(method="first").astype(int).to_numpy() - 1
        d.loc[g.index, "C"] = vals[order]
    d["avail_games"] = d["v2_total"] / d["est_ppg"].where(d["est_ppg"] > 0)
    return d


def score(d, col):
    g = d[d["future_played"] >= 3]
    rho = np.mean([ms.spearman(h[col], h["future_ppg"]) for _, h in g.groupby(["season", "week", "position"]) if len(h) > 5])
    tot = d[col] * d["avail_games"]
    rho_t = np.mean([ms.spearman(tot[h.index], h["future_points"]) for _, h in d.groupby(["season", "week", "position"]) if len(h) > 5])
    return {"rho_ppg": rho, "mae_ppg": float((g[col] - g["future_ppg"]).abs().mean()),
            "bias": float((g[col] - g["future_ppg"]).mean()), "rho_total": rho_t}


def fitted_three(d, grid=np.round(np.arange(0, 1.01, 0.1), 2)):
    """LOSO: per position, weights (a, b, c) on the simplex maximizing rank corr on training seasons."""
    out = pd.Series(np.nan, index=d.index)
    chosen = {}
    combos = [(a, b, round(1 - a - b, 2)) for a in grid for b in grid if a + b <= 1.0001]
    for held in sorted(d["season"].unique()):
        for pos in SKILL:
            tr = d[(d["season"] != held) & (d["position"] == pos) & (d["future_played"] >= 3)]
            best, bw = -9, None
            groups = [h for _, h in tr.groupby(["season", "week"]) if len(h) > 5]
            for a, b, c in combos:
                s = np.mean([ms.spearman(a * h["A"] + b * h["B"] + c * h["C"], h["future_ppg"]) for h in groups])
                if s > best:
                    best, bw = s, (a, b, c)
            chosen[(held, pos)] = bw
            m = (d["season"] == held) & (d["position"] == pos)
            out[m] = bw[0] * d.loc[m, "A"] + bw[1] * d.loc[m, "B"] + bw[2] * d.loc[m, "C"]
    return out, chosen


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    d = load()
    print("rows", len(d), d.groupby("season").size().to_dict())
    d["current (fitted 2-stage)"] = np.nan
    # the live model's structure: est = wE*B + (1-wE)*A, then (1-wC)*est + wC*rank-matched ECR
    p_all = {}
    for held in sorted(d["season"].unique()):
        rows = pd.read_csv(OUT / "player_rows.csv")
        p = ms.fit_params(rows[rows["season"] != held])
        wc = ms.fit_ecr_weight([y for y in ms.SEASONS if y != held])
        m = d["season"] == held
        we = d.loc[m, "position"].map(p.espn_weight)
        est = np.where(d.loc[m, "espn_now"] > 0, we * d.loc[m, "B"] + (1 - we) * d.loc[m, "A"], d.loc[m, "A"])
        wcs = d.loc[m, "position"].map(wc).fillna(0)
        d.loc[m, "current (fitted 2-stage)"] = (1 - wcs) * est + wcs * d.loc[m, "C"]
        p_all[held] = (p.espn_weight, wc)
    d["equal thirds"] = (d["A"] + d["B"] + d["C"]) / 3
    d["median of 3"] = d[["A", "B", "C"]].median(axis=1)
    d["A+B halves"] = (d["A"] + d["B"]) / 2
    d["A+C halves"] = (d["A"] + d["C"]) / 2
    d["B+C halves"] = (d["B"] + d["C"]) / 2
    d["fitted 3-way"], chosen = fitted_three(d)
    cols = ["A", "B", "C", "A+B halves", "A+C halves", "B+C halves", "equal thirds", "median of 3", "current (fitted 2-stage)", "fitted 3-way"]
    res = pd.DataFrame({c: score(d, c) for c in cols}).T
    print(res.round(4).to_string())
    # per season rank corr
    per = pd.DataFrame({c: {y: score(d[d["season"] == y], c)["rho_ppg"] for y in sorted(d["season"].unique())} for c in cols})
    print("\nrank corr (ppg) by season:\n", per.round(3).to_string())
    # paired bootstrap: equal thirds vs current, fitted 3-way vs equal thirds
    g = d[d["future_played"] >= 3]
    rng = np.random.default_rng(0)
    for a, b in (("current (fitted 2-stage)", "equal thirds"), ("equal thirds", "fitted 3-way"), ("current (fitted 2-stage)", "median of 3")):
        diffs = np.array([ms.spearman(h[b], h["future_ppg"]) - ms.spearman(h[a], h["future_ppg"]) for _, h in g.groupby(["season", "week", "position"]) if len(h) > 5])
        bs = [diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(2000)]
        print(f"{b} minus {a}: {diffs.mean():+.4f} [{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}]")
    print("\nfitted 3-way weights (A ours, B ESPN, C experts) by held-out season:")
    print(pd.Series(chosen).unstack().to_string())
    d.to_csv(OUT / "source_averaging_rows.csv", index=False)
