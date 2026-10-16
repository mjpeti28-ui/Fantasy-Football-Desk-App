"""Ownership as a market signal (ESPN % rostered from the FantasyPros archive).

Checkpoint rows from ecr_blend_rows.csv (v2 + ECR blend, LOSO). Adds ESPN-wide ownership
at the last Friday scrape before W and its change over the prior 1-2 weeks. Tests:
  ROS ppg / total points rank corr gain on top of v2+ECR (LOSO regression)
  waiver replay: pick free agents (in *this* league) by blend vs blend + ownership trend
  python3 backtest/market_signal.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import nflverse_data as nd  # noqa: E402
from backtest import ecr_signal as es  # noqa: E402
from backtest import multi_season as ms  # noqa: E402


def ownership() -> pd.DataFrame:
    E = nd.ecr_table().dropna(subset=["espn_id"])
    E = E[E["page_type"].isin(["redraft-qb", "redraft-rb", "redraft-wr", "redraft-te"]) & E["player_owned_espn"].notna()]
    E["espn_id"] = E["espn_id"].astype("int64")
    o = E.groupby(["season", "week", "espn_id"])["player_owned_espn"].max().rename("own").reset_index()
    return o


def gain(x, a, b, target, by=("season", "week", "position")):
    per = [ms.spearman(g[b], g[target]) - ms.spearman(g[a], g[target]) for _, g in x.groupby(list(by)) if len(g) > 5]
    per = np.array(per)
    rng = np.random.default_rng(0)
    bs = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(2000)]
    return f"{per.mean():+.4f} [{np.percentile(bs, 2.5):+.4f},{np.percentile(bs, 97.5):+.4f}]"


if __name__ == "__main__":
    x = pd.read_csv(es.OUT / "ecr_blend_rows.csv")
    o = ownership()
    # ownership known when W's waivers run: the scrape for week W-1; trend vs W-2 and W-3
    for lag, name in ((1, "own"), (2, "own_m1"), (3, "own_m2")):
        x = x.merge(o.assign(week=o["week"] + lag).rename(columns={"own": name}), on=["season", "week", "espn_id"], how="left")
    print("ownership coverage", round(x["own"].notna().mean(), 3))
    x = x.dropna(subset=["own"])
    x["d1"] = (x["own"] - x["own_m1"]).fillna(0)
    x["d2"] = (x["own"] - x["own_m2"]).fillna(x["d1"])
    x["own01"] = x["own"] / 100
    x["tot_blend"] = x["v2_total"] * x["est_blend"] / x["est_ppg"]
    g = x[x["future_played"] >= 3].copy()
    g["p0"] = es.loso_reg(g, ["est_blend"], "future_ppg", "future_played")
    for name, cols in (("+own", ["est_blend", "own01"]), ("+trend", ["est_blend", "d1", "d2"]), ("+own+trend", ["est_blend", "own01", "d1", "d2"])):
        g["p_" + name] = es.loso_reg(g, cols, "future_ppg", "future_played")
        print(f"ROS ppg {name:11s} gain {gain(g, 'p0', 'p_' + name, 'future_ppg')}  MAE {(g['p_'+name]-g['future_ppg']).abs().mean():.3f} (base {(g['p0']-g['future_ppg']).abs().mean():.3f})")
    # totals, including players who stop playing (the market may know about injuries/roles)
    x["neg_d2"] = x["d2"]
    t = x.copy()
    t["p0"] = es.loso_reg(t.assign(fp=t["future_points"]), ["tot_blend"], "fp")
    t["p1"] = es.loso_reg(t.assign(fp=t["future_points"]), ["tot_blend", "own01", "d1", "d2"], "fp")
    print("ROS total points +own+trend gain", gain(t, "p0", "p1", "future_points"))
    # waiver replay in this league's free-agent pool
    fa = t[~t["rostered"].astype(bool)]
    recs = []
    for (yr, W, pos), h in fa.groupby(["season", "week", "position"]):
        k = int(h["added_now"].sum())
        if k == 0:
            continue
        recs.append({"k": k, "position": pos, "managers": h.loc[h["added_now"].astype(bool), "future_points"].mean(),
                     "blend": h.nlargest(k, "tot_blend")["future_points"].mean(),
                     "blend+market": h.nlargest(k, "p1")["future_points"].mean(),
                     "trend only": h.nlargest(k, "d2")["future_points"].mean()})
    r = pd.DataFrame(recs)
    cols = ["managers", "blend", "blend+market", "trend only"]
    w = r.assign(**{c: r[c] * r["k"] for c in cols})
    tot = w.groupby("position")[cols + ["k"]].sum()
    tot.loc["ALL"] = tot.sum()
    print("waiver replay (FA pool with ownership data):\n", tot[cols].div(tot["k"], axis=0).round(1).assign(k=tot["k"]).to_string())
