"""Waiver wire: who should you pick up, and do this league's managers pick well?

For every season 2021-2025 and every week W = 2..12 (after week W's games, before
week W+1): the free-agent pool is everyone with a forecast who wasn't rostered in
week W. The players actually added that week are those rostered in W+1 but not W.
Each ranking rule picks the same number of players per position as managers
actually added; we compare realized points from week W+1 through 17.

Rules: v2 forecast (expected ROS points, LOSO params), ESPN's weekly projection
for W+1, last week's points (the "hot hand"), and managers' actual adds.
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

OUT = ROOT / "history" / "backtest" / "multi"
SKILL = ["QB", "RB", "WR", "TE"]


def run():
    rows = pd.read_csv(OUT / "player_rows.csv")
    recs = []
    for yr in ms.SEASONS:
        p = ms.fit_all(rows, [y for y in ms.SEASONS if y != yr])
        s = v.load_season(yr)
        wk = s.weekly
        for W in range(2, 13):
            rost_W = set(s.roster_at(W)["espn_id"])
            rost_next = set(s.roster_at(W + 1)["espn_id"])
            adds = rost_next - rost_W
            t = v.player_table(s, W + 1, p, rostered=rost_W)
            pool = t[~t.index.isin(rost_W) & t["position"].isin(SKILL)].copy()
            if pool.empty:
                continue
            fut = wk[(wk["week"] > W) & (wk["week"] <= v.FANTASY_LAST_WEEK)].groupby("espn_id")["points"].sum()
            last = wk[wk["week"] == W].set_index("espn_id")["points"]
            pool["v2"] = v.expected_ros_points(pool, W + 1, v.FANTASY_LAST_WEEK)
            pool["espn"] = pool["espn_now"].fillna(0)
            pool["hot"] = pool.index.map(last).fillna(0)
            pool["realized"] = pool.index.map(fut).fillna(0)
            pool["added"] = pool.index.isin(adds)
            for pos, g in pool.groupby("position"):
                k = int(g["added"].sum())
                if k == 0:
                    continue
                rec = {"season": yr, "week": W, "position": pos, "k": k,
                       "managers": g.loc[g["added"], "realized"].mean(),
                       "best_possible": g["realized"].nlargest(k).mean()}
                for rule in ("v2", "espn", "hot"):
                    rec[rule] = g.nlargest(k, rule)["realized"].mean()
                rec["pool_rank_corr_v2"] = g["v2"].rank().corr(g["realized"].rank())
                recs.append(rec)
    return pd.DataFrame(recs)


if __name__ == "__main__":
    d = run()
    d.to_csv(OUT / "waiver_backtest.csv", index=False)
    w = d.assign(**{c: d[c] * d["k"] for c in ("managers", "v2", "espn", "hot", "best_possible")})
    tot = w.groupby("position")[["k", "managers", "v2", "espn", "hot", "best_possible"]].sum()
    for c in ("managers", "v2", "espn", "hot", "best_possible"):
        tot[c] = tot[c] / tot["k"]
    print("Realized rest-of-season points per pickup (same number of picks as managers made):")
    print(tot.round(1).to_string())
    allk = w[["managers", "v2", "espn", "hot", "best_possible"]].sum() / d["k"].sum()
    print("ALL:", allk.round(1).to_dict(), "| pickups:", int(d["k"].sum()))
    print("by season (v2 minus managers, pts per pickup):", (w.groupby("season")["v2"].sum() / d.groupby("season")["k"].sum() - w.groupby("season")["managers"].sum() / d.groupby("season")["k"].sum()).round(1).to_dict())
