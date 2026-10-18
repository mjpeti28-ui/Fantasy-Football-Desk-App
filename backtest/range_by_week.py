"""Do ESPN's weekly projections miss by more early in the season? Coverage of the 80% outcome range by week bucket.

Ratio = actual / ESPN weekly projection, rostered non-IR players who played (same basis as tools/ratio_dist_espn.py).
Leave-one-season-out over 2021-25: fit 10th/90th pct ratios on the other seasons, either pooled (weeks 3-14, what the
app uses) or per week bucket, and check coverage on the held-out season; then the same fits (all of 2021-25) on 2026.

    python backtest/range_by_week.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "history" / "actuals"
POS = ["QB", "RB", "WR", "TE"]
BUCKETS = [(1, 2), (3, 6), (7, 10), (11, 14), (15, 17)]


def rows(y):
    pj = pd.read_csv(A / str(y) / "espn_weekly_proj.csv").drop_duplicates(["week", "espn_id"])
    pw = pd.read_csv(A / str(y) / "players_weekly.csv")
    pw = pw[(pw["kind"] == "actual") & pw["played"].astype(bool)].drop_duplicates(["espn_id", "week"])
    R = pd.read_csv(A / str(y) / "rosters_by_week.csv")
    R = R[(R["lineup_slot"] != "IR") & R["position"].isin(POS)]
    m = R.merge(pj, on=["week", "espn_id"]).merge(pw[["espn_id", "week", "points"]], on=["espn_id", "week"])
    m = m[m["espn_proj"] > 0].copy()
    m["r"] = m["points"] / m["espn_proj"]
    m["season"] = y
    m["bk"] = pd.cut(m["week"], [b[0] - 0.5 for b in BUCKETS] + [BUCKETS[-1][1] + 0.5], labels=[f"{a}-{b}" for a, b in BUCKETS])
    return m[["season", "week", "bk", "position", "r", "espn_proj", "points"]]


def cover(test, lo, hi):
    return ((test["r"] >= lo) & (test["r"] <= hi)).mean()


def main():
    D = pd.concat([rows(y) for y in range(2021, 2027) if (A / str(y) / "espn_weekly_proj.csv").exists()])
    print("rows by season/bucket\n", D.groupby(["season", "bk"], observed=True).size().unstack())
    hist = D[D["season"] < 2026]
    out = []
    for test_y in sorted(D["season"].unique()):
        tr = hist[hist["season"] != test_y]
        te = D[D["season"] == test_y]
        for p in POS:
            trp, tep = tr[tr["position"] == p], te[te["position"] == p]
            pooled = trp[trp["week"].between(3, 14)]["r"].quantile([0.1, 0.9]).to_numpy()
            for bk, g in tep.groupby("bk", observed=True):
                tb = trp[trp["bk"] == bk]["r"]
                qb = tb.quantile([0.1, 0.9]).to_numpy()
                out.append(dict(season=test_y, pos=p, bk=bk, n=len(g), pooled=cover(g, *pooled), bucket=cover(g, *qb),
                                w_pooled=pooled[1] - pooled[0], w_bucket=qb[1] - qb[0],
                                sd=g["points"].sub(g["espn_proj"]).std()))
    O = pd.DataFrame(out)
    O.to_csv(ROOT / "backtest" / "range_by_week.csv", index=False)
    wavg = lambda g: pd.Series({c: np.average(g[c], weights=g["n"]) for c in ["pooled", "bucket", "w_pooled", "w_bucket", "sd"]} | {"n": g["n"].sum()})
    print("\nLOSO 2021-25 by bucket (coverage of 80% range; widths in ratio units; sd = pts miss sd)")
    print(O[O.season < 2026].groupby("bk", observed=True).apply(wavg).round(3))
    print("\nby position x bucket, LOSO 2021-25")
    print(O[O.season < 2026].groupby(["pos", "bk"], observed=True).apply(wavg)[["pooled", "bucket", "w_bucket", "n"]].round(3).unstack(0))
    print("\nper season, weeks 1-6 vs 7+ pooled coverage")
    O["half"] = np.where(O["bk"].isin(["1-2", "3-6"]), "1-6", "7+")
    print(O.groupby(["season", "half"]).apply(wavg)[["pooled", "bucket", "n"]].round(3).unstack())
    print("\n2026 by bucket")
    print(O[O.season == 2026].groupby("bk", observed=True).apply(wavg).round(3))


if __name__ == "__main__":
    main()
