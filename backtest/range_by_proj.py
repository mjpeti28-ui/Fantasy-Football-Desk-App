"""Should the weekly range width depend on projection size? (follow-up to range_by_week.py)

The app's 80% range is ESPN's weekly projection x the 10th/90th pct actual/ESPN ratio, one spread per position,
fit on all rostered non-IR players who played. For started players that covers 82-85%, too wide. Here: fit the
ratio spread per position x ESPN-projection tier on 2021-25 (leave-one-season-out), compare coverage and pinball
loss with the pooled spread, overall and for starters; then the 2021-25 fit on 2026.

    python backtest/range_by_proj.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "history" / "actuals"
POS = ["QB", "RB", "WR", "TE"]
QS = [0.1, 0.9]


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
    m["start"] = m["lineup_slot"] != "BE"
    return m


def pinball(y, q, tau):
    d = y - q
    return np.mean(np.maximum(tau * d, (tau - 1) * d))


def main(n_tiers=3):
    D = pd.concat([rows(y) for y in range(2021, 2027)])
    hist = D[D["season"] < 2026]
    out = []
    for ty in range(2021, 2027):
        tr = hist[(hist["season"] != ty) & hist["week"].between(3, 14)]
        te = D[D["season"] == ty].copy()
        for p in POS:
            trp = tr[tr["position"] == p]
            cuts = np.quantile(trp["espn_proj"], np.linspace(0, 1, n_tiers + 1)[1:-1])
            pooled = trp["r"].quantile(QS).to_numpy()
            tq = {t: g["r"].quantile(QS).to_numpy() for t, g in trp.groupby(np.searchsorted(cuts, trp["espn_proj"]))}
            g = te[te["position"] == p].copy()
            g["tier"] = np.searchsorted(cuts, g["espn_proj"])
            for name, Q in [("pooled", lambda t: pooled), ("tier", lambda t: tq[t])]:
                lo = np.array([Q(t)[0] for t in g["tier"]]) * g["espn_proj"]
                hi = np.array([Q(t)[1] for t in g["tier"]]) * g["espn_proj"]
                g[f"in_{name}"] = (g["points"] >= lo) & (g["points"] <= hi)
                g[f"w_{name}"] = hi - lo
                g[f"pb_{name}"] = [pinball(np.array([y]), np.array([l]), .1) + pinball(np.array([y]), np.array([h]), .9)
                                    for y, l, h in zip(g["points"], lo, hi)]
            out.append(g)
    O = pd.concat(out)
    cols = ["in_pooled", "in_tier", "w_pooled", "w_tier", "pb_pooled", "pb_tier"]
    O["set"] = np.where(O["season"] == 2026, "2026", "LOSO 21-25")
    print(f"tiers={n_tiers}  (in_ = coverage of 80% range, w_ = width in pts, pb_ = pinball loss, lower better)")
    for who, M in [("all rostered", O), ("starters", O[O["start"]]), ("bench", O[~O["start"]])]:
        print(f"\n{who}\n", M.groupby("set")[cols].mean().round(3).assign(n=M.groupby("set").size()))
    S = O[O["start"] & (O["season"] < 2026)]
    print("\nstarters LOSO by position x tier\n", S.groupby(["position", "tier"])[["in_pooled", "in_tier", "w_pooled", "w_tier"]].mean().round(3).assign(n=S.groupby(["position", "tier"]).size()))
    print("\nper season, starters\n", O[O["start"]].groupby("season")[["in_pooled", "in_tier", "pb_pooled", "pb_tier"]].mean().round(3))
    return O


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
