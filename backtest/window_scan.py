"""Is there a better "last X games" window?

(a) Across seasons: last X games of the previous season -> next season ppg, raw and after
    ESPN's preseason projection (fantasy-relevant players, 2023 excluded for the ESPN part).
(b) In season: at checkpoints W=4..13, the mean of the last X games played before W, and an
    exponentially weighted mean, added on top of v2 (LOSO per-position regression), vs v2.
Many windows are scanned, so expect the best one to look better than it really is.
  python3 backtest/window_scan.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backtest import multi_season as ms  # noqa: E402
from backtest import usage_signal as us  # noqa: E402
OUT = ROOT / "history" / "backtest" / "multi"
THR = {"QB": 14, "RB": 9, "WR": 9, "TE": 7}


def weekly():
    rows = []
    for y in range(2021, 2026):
        d = pd.read_csv(ROOT / "history" / "actuals" / str(y) / "players_weekly.csv")
        rows.append(d.assign(season=y))
    d = pd.concat(rows)
    proj = d[d["kind"] == "espn_season_proj"][["season", "espn_id", "points"]].rename(columns={"points": "proj"})
    a = d[(d["kind"] == "actual") & d["week"].between(1, 17) & (d["played"] == True)]  # noqa: E712
    return a.sort_values(["season", "espn_id", "week"]), proj


def resid(y, x):
    b = np.polyfit(x, y, 1)
    return y - np.polyval(b, x)


def part_a(a, proj):
    full = a.groupby(["season", "espn_id"])["points"].agg(full="mean", g="count").reset_index()
    nxt = full.rename(columns={"full": "next", "g": "gn"}).assign(season=lambda x: x["season"] - 1)
    pos = a.drop_duplicates("espn_id").set_index("espn_id")["position"]
    base = full.merge(nxt, on=["season", "espn_id"])
    base = base[(base["g"] >= 6) & (base["gn"] >= 6)]
    base["pos"] = base["espn_id"].map(pos)
    base = base.merge(proj.assign(season=proj["season"] - 1), on=["season", "espn_id"], how="left")
    out = []
    for X in (1, 2, 3, 4, 6, 8, 10, 12, 17):
        lastX = a.groupby(["season", "espn_id"]).tail(X).groupby(["season", "espn_id"])["points"].mean().rename("lastX")
        b = base.join(lastX, on=["season", "espn_id"])
        for p in THR:
            g = b[(b["pos"] == p) & (b["full"] >= THR[p])]
            g2 = g[(g["season"] != 2022) & (g["proj"] > 0)]  # season+1 == 2023 is the broken projection year
            out.append({"X": X, "pos": p, "raw": ms.spearman(g["lastX"], g["next"]),
                        "beyond ESPN": ms.spearman(resid(g2["lastX"], g2["proj"]), resid(g2["next"], g2["proj"]))})
    return pd.DataFrame(out)


def recent_features(a, rows):
    feats = []
    for (yr, W), keys in rows.groupby(["season", "week"])["espn_id"]:
        past = a[(a["season"] == yr) & (a["week"] < W)]
        f = pd.DataFrame(index=pd.Index(keys.unique(), name="espn_id"))
        for X in (1, 2, 3, 4, 6):
            f[f"last{X}"] = past.groupby("espn_id").tail(X).groupby("espn_id")["points"].mean()
        for hl in (2, 4):
            w = past.assign(w=0.5 ** ((W - 1 - past["week"]) / hl))
            f[f"ewm{hl}"] = (w["points"] * w["w"]).groupby(w["espn_id"]).sum() / w.groupby("espn_id")["w"].sum()
        feats.append(f.reset_index().assign(season=yr, week=W))
    return pd.concat(feats, ignore_index=True)


def part_b(a):
    d = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob("usage_rows_*.csv"))], ignore_index=True)
    d = d[d["week"] >= 4]
    d = d.merge(recent_features(a, d), on=["season", "week", "espn_id"], how="left")
    wins = ["last1", "last2", "last3", "last4", "last6", "ewm2", "ewm4"]
    for c in wins:
        d[c] = d[c].fillna(d["est_ppg"])
    res = []
    for c in ["none"] + wins:
        cols = ["est_ppg"] if c == "none" else ["est_ppg", c]
        d["pred"] = np.nan
        for held in sorted(d["season"].unique()):
            tr = d[(d["season"] != held) & (d["future_played"] >= 3) & (d["est_ppg"] > 0)]
            for pos in us.SKILL:
                A = tr[tr["position"] == pos]
                m = (d["season"] == held) & (d["position"] == pos)
                beta = us.ols(A[cols].to_numpy(float), A["future_ppg"].to_numpy(float), A["future_played"].to_numpy(float))
                d.loc[m, "pred"] = us.predict(beta, d.loc[m, cols].to_numpy(float))
        g0 = d[(d["future_played"] >= 3) & d["rostered"]]
        for pos in us.SKILL:
            g = g0[g0["position"] == pos]
            rho = np.mean([ms.spearman(x["pred"], x["future_ppg"]) for _, x in g.groupby(["season", "week"]) if len(x) > 5])
            res.append({"window": c, "pos": pos, "rho": rho})
    r = pd.DataFrame(res).pivot(index="window", columns="pos", values="rho")
    return r.sub(r.loc["none"]).round(3), r.loc["none"].round(3)


if __name__ == "__main__":
    a, proj = weekly()
    pa = part_a(a, proj)
    print("(a) previous season, last X games -> next season (rank corr)")
    print(pa.pivot(index="X", columns="pos", values="raw").round(2).to_string())
    print("beyond ESPN preseason projection:")
    print(pa.pivot(index="X", columns="pos", values="beyond ESPN").round(2).to_string())
    diff, base = part_b(a)
    print("\n(b) in season, gain in rank corr over v2 (v2 baseline:", base.to_dict(), ")")
    print(diff.to_string())
