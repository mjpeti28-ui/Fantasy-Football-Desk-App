"""Should the model recalibrate itself during the season from its own ledger?

Replays each past season (model_rows_*.csv: LOSO forecasts at checkpoints 3..13 with weekly
outcomes). At checkpoint W only outcomes before W are known. Updates, per position:
  scale : f = realized / expected points so far (from all earlier checkpoints' forecasts of
          weeks before W), shrunk: f = 1 + (raw - 1) * E / (E + lam)      -> est x f
  avail : g = games played / expected games so far, shrunk the same way  -> availability x g
Compared with static forecasts (lam = inf) on: rest-of-season total MAE and realized/predicted,
next-week points MAE, game-level Brier, and rank corr of ROS totals across positions (what
trades compare). Also a stress test: the held-out season's forecasts deliberately 10% too high
(or availability 10% too high) -- does updating recover?
  python3 backtest/inseason_recal.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backtest import multi_season as ms  # noqa: E402

OUT = ROOT / "history" / "backtest" / "multi"
SEASONS = [2021, 2022, 2023, 2024, 2025]
LAST = 17
LAMS = [0, 250, 1000, 4000, np.inf]
LAMS_A = [0, 25, 100, 400, np.inf]


def load():
    d = pd.concat([pd.read_csv(OUT / f"model_rows_{y}.csv") for y in SEASONS], ignore_index=True)
    d = d[(d["team_id"] > 0) | (d["est_ppg"] >= 3)].reset_index(drop=True)
    return d


def evaluate(d: pd.DataFrame, est_mult: float = 1.0, avail_mult: float = 1.0, parts=("scale", "avail")):
    recs = []
    for y in SEASONS:
        x = d[d["season"] == y].copy()
        x["est"] = x["est_ppg"] * est_mult
        for w in range(3, LAST + 1):
            if f"xa_w{w}" in x:
                x[f"xa_w{w}"] = (x[f"xa_w{w}"] * avail_mult).clip(upper=1)
        for W in range(4, 14):
            prior = x[x["W"] < W]
            cur = x[x["W"] == W]
            if cur.empty:
                continue
            # evidence so far, per position
            ev = {}
            for pos, g in prior.groupby("position"):
                E = R = XG = PG = 0.0
                for w in range(3, W):
                    m = g["W"] <= w
                    if not m.any() or f"xa_w{w}" not in g:
                        continue
                    xa = g.loc[m, f"xa_w{w}"].fillna(0)
                    E += float((g.loc[m, "est"] * xa).sum())
                    R += float(g.loc[m, f"pts_w{w}"].fillna(0).sum())
                    XG += float(xa.sum())
                    PG += float(g.loc[m, f"play_w{w}"].fillna(0).sum())
                ev[pos] = (E, R, XG, PG)
            fut = [w for w in range(W, LAST + 1)]
            real_tot = cur[[f"pts_w{w}" for w in fut]].sum(axis=1)
            nxt = cur[f"pts_w{W}"]
            played_nxt = cur[f"play_w{W}"] == 1
            for lam, lam_a in zip(LAMS, LAMS_A):
                f = cur["position"].map(lambda p: _shrink(ev.get(p, (0, 0, 0, 0))[1], ev.get(p, (0, 0, 0, 0))[0], lam)) if "scale" in parts else pd.Series(1.0, index=cur.index)
                g = cur["position"].map(lambda p: _shrink(ev.get(p, (0, 0, 0, 0))[3], ev.get(p, (0, 0, 0, 0))[2], lam_a)) if "avail" in parts else pd.Series(1.0, index=cur.index)
                xa = np.column_stack([(cur[f"xa_w{w}"] * g).clip(upper=1) for w in fut])
                pred_tot = (cur["est"] * f).to_numpy()[:, None] * xa
                pred_tot = pred_tot.sum(axis=1)
                plays = cur[[f"play_w{w}" for w in fut]].to_numpy()
                recs.append({
                    "season": y, "W": W, "lam": lam,
                    "ros_abs_err": float(np.abs(pred_tot - real_tot).mean()),
                    "ros_real": float(real_tot.sum()), "ros_pred": float(pred_tot.sum()),
                    "next_abs_err": float(np.abs(cur.loc[played_nxt, "est"] * f[played_nxt] - nxt[played_nxt]).mean()),
                    "brier": float(np.nanmean((xa - plays) ** 2)),
                    "cross_pos_rank": ms.spearman(pred_tot, real_tot),
                    "rost_rank": ms.spearman(pred_tot[(cur["team_id"] > 0).to_numpy()], real_tot[(cur["team_id"] > 0).to_numpy()]),
                })
    return pd.DataFrame(recs)


def _shrink(num, den, lam):
    if den <= 0 or not np.isfinite(lam) and lam == np.inf:
        return 1.0
    raw = num / den
    return 1.0 + (raw - 1.0) * den / (den + lam)


def summarize(r: pd.DataFrame, label: str):
    g = r.groupby("lam")
    t = pd.DataFrame({"ros_MAE": g["ros_abs_err"].mean(), "real/pred": g["ros_real"].sum() / g["ros_pred"].sum(),
                      "next_MAE": g["next_abs_err"].mean(), "brier": g["brier"].mean(),
                      "cross_pos_rank": g["cross_pos_rank"].mean(), "rostered_rank": g["rost_rank"].mean()})
    print(f"\n== {label} ==")
    print(t.round(4).to_string())
    base = r[r["lam"] == np.inf].set_index(["season", "W"])
    rng = np.random.default_rng(0)
    for lam in LAMS[:-1]:
        o = r[r["lam"] == lam].set_index(["season", "W"])
        dd = (o["ros_abs_err"] - base["ros_abs_err"]).to_numpy()
        bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(2000)]
        seas = (o["ros_abs_err"] - base["ros_abs_err"]).groupby(level=0).mean().round(2).to_dict()
        print(f"  lam={lam}: ROS MAE change {dd.mean():+.3f} [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}] by season {seas}")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    d = load()
    summarize(evaluate(d), "actual seasons (no injected drift)")
    summarize(evaluate(d, est_mult=1.10), "stress: forecasts 10% too high")
    summarize(evaluate(d, est_mult=0.90), "stress: forecasts 10% too low")
    summarize(evaluate(d, avail_mult=1.10), "stress: availability 10% too high")
