"""Does expert disagreement (FantasyPros ECR sd) measure forecast uncertainty?

A. ROS: rank corr of relative expert spread (sd/ecr) with the forecast's miss, after
   controlling for forecast level (bigger forecasts miss by more points) -- absolute and
   relative error, per position.
B. Weekly: same question for next-week points vs ESPN's projection.
C. Distribution test (weekly): 10/25/50/75/90% quantiles of actual/projection, pooled by
   position (baseline) vs split by spread tercile (LOSO). Metrics: 80% interval coverage
   per tercile, pinball loss.
  python3 backtest/ecr_uncertainty.py
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
QS = [0.1, 0.25, 0.5, 0.75, 0.9]


def resid(y, x):
    b = np.polyfit(x, y, 2)
    return y - np.polyval(b, x)


def partial(g, level, err, spread):
    return ms.spearman(resid(g[spread], g[level]), resid(g[err], g[level]))


def boot(vals, n=1000):
    vals = np.asarray(vals)
    rng = np.random.default_rng(0)
    bs = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(n)]
    return vals.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def pinball(y, qpred, q):
    d = y - qpred
    return np.mean(np.maximum(q * d, (q - 1) * d))


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    R = pd.read_csv(OUT / "ecr_ros.csv")
    R["spread"] = R["sd"] / R["ecr"].clip(lower=1)
    R["abs_err"] = (R["p_both"] - R["future_ppg"]).abs()
    R["rel_err"] = R["abs_err"] / R["p_both"].clip(lower=1)
    print("A. ROS: partial rank corr of expert spread with forecast miss (controls for forecast level)")
    for err in ("abs_err", "rel_err"):
        per = [partial(g, "p_both", err, "spread") for _, g in R.groupby(["season", "week", "position"]) if len(g) > 10]
        m, lo, hi = boot(per)
        bypos = {p: round(np.mean([partial(g, "p_both", err, "spread") for _, g in R[R["position"] == p].groupby(["season", "week"]) if len(g) > 10]), 3) for p in ("QB", "RB", "WR", "TE")}
        print(f"  {err}: {m:+.3f} [{lo:+.3f},{hi:+.3f}]  by position {bypos}")
    # does spread just proxy for rank (mid-ranked players have both)? within-rank-band check
    R["band"] = pd.cut(R["ecr"], [0, 12, 24, 40, 999])
    print("  mean abs_err by rank band x spread tercile:")
    R["sp3"] = R.groupby(["position", "band"], observed=True)["spread"].transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low", "mid", "high"]))
    print(R.groupby(["band", "sp3"], observed=True)["abs_err"].mean().unstack().round(2).to_string())

    W = pd.read_csv(OUT / "ecr_weekly.csv")
    W = W[W["espn_proj"] > 1].copy()
    W["spread"] = W["sd"] / W["ecr"].clip(lower=1)
    W["abs_err"] = (W["points"] - W["espn_proj"]).abs()
    per = [partial(g, "espn_proj", "abs_err", "spread") for _, g in W.groupby(["season", "week", "position"]) if len(g) > 10]
    m, lo, hi = boot(per)
    print(f"\nB. weekly: partial rank corr spread vs |points - ESPN proj|: {m:+.3f} [{lo:+.3f},{hi:+.3f}]")

    print("\nC. weekly distribution: pooled quantiles vs spread-tercile quantiles (LOSO)")
    W["ratio"] = W["points"] / W["espn_proj"]
    W["t3"] = W.groupby(["season", "week", "position"])["spread"].transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False) if len(s) >= 3 else 1)
    rows = []
    for held in sorted(W["season"].unique()):
        tr, te = W[W["season"] != held], W[W["season"] == held].copy()
        for pos in te["position"].unique():
            a, b = tr[tr["position"] == pos], te["position"] == pos
            pooled = a["ratio"].quantile(QS).to_numpy()
            for t in (0, 1, 2):
                qa = a[a["t3"] == t]["ratio"].quantile(QS).to_numpy()
                m = b & (te["t3"] == t)
                for j, q in enumerate(QS):
                    te.loc[m, f"base_{q}"] = pooled[j] * te.loc[m, "espn_proj"]
                    te.loc[m, f"sd_{q}"] = qa[j] * te.loc[m, "espn_proj"]
        rows.append(te)
    T = pd.concat(rows)
    out = []
    for t in (0, 1, 2):
        g = T[T["t3"] == t]
        rec = {"spread tercile": ["low", "mid", "high"][t], "n": len(g)}
        for k in ("base", "sd"):
            rec[f"{k} 80% coverage"] = float(((g["points"] >= g[f"{k}_0.1"]) & (g["points"] <= g[f"{k}_0.9"])).mean())
            rec[f"{k} width"] = float((g[f"{k}_0.9"] - g[f"{k}_0.1"]).mean())
        out.append(rec)
    print(pd.DataFrame(out).round(3).to_string(index=False))
    for k in ("base", "sd"):
        pl = np.mean([pinball(T["points"].to_numpy(), T[f"{k}_{q}"].to_numpy(), q) for q in QS])
        print(f"  mean pinball loss {k}: {pl:.4f}")
    diffs = []
    for (s, w), g in T.groupby(["season", "week"]):
        diffs.append(np.mean([pinball(g["points"].to_numpy(), g[f"sd_{q}"].to_numpy(), q) - pinball(g["points"].to_numpy(), g[f"base_{q}"].to_numpy(), q) for q in QS]))
    m, lo, hi = boot(diffs)
    print(f"  pinball sd - base (negative = better) {m:+.4f} [{lo:+.4f},{hi:+.4f}]")
