"""Summarize backtest/currency_validation.py outputs across seasons."""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "history" / "backtest" / "multi"


def rho(a, b):
    a, b = pd.Series(list(a)), pd.Series(list(b))
    ok = a.notna() & b.notna()
    return float(a[ok].rank().corr(b[ok].rank())) if ok.sum() > 3 else np.nan


def main():
    T = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(str(OUT / "currency_teams_*.csv"))))
    R = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(str(OUT / "currency_trades_*.csv"))))
    lines = ["== Team outcomes: rank corr with actual points, checkpoint -> end of regular season (12 teams per season-week)"]
    z = lambda s: (s - s.mean()) / (s.std(ddof=0) or 1)
    recs = []
    for (yr, w), g in T.groupby(["season", "week"]):
        recs.append({"season": yr, "week": w,
                     "new: expected lineup pts": rho(g["pred_points"], g["actual_points"]),
                     "old: combined z": rho(g["old_combined"], g["actual_points"]),
                     "ppg so far": rho(g["ppg_so_far"], g["actual_points"]),
                     "new + ppg so far": rho(z(g["pred_points"]) + z(g["ppg_so_far"]), g["actual_points"])})
    X = pd.DataFrame(recs)
    lines.append(X.groupby("season").mean().drop(columns="week").round(3).to_string())
    lines.append("all: " + ", ".join(f"{c} {X[c].mean():.3f}" for c in X.columns if c not in ("season", "week")))
    lines.append(X.groupby("week").mean().drop(columns="season").round(3).to_string())

    lines.append("")
    lines.append("== Hypothetical trades: predicted vs realized change in a team's lineup points (weeks W-17, hindsight-optimal lineups)")
    long = pd.concat([
        R[["season", "week", "shape", "pred_a", "real_a"] + (["old_a"] if "old_a" in R else [])].rename(columns={"pred_a": "pred", "real_a": "real", "old_a": "old"}),
        R[["season", "week", "shape", "pred_b", "real_b"] + (["old_b"] if "old_b" in R else [])].rename(columns={"pred_b": "pred", "real_b": "real", "old_b": "old"}),
    ])
    lines.append(f"n = {len(long)} team-sides from {len(R)} trades; old model scored on a subset of {long['old'].notna().sum()}")
    by = long.groupby("season").apply(lambda g: pd.Series({"new rho": rho(g["pred"], g["real"])}))
    lines.append(by.round(3).to_string())
    sub = long.dropna(subset=["old"])
    lines.append(f"same subset -> new rho {rho(sub['pred'], sub['real']):.3f} vs old (combined z) rho {rho(sub['old'], sub['real']):.3f}")
    lines.append(f"sign accuracy on subset: new {(np.sign(sub['pred']) == np.sign(sub['real'])).mean():.3f}, old {(np.sign(sub['old']) == np.sign(sub['real'])).mean():.3f}")
    for shape, g in long.groupby("shape"):
        lines.append(f"{shape}: rho {rho(g['pred'], g['real']):.3f}, n={len(g)}")
    slope = float(np.polyfit(long["pred"], long["real"], 1)[0])
    lines.append(f"calibration: realized change ~= {slope:.2f} x predicted change")
    long["bin"] = pd.cut(long["pred"].abs(), [0, 5, 10, 20, 40, 80, 1e9])
    tab = long.groupby("bin").apply(lambda g: pd.Series({
        "n": len(g), "sign_accuracy": (np.sign(g["pred"]) == np.sign(g["real"])).mean(),
        "median_pred": g["pred"].abs().median(), "median_realized_in_pred_direction": (g["real"] * np.sign(g["pred"])).median()}))
    lines.append(tab.round(3).to_string())
    text = "\n".join(lines)
    print(text)
    (OUT / "currency_summary.txt").write_text(text)


if __name__ == "__main__":
    main()
