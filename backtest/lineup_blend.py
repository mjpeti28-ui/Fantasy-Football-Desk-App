"""Start/sit: does blending weekly ECR and Vegas into ESPN's projection set better lineups?

Uses vegas_rows.csv (rostered players with an ESPN projection > 0 and game lines; LOSO
regressions from vegas_signal.py). For every team-week, lineups chosen by each rule, scored on
actual points. Rules: raw ESPN, LOSO-calibrated ESPN, + weekly ECR, + ECR + Vegas, and a
tiebreaker (blend only decides when two options are within 1 projected point).
Metrics: points per team-week vs raw ESPN (paired, 95% CI), per season, share of lineups that
change, and gain in the weeks where the lineup actually changed.
  python3 backtest/lineup_blend.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import valuation as v  # noqa: E402
from backtest.start_sit import choose  # noqa: E402

OUT = ROOT / "history" / "backtest" / "multi"

if __name__ == "__main__":
    x = pd.read_csv(OUT / "vegas_rows.csv")
    x["position"] = x["position"].replace({"DST": "DST"})
    # tiebreak: ESPN, nudged by the blend's opinion, capped at +/- 1 point
    x["tiebreak"] = x["espn_proj"] + np.clip(x["p_espn+ecr+vegas"] - x["p_espn"], -1, 1)
    rules = {"raw ESPN": "espn_proj", "calibrated ESPN": "p_espn", "+ECR": "p_espn+ecr",
             "+ECR+Vegas": "p_espn+ecr+vegas", "tiebreak": "tiebreak"}
    slots = {y: v.load_season(y).slots for y in x["season"].unique()}
    recs = []
    for (y, w, t), g in x.groupby(["season", "week", "team_id"]):
        pts = dict(zip(g["espn_id"], g["points"]))
        rec = {"season": y, "week": w, "team": t}
        picks = {}
        for name, col in rules.items():
            cands = [(int(i), p if p != "D/ST" else "DST", float(s)) for i, p, s in zip(g["espn_id"], g["position"], g[col])]
            pk = choose(cands, slots[y])
            picks[name] = frozenset(pk)
            rec[name] = sum(pts.get(i, 0.0) for i in pk)
        for name in rules:
            rec["chg_" + name] = picks[name] != picks["raw ESPN"]
        recs.append(rec)
    d = pd.DataFrame(recs)
    rng = np.random.default_rng(0)
    print(f"team-weeks: {len(d)}")
    for name in list(rules)[1:]:
        diff = (d[name] - d["raw ESPN"]).to_numpy()
        bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
        ch = d["chg_" + name]
        by = (d[name] - d["raw ESPN"]).groupby(d["season"]).mean().round(2).to_dict()
        print(f"{name:16s} {diff.mean():+.3f} pts/team-week [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}] "
              f"| lineups changed {ch.mean():.1%}, gain when changed {diff[ch.to_numpy()].mean():+.2f} | by season {by}")
    d.to_csv(OUT / "lineup_blend.csv", index=False)
