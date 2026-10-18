"""Actual / ESPN-weekly-projection ratios per position, 2021-25 weeks 3-14, for the app's weekly ranges and odds.

The app scales ESPN's weekly number by a ratio spread (Compare, head-to-head, Live, win odds). Until now it used
history/backtest/multi/ratio_dist.json, measured against our own forecast and including games a player missed;
applied to ESPN's sharper number it covered ~93% inside its "80%" range (2021-25 and 2026 alike).
This spread is measured against ESPN's number, on rostered, non-IR players who played (the screens flag
players who are out separately), and covers 78-82% leave-one-season-out and 79% in 2026 weeks 1-5.

    python tools/ratio_dist_espn.py   ->   history/backtest/multi/ratio_dist_espn.json
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "history" / "actuals"
OUT = ROOT / "history" / "backtest" / "multi" / "ratio_dist_espn.json"
POS = ["QB", "RB", "WR", "TE"]


def rows(y: int) -> pd.DataFrame:
    pj = pd.read_csv(A / str(y) / "espn_weekly_proj.csv").drop_duplicates(["week", "espn_id"])
    pw = pd.read_csv(A / str(y) / "players_weekly.csv")
    pw = pw[(pw["kind"] == "actual") & pw["played"].astype(bool)].drop_duplicates(["espn_id", "week"])
    R = pd.read_csv(A / str(y) / "rosters_by_week.csv")
    R = R[(R["lineup_slot"] != "IR") & R["position"].isin(POS)]
    m = R.merge(pj, on=["week", "espn_id"]).merge(pw[["espn_id", "week", "points"]], on=["espn_id", "week"])
    return m[(m["espn_proj"] > 0) & m["week"].between(3, 14)]


if __name__ == "__main__":
    H = pd.concat([rows(y) for y in range(2021, 2026)])
    out = {p: [round(float(x), 4) for x in (H[H["position"] == p]["points"] / H[H["position"] == p]["espn_proj"]).sort_values()] for p in POS}
    OUT.write_text(json.dumps(out))
    print(OUT, {p: len(v) for p, v in out.items()})
