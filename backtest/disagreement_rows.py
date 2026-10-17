"""Where Desk's weekly number disagrees with ESPN's: one row per (season, week, player) with
ESPN's published weekly projection, Desk's v2 week forecast (est_ppg x avail_now, the live
blend: ESPN weekly + our shrunk ppg + ROS ECR), the no-ESPN piece, and what happened.

Params are leave-one-season-out (fit_all on the other seasons), 2021-2025. Players kept:
rostered (non-IR) or ESPN-projected > 0 that week; bye weeks dropped.
  python3 backtest/disagreement_rows.py   -> history/backtest/multi/disagreement_rows.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import valuation as v  # noqa: E402
from backtest import multi_season as ms  # noqa: E402

OUT = ROOT / "history" / "backtest" / "multi"

if __name__ == "__main__":
    rows = pd.read_csv(OUT / "player_rows.csv")
    parts = []
    for yr in ms.SEASONS:
        p = ms.fit_all(rows, [y for y in ms.SEASONS if y != yr])
        s = v.load_season(yr)
        wk = s.weekly
        for W in range(1, v.FANTASY_LAST_WEEK + 1):
            if W > int(s.rosters["week"].max()) or W > int(wk["week"].max()):
                break
            if not (s.espn_weekly["week"] == W).any():
                continue
            t = v.player_table(s, W, p)
            e = s.espn_weekly[s.espn_weekly["week"] == W].set_index("espn_id")["espn_proj"]
            ep = s.espn_weekly[s.espn_weekly["week"] == W - 1].set_index("espn_id")["espn_proj"]
            g = wk[wk["week"] == W].groupby("espn_id").agg(points=("points", "sum"), played=("played", "max"))
            r = s.roster_at(W).set_index("espn_id")
            t["espn"] = e.reindex(t.index).fillna(0.0)
            t["espn_prev"] = ep.reindex(t.index)
            t["team_id"] = r["team_id"].reindex(t.index).fillna(0).astype(int)
            t["slot"] = r["lineup_slot"].reindex(t.index).fillna("FA")
            t = t[(t["bye"] != W) & (t["slot"] != "IR") & ((t["team_id"] > 0) | (t["espn"] > 0))]
            t["desk"] = t["est_ppg"] * t["avail_now"]
            t["played"] = g["played"].reindex(t.index).fillna(False).astype(bool)
            t["actual"] = g["points"].reindex(t.index).where(t["played"], 0.0).fillna(0.0)
            t["season"], t["week"] = yr, W
            parts.append(t.reset_index()[["season", "week", "espn_id", "name", "position", "team_id", "slot", "status",
                                          "n", "espn", "espn_prev", "ours_ppg", "est_ppg", "ecr_rank", "avail_now",
                                          "desk", "played", "actual"]])
        print(yr, "done", flush=True)
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(OUT / "disagreement_rows.csv", index=False)
    print(len(df), "rows")
