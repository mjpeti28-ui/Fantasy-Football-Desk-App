"""season_sim: a clearly dominant team should be the favorite; probabilities are sane."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import season_sim as ss  # noqa: E402
import valuation as v  # noqa: E402


def test_dominant_team_is_favorite():
    teams = range(1, 11)
    ids, rows, rost = [], [], []
    pid = 1
    for t in teams:
        for pos, n in (("QB", 1), ("RB", 3), ("WR", 4), ("TE", 1), ("K", 1), ("DST", 1)):
            for _ in range(n):
                ppg = 20.0 if t == 1 else 10.0
                rows.append({"espn_id": pid, "position": pos, "est_ppg": ppg, "avail_now": 1.0, "avail": 1.0,
                             "bye": None, "from_week": 3, "name": f"p{pid}", "status": "active"})
                rost.append({"week": 3, "team_id": t, "espn_id": pid})
                pid += 1
    tbl = pd.DataFrame(rows).set_index("espn_id")
    s = v.Season(season=1999, weekly=pd.DataFrame(columns=["espn_id", "week", "points", "played"]), prior_total={},
                 rosters=pd.DataFrame(rost), team_weekly=pd.DataFrame(columns=["team_id", "week", "points"]),
                 settings={"lineup_slot_counts": {"QB": 1, "RB": 1, "RB/WR": 1, "WR": 1, "WR/TE": 1, "TE": 1, "FLEX": 1, "K": 1, "DST": 1},
                           "regular_season_weeks": 14, "playoff_teams": 8},
                 byes={}, position={}, name={})
    ratios = {p: np.array([0.5, 1.0, 1.5]) for p in v.POSITIONS}
    out = ss.simulate(s, 3, v.ForecastParams(), ratios, n_sims=200, tbl=tbl, replacement={p: 0.0 for p in v.POSITIONS})
    assert out.index[0] == 1 and out.at[1, "seed1"] > 0.95 and out.at[1, "title"] > 0.5
    assert abs(out["title"].sum() - 1.0) < 1e-9 and abs(out["playoffs"].sum() - 8.0) < 1e-9
