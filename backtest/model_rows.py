"""Current-model forecasts at every checkpoint of every past season (leave-one-season-out),
with what happened afterward week by week. Reused by the option-value and recalibration
experiments.

One row per (season, checkpoint W in 3..13, player with a forecast at QB/RB/WR/TE/K/DST):
  est_ppg, avail_now, av_a, av_b, avail, bye, status, espn_now, ecr_rank, ecr_sd, depth,
  team_id (fantasy team at W, 0 = free agent), and pts_w{w} / play_w{w} for w = W..17.

  python3 backtest/model_rows.py 2021 [2022 ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import avail_model as am  # noqa: E402
import nflverse_data as nd  # noqa: E402
import valuation as v  # noqa: E402
from backtest import multi_season as ms  # noqa: E402

OUT = ROOT / "history" / "backtest" / "multi"
CHECKS = range(3, 14)


def build(season: int) -> pd.DataFrame:
    rows = pd.read_csv(OUT / "player_rows.csv")
    p = ms.fit_all(rows, [y for y in ms.SEASONS if y != season])
    s = v.load_season(season)
    E = nd.ecr_table().dropna(subset=["espn_id"])
    E = E[(E["season"] == season) & E["page_type"].isin(["redraft-qb", "redraft-rb", "redraft-wr", "redraft-te"])]
    E["espn_id"] = E["espn_id"].astype("int64")
    cache = am._season_cache(season)
    dmin = cache[1] if cache else {}
    wk = s.weekly[s.weekly["week"].between(1, v.FANTASY_LAST_WEEK)]
    pts = wk.groupby(["espn_id", "week"])["points"].sum()
    played = set(zip(wk.loc[wk["played"], "espn_id"], wk.loc[wk["played"], "week"]))
    out = []
    for W in CHECKS:
        r = s.roster_at(W)
        owner = dict(zip(r["espn_id"], r["team_id"]))
        # last Friday's ECR (what the model itself uses) -- also keep sd
        e = E[E["week"] <= W - 1]
        if not e.empty:
            e = e[e["week"] == e["week"].max()].groupby("espn_id")[["ecr", "sd"]].min()
        t = v.player_table(s, W, p, rostered=set(r["espn_id"]))
        t = t[(t["est_ppg"] > 0)]
        keep = t.index.isin(list(owner)) | (t["est_ppg"] >= 2.0)
        t = t[keep].copy()
        t["season"], t["W"] = season, W
        t["team_id"] = [int(owner.get(i, 0)) for i in t.index]
        t["ecr_sd"] = t.index.map(e["sd"]) if len(e) else np.nan
        t["depth"] = [dmin.get((int(i), W), dmin.get((int(i), W - 1), np.nan)) for i in t.index]
        for w in range(W, v.FANTASY_LAST_WEEK + 1):
            t[f"pts_w{w}"] = [float(pts.get((i, w), 0.0)) for i in t.index]
            t[f"play_w{w}"] = [float((i, w) in played) for i in t.index]
            t[f"xa_w{w}"] = v._avail_for_week(t, w)
        cols = ["season", "W", "name", "position", "est_ppg", "avail_now", "avail", "av_a", "av_b", "bye", "status",
                "espn_now", "ecr_rank", "ecr_sd", "depth", "team_id"]
        cols = [c for c in cols if c in t.columns] + [c for c in t.columns if c.startswith(("pts_w", "play_w", "xa_w"))]
        out.append(t[cols].reset_index())
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    for a in sys.argv[1:]:
        d = build(int(a))
        d.to_csv(OUT / f"model_rows_{a}.csv", index=False)
        print(a, len(d), "rows")
