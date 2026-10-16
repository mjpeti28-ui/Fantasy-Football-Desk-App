"""Are the season simulator's seed/title probabilities calibrated? LOSO, 2021-2025."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import season_sim as ss  # noqa: E402
import valuation as v  # noqa: E402
from backtest import multi_season as ms  # noqa: E402

OUT = ROOT / "history" / "backtest" / "multi"


def run(season: int, checkpoints=(1, 4, 6, 8, 10, 12), n_sims=2000, levels: bool = False) -> pd.DataFrame:
    rows = pd.read_csv(OUT / "player_rows.csv")
    train = [y for y in ms.SEASONS if y != season]
    p = ms.fit_all(rows, train)
    ratios = ss.fit_ratio_dist(train, lambda yr: ms.fit_params(rows[(rows["season"] != yr) & (rows["season"] != season)]))
    fs = pd.read_csv(v.HIST / "final_standings.csv")
    fs = fs[fs["season"] == season].set_index("team_id")
    s = v.load_season(season)
    out = []
    for W in checkpoints:
        try:
            sim = ss.simulate(s, W, p, ratios, n_sims=n_sims, seed=season * 100 + W, levels=levels)
        except ValueError:  # no usable forecasts (2023 week 1)
            continue
        for t, r in sim.iterrows():
            out.append({"season": season, "week": W, "team_id": t, **r.to_dict(),
                        "won_title": float(fs.at[t, "final"] == 1), "made_playoffs": float(fs.at[t, "seed"] <= 8),
                        "got_top2": float(fs.at[t, "seed"] <= 2), "got_seed1": float(fs.at[t, "seed"] == 1)})
    return pd.DataFrame(out)


if __name__ == "__main__":
    args = sys.argv[1:]
    lv = "--levels" in args
    n = 1000 if "--fast" in args else 2000
    tag = ("_levels" if lv else "_base") if "--fast" in args else ""
    for yr in [int(a) for a in args if a.isdigit()]:
        df = run(yr, n_sims=n, levels=lv)
        df.to_csv(OUT / f"sim_validation_{yr}{tag}.csv", index=False)
        print(yr, len(df), tag)
