"""Backfill the prediction ledger for a past season and score it -- tests ledger.py and gives
a baseline for what weekly scores should look like.
  python3 backtest/ledger_backfill.py 2025 [--params path] [--tag _x]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import ledger  # noqa: E402
import valuation as v  # noqa: E402
from backtest import multi_season as ms  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("season", type=int)
    ap.add_argument("--params", default="")
    ap.add_argument("--tag", default="_backfill")
    ap.add_argument("--weeks", default="3-14")
    a = ap.parse_args()
    rows = pd.read_csv(ROOT / "history" / "backtest" / "multi" / "player_rows.csv")
    p = v.ForecastParams.load(Path(a.params)) if a.params else ms.fit_all(rows, [y for y in ms.SEASONS if y != a.season])
    s = v.load_season(a.season)
    lo, hi = (int(x) for x in a.weeks.split("-"))
    for W in range(lo, hi + 1):
        r = s.roster_at(W)
        tbl = v.player_table(s, W, p, rostered=set(r["espn_id"]))
        ros = {int(t): [int(i) for i in g["espn_id"] if int(i) in tbl.index] for t, g in r.groupby("team_id")}
        repl = v.waiver_replacement(tbl, [i for x in ros.values() for i in x], scale=p.waiver_scale)
        tvals = {t: v.team_value(ids, tbl, s.slots, range(W, v.FANTASY_LAST_WEEK + 1), repl, sims=60, seed=3) for t, ids in ros.items()}
        ledger.write(a.season, W, tbl, ros, tvals, tag=a.tag)
        ledger.write_sources(a.season, W, tbl, fp_file=Path("/nonexistent"), tag=a.tag)
    res = ledger.score(a.season, tag=a.tag)
    src = ledger.score_sources(a.season, tag=a.tag)
    if len(src):
        print(src.groupby("source")[["rank_corr", "mae", "bias"]].mean().round(3).sort_values("rank_corr", ascending=False).to_string())
    pd.set_option("display.width", 220)
    print(res.round(3).to_string(index=False))
