"""Rest-of-season odds for every team: #1 seed, top-2 seed, playoffs, title.

    python tools/season_odds.py                 # current 2026 rosters
    python tools/season_odds.py --sims 5000
    python tools/season_odds.py --give "Garrett Wilson" --with 1 --get "Isaiah Likely"   # odds with a trade applied

Backtest (backtest/sim_validation.py, 2021-2025): seeding odds have real skill by
mid-season (Brier skill vs base rates ~.3-.5 for playoffs/top-2/#1 seed at weeks 8-10);
title odds barely beat chance (skill ~.05) -- the playoffs are three single-week games.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env import load_local_env  # noqa: E402

load_local_env()

import pandas as pd  # noqa: E402

import season_sim as ss  # noqa: E402
import trade_value as tv  # noqa: E402

RATIOS = REPO_ROOT / "history" / "backtest" / "multi" / "ratio_dist.json"


def ratios():
    import numpy as np

    if RATIOS.exists():
        return {k: np.array(x) for k, x in json.loads(RATIOS.read_text()).items()}
    from backtest import multi_season as ms

    rows = pd.read_csv(REPO_ROOT / "history" / "backtest" / "multi" / "player_rows.csv")
    r = ss.fit_ratio_dist(ms.SEASONS, lambda yr: ms.fit_params(rows[rows["season"] != yr]))
    RATIOS.write_text(json.dumps({k: [round(float(x), 4) for x in v] for k, v in r.items()}))
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sims", type=int, default=3000)
    ap.add_argument("--team", type=int, default=3)
    ap.add_argument("--with", dest="other", type=int)
    ap.add_argument("--give", default="")
    ap.add_argument("--get", default="")
    a = ap.parse_args(argv)
    ctx = tv.load_context()
    r = ratios()
    names = {t: ctx.team_name(t).strip() for t in ctx.rosters}
    base = ss.simulate(ctx.season, ctx.week, ctx.params, r, rosters=ctx.rosters, n_sims=a.sims, seed=1,
                       tbl=ctx.tbl, replacement=ctx.replacement)
    base.index = [names[t] for t in base.index]
    pd.set_option("display.width", 200)
    print(f"week {ctx.week}, {a.sims} simulated seasons ({len(ctx.rosters)} teams)")
    if ctx.week < 6:
        print("EARLY SEASON: seeding/playoff odds only gain real skill around week 6 in backtests; read loosely.")
    base = base.sort_values("top2", ascending=False)
    print(base[["playoffs", "top2", "seed1", "reg_pts", "final", "title"]].round(3).to_string())
    print("title/final odds are near-lottery (backtest Brier skill ~.05); plan around playoff and top-2 odds.")
    if a.other:
        give = tv.resolve(ctx, a.team, [x.strip() for x in a.give.split(",") if x.strip()])
        get = tv.resolve(ctx, a.other, [x.strip() for x in a.get.split(",") if x.strip()])
        ros = dict(ctx.rosters)
        ros[a.team] = tv._after(ctx, ctx.rosters[a.team], give, get)
        ros[a.other] = tv._after(ctx, ctx.rosters[a.other], get, give)
        new = ss.simulate(ctx.season, ctx.week, ctx.params, r, rosters=ros, n_sims=a.sims, seed=1,
                          tbl=ctx.tbl, replacement=ctx.replacement)
        new.index = [names[t] for t in new.index]
        for t in (a.team, a.other):
            n = names[t]
            d2 = new.at[n, "top2"] - base.at[n, "top2"]
            dp = new.at[n, "playoffs"] - base.at[n, "playoffs"]
            noise = 2 * (0.25 / a.sims) ** 0.5  # rough 2-sigma for a difference in a ~50% probability

            def fmt(x):
                return "about 0" if abs(x) < noise else f"{x:+.3f}"
            print(f"{n}: top-2 seed {base.at[n, 'top2']:.3f} -> {new.at[n, 'top2']:.3f} ({fmt(d2)}), "
                  f"playoffs {base.at[n, 'playoffs']:.3f} -> {new.at[n, 'playoffs']:.3f} ({fmt(dp)}), "
                  f"reg pts {base.at[n, 'reg_pts']:.0f} -> {new.at[n, 'reg_pts']:.0f}; "
                  f"title {base.at[n, 'title']:.3f} -> {new.at[n, 'title']:.3f} (rough)")
        print(f"(changes under ~{noise:.3f} are simulation noise at {a.sims} sims; same random draws are used before and after)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
