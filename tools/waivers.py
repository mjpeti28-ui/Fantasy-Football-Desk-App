"""Waiver-wire suggestions for the live season.

    python tools/waivers.py            # team 3, top candidates per position
    python tools/waivers.py --team 3 --top 8

Shortlist = the top v2 rest-of-season forecasts per position; ESPN's projection for the
coming week is shown as a column (use it when you need someone to start *this* week).
Backtest (2021-2025 waiver replay, real pickups, current model): v2 alone 83.6 rest-of-
season points per pickup vs 82.4 for the old z(ESPN week) + z(v2) blend (difference not
significant, +1.2 [-1.3, +3.7]) and 79.0 for what this league's managers actually added.
Managers see news (role changes) the model can't, so treat this as a shortlist to
research, not a verdict.

For each shortlisted player it also shows your team's expected lineup-points change
(weeks now-17) if you add him and drop whichever player costs you least.
Caveat: the model assumes a waiver streamer is always available at every position,
so it can undervalue a thin position (e.g. your only TE) relative to reality.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env import load_local_env  # noqa: E402

load_local_env()

import pandas as pd  # noqa: E402

import trade_value as tv  # noqa: E402
import valuation as v  # noqa: E402


OUT_TAGS = {"INJURY_RESERVE", "OUT", "SUSPENSION", "PUP"}


def fa_injury_tags(max_age_days: float = 2.0) -> dict:
    """ESPN's current injury tag for free agents (valuation only applies live tags to rostered
    players). Uses the latest snapshot's espn_live.csv if fresh, else pulls it now."""
    import time
    snaps = sorted((REPO_ROOT / "history" / "snapshots" / "2026").glob("*/espn_live.csv"))
    if snaps and time.time() - snaps[-1].stat().st_mtime < max_age_days * 86400:
        live = pd.read_csv(snaps[-1])
    else:
        import tempfile

        sys.path.insert(0, str(REPO_ROOT / "tools"))
        import weekly_run
        f = Path(tempfile.gettempdir()) / "espn_live_waivers.csv"
        weekly_run.espn_live_state(f)
        live = pd.read_csv(f)
    return dict(zip(live["espn_id"].astype(int), live["injury_status"].fillna("")))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team", type=int, default=3)
    ap.add_argument("--top", type=int, default=6)
    a = ap.parse_args(argv)
    ctx = tv.load_context()
    rostered = {i for ids in ctx.rosters.values() for i in ids}
    t = ctx.tbl
    pool = t[~t.index.isin(rostered) & t["position"].isin(["QB", "RB", "WR", "TE", "K", "DST"])].copy()
    pool["v2_ros"] = v.expected_ros_points(pool, ctx.week, v.FANTASY_LAST_WEEK)
    pool["espn_wk"] = pool["espn_now"].fillna(0)
    try:
        tags = fa_injury_tags()
    except Exception as e:  # noqa: BLE001
        tags = {}
        print(f"(couldn't load ESPN injury tags for free agents: {e})")
    pool["tag"] = pool.index.map(lambda i: tags.get(int(i), ""))
    hurt = pool[pool["tag"].isin(OUT_TAGS) & (pool["v2_ros"] > pool.groupby("position")["v2_ros"].transform("median"))]
    pool = pool[~pool["tag"].isin(OUT_TAGS)]
    out = []
    weeks = range(ctx.week, v.FANTASY_LAST_WEEK + 1)
    base = v.team_value(ctx.rosters[a.team], t, ctx.season.slots, weeks, ctx.replacement, sims=200, seed=3)
    for pos, g in pool.groupby("position"):
        g = g.nlargest(a.top, "v2_ros")
        for pid, r in g.iterrows():
            # try every possible drop and keep the one that hurts least
            best = None
            for d in ctx.rosters[a.team]:
                after = [i for i in ctx.rosters[a.team] if i != d] + [pid]
                val = v.team_value(after, t, ctx.season.slots, weeks, ctx.replacement, sims=200, seed=3)
                if best is None or val > best[0]:
                    best = (val, d)
            gain = best[0] - base
            drop = [ctx.tbl.at[best[1], "name"]]
            out.append({"pos": pos, "player": r["name"], "espn_this_week": round(r["espn_wk"], 1),
                        "v2_ros_pts": round(r["v2_ros"], 0), "your_gain_pts": round(gain, 1), "drop": ", ".join(drop),
                        "why": tv.why(ctx, pid)})
    pd.set_option("display.width", 200)
    df = pd.DataFrame(out)
    if len(hurt):
        print("left off (ESPN tags them IR/OUT/suspended; the model can't see that for free agents): "
              + ", ".join(f"{n} ({t})" for n, t in zip(hurt["name"], hurt["tag"])))
    print(f"week {ctx.week}; gains are expected lineup points weeks {ctx.week}-17 (edges under ~10 are noise)")
    print(f"why = ours / ESPN week / experts rank -> blended ppg (typical miss +/-{tv.TYPICAL_MISS_PPG:.0f} ppg), chance he plays")
    for pos in ["QB", "RB", "WR", "TE", "K", "DST"]:
        sub = df[df["pos"] == pos].sort_values("your_gain_pts", ascending=False)
        if len(sub):
            print(sub.drop(columns="pos").to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
