"""Points-based trade tool (experimental; see trade_value.py for method and limits).

    python tools/trade_value.py evaluate --team 3 --give "Chuba Hubbard,Jalen Coker" --with 10 --get "Some Player"
    python tools/trade_value.py find --team 3 [--with 10] [--top 15] [--wide]
    python tools/trade_value.py teams
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

import trade_value as tv  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["evaluate", "find", "teams"])
    ap.add_argument("--team", type=int, default=3)
    ap.add_argument("--with", dest="other", type=int)
    ap.add_argument("--give", default="")
    ap.add_argument("--get", default="")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--min-their-gain", type=float, default=0.0,
                    help="require the partner to gain at least this many points (weak historical hint that partner gain matters)")
    ap.add_argument("--refresh", action="store_true", help="re-pull the live season from ESPN first")
    ap.add_argument("--json", action="store_true", help="evaluate: also print the raw JSON")
    ap.add_argument("--wide", action="store_true",
                    help="also search 2-for-2, 3-for-1 and 3-for-2 (slow; outside the calibration backtest)")
    a = ap.parse_args(argv)
    if a.refresh:
        import espn_history

        espn_history.build_season(2026, last_week=int(pd.read_csv(REPO_ROOT / "history/actuals/2026/rosters_by_week.csv")["week"].max()))
    ctx = tv.load_context()
    if a.command == "teams":
        for t in sorted(ctx.rosters):
            print(t, ctx.team_name(t))
        return 0
    if a.command == "evaluate":
        give = tv.resolve(ctx, a.team, [x.strip() for x in a.give.split(",") if x.strip()])
        get = tv.resolve(ctx, a.other, [x.strip() for x in a.get.split(",") if x.strip()])
        ev = tv.evaluate(ctx, a.team, give, a.other, get)
        for t, d in ev["teams"].items():
            print(f"{d['name']}: gives {', '.join(d['gives'])} | gets {', '.join(d['gets'])}"
                  + (f" | drops {', '.join(d['would_drop'])}" if d["would_drop"] else ""))
            print(f"  gain {d['delta_total']:+.1f} (calibrated {d['delta_total_calibrated']:+.1f}; "
                  f"playoff weeks {d['delta_playoffs']:+.1f}) -> P(direction right) {d['p_direction_right']:.0%}, {d['edge_label']}")
        print("why (ours / ESPN / experts -> blended ppg, chance he plays):")
        for n, w in ev["teams"][a.team]["why"].items():
            print(f"  {n}: {w}")
        print("edges under 10 are noise (right 47-57% in backtests); for seed/playoff odds run tools/season_odds.py "
              "with the same --give/--with/--get")
        if a.json:
            print(json.dumps(ev, indent=2))
        return 0
    if a.other:
        ctx.rosters = {k: v for k, v in ctx.rosters.items() if k in (a.team, a.other)}
    pd.set_option("display.width", 220)
    shapes = tv.WIDE_SHAPES if a.wide else ("1-1", "2-1", "1-2")
    partners = [a.other] if a.other else None
    print(tv.find(ctx, a.team, top=a.top, min_their_gain=a.min_their_gain, shapes=shapes,
                  partners=partners).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
