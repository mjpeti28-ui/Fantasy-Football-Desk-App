"""Snapshot projection sources and score them against real results.

    python tools/track_projections.py snapshot   # archive current projection files if changed
    python tools/track_projections.py score      # snapshot, pull any new actuals, print accuracy

Run ``score`` each Tuesday after Monday Night Football. It also refreshes ESPN's
weekly projections (history/actuals/2026/espn_weekly_proj.csv), which v2 needs. See projection_tracking.py
for method and limits (rostered players only; single weeks are noise).
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

import projection_tracking as pt  # noqa: E402

SCORES_DIR = REPO_ROOT / "history" / "projection_scores"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["snapshot", "score"])
    args = parser.parse_args(argv)

    written = pt.snapshot_projections()
    try:  # keep ESPN's weekly projections current (1 ESPN call per week; also feeds valuation v2)
        from datetime import datetime, timezone

        import espn_history

        upto = pt.first_week_after(datetime.now(timezone.utc))
        espn_history.build_weekly_projections(pt.SEASON, range(1, min(upto, pt.REGULAR_SEASON_WEEKS) + 1))
    except Exception as exc:  # network/ESPN hiccups shouldn't block scoring
        print(f"ESPN weekly projections not refreshed ({exc})")
    print("snapshots:", ", ".join(p.name for p in written) if written else "no changes since last snapshot")
    if args.command == "snapshot":
        return 0

    result = pt.score()
    summary = result["summary"]
    if summary.empty:
        print("nothing to score yet (need a completed week with a snapshot taken before its kickoff)")
        return 0
    pd.set_option("display.width", 160)
    print(summary.to_string(index=False))
    weeks = int(summary["weeks"].max())
    if weeks < 4:
        print(f"\nonly {weeks} week(s) scored -- treat this as anecdote, not evidence.")
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SCORES_DIR / f"{pt.SEASON}_summary.csv", index=False)
    result["detail"].to_csv(SCORES_DIR / f"{pt.SEASON}_detail.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
