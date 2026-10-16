"""Pull FantasyPros rest-of-season PPR projections via the API into the projections CSV.

Replaces PFF's ``projections.csv`` as the points source. Output has the same
schema (``fantasyPointsRank, playerName, teamName, position, byeWeek,
fantasyPoints``) plus ``fpid``, so ``data.load_projections`` reads it unchanged.

Free tier = 10 players per call, 100 calls/day (see ``fantasypros_api.py``), so
this fetches only who the model needs:

- the top N at each position by the FantasyPros ROS board (deep enough to
  cover every team's starters plus the replacement-level window), and
- every rostered player, wherever they rank.

Everyone else falls back to ``data.py``'s rank-curve imputation.

Needs ``FantasyPros_ROS_rankings.csv`` (run ``tools/fetch_fantasypros.py``
first) for the player universe and FantasyPros IDs.

Usage (from the repository root):

    python tools/fetch_fantasypros_projections.py            # ~25-30 API calls
    python tools/fetch_fantasypros_projections.py --plan     # show call count, spend nothing

Same-day reruns are free: responses are cached per day under .cache/fantasypros/.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env import load_local_env  # noqa: E402

load_local_env()

from fantasypros_api import FantasyProsClient, ppr_points  # noqa: E402
from teams import canonical_dst_name, canonical_team  # noqa: E402

# 10-team league, 7 starting slots (QB, RB, RB/WR, WR, WR/TE, TE, flex).
# Depth covers all starters + the replacement window with room to spare.
DEFAULT_DEPTH = {"QB": 30, "RB": 60, "WR": 80, "TE": 30, "K": 15, "DST": 16}
OUTPUT_FIELDS = ["fantasyPointsRank", "playerName", "teamName", "position", "byeWeek", "fantasyPoints", "fpid"]


def _rostered_names() -> List[str]:
    from data import load_rosters

    rosters = load_rosters()
    return [n.replace(" (IR)", "") for team in rosters.values() for names in team.values() for n in names]


def select_players(board: pd.DataFrame, depth: Dict[str, int], rostered: List[str]) -> tuple[Dict[str, Set[int]], List[str]]:
    """Return ({position: {fpid}}, unmatched roster names)."""
    import alias
    from data import normalize_name

    board = board.dropna(subset=["FPID"]).copy()
    board["FPID"] = board["FPID"].astype(int)
    board["name_norm"] = board["Name"].apply(normalize_name)

    wanted: Dict[str, Set[int]] = {pos: set() for pos in depth}
    for pos, n in depth.items():
        top = board[board["Position"] == pos].sort_values("Position Rank").head(n)
        wanted[pos].update(top["FPID"])

    unmatched: List[str] = []
    for name in rostered:
        match = alias.find_best(name, board)
        if match is None:
            unmatched.append(name)
            continue
        row = board[board["Name"] == match].iloc[0]
        wanted.setdefault(row["Position"], set()).add(int(row["FPID"]))
    return wanted, unmatched


def to_rows(players_by_pos: Dict[str, List[dict]], board: pd.DataFrame) -> List[dict]:
    b = board.dropna(subset=["FPID", "Bye Week"])
    bye = dict(zip(b["FPID"].astype(int), b["Bye Week"].astype(int)))
    rows = []
    for pos, players in players_by_pos.items():
        for p in players:
            pts = ppr_points(p)
            if pts is None:
                continue
            team = canonical_team(p.get("team_id"))
            name = p.get("name")
            if pos == "DST":
                name = canonical_dst_name(name, team) or name
            rows.append(
                {
                    "playerName": name,
                    "teamName": team,
                    "position": pos.lower(),
                    "byeWeek": bye.get(int(p["fpid"])) if p.get("fpid") is not None else None,
                    "fantasyPoints": round(pts, 2),
                    "fpid": p.get("fpid"),
                }
            )
    rows.sort(key=lambda r: -r["fantasyPoints"])
    for i, r in enumerate(rows, start=1):
        r["fantasyPointsRank"] = i
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rankings", default="FantasyPros_ROS_rankings.csv")
    parser.add_argument("--out", default="FantasyPros_ROS_projections.csv")
    parser.add_argument("--season", type=int, default=int(os.getenv("ESPN_SEASON", "2026")))
    parser.add_argument("--plan", action="store_true", help="Print the call plan and exit without calling the API.")
    args = parser.parse_args(argv)

    board_path = REPO_ROOT / args.rankings
    if not board_path.exists() or "FPID" not in pd.read_csv(board_path, nrows=0).columns:
        print(f"{args.rankings} missing or has no FPID column -- run tools/fetch_fantasypros.py first.", file=sys.stderr)
        return 1
    board = pd.read_csv(board_path)

    wanted, unmatched = select_players(board, DEFAULT_DEPTH, _rostered_names())
    client = FantasyProsClient()
    counts = {pos: len(ids) for pos, ids in wanted.items()}
    needed = client.calls_needed(counts)
    print(f"players to fetch: {counts} -> up to {needed} API calls "
          f"(cached batches are free; {client.calls_remaining_today()} usable calls left today)")
    if unmatched:
        print(f"  rostered but not on the FantasyPros board (will be imputed): {unmatched}")
    if args.plan:
        return 0

    players_by_pos = {pos: client.ros_projections(args.season, pos, ids) for pos, ids in wanted.items() if ids}
    rows = to_rows(players_by_pos, board)
    returned = sum(len(v) for v in players_by_pos.values())
    requested = sum(counts.values())
    if returned < requested:
        print(f"  note: asked for {requested} players, API returned {returned}")

    out = REPO_ROOT / args.out
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out.name}: {len(rows)} projections; {client.calls_used_today()} API calls used today")
    import projection_tracking

    snaps = projection_tracking.snapshot_projections()
    if snaps:
        print("archived for accuracy tracking:", ", ".join(p.name for p in snaps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
