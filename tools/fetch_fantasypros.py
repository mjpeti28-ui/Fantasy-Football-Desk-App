"""Fetch FantasyPros Rest-of-Season PPR consensus rankings into the rankings CSV this project consumes.

Why FantasyPros: PFF's "Draft Rankings" board (``fetch_pff.py``) is refreshed
roughly at season start and goes stale fast (confirmed 9 days behind live
role changes as of 2026-09-22); FantasyPros' ROS consensus refreshes daily.
Caveat: the ROS consensus is thin -- 5-7 experts per page as of 2026-09-22.

Structure (built so the FantasyPros API can replace the scrape without touching
anything downstream):

1. **Transport** -- :func:`fetch_consensus` returns one consensus-rankings
   payload per position. Today it scrapes the ``var ecrData = {...}`` blob
   embedded in the public rankings pages. That blob has the *same shape* as
   the API's ``GET /nfl/{season}/consensus-rankings`` response
   (``NFLRankingsResponse`` in the v2 spec: ``sport``, ``scoring``,
   ``ranking_type_name``, ``position_id``, ``players[]`` with ``player_id``,
   ``rank_ecr``, ``pos_rank``, ``player_team_id``, ``player_bye_week``...), so
   swapping in the API means reimplementing only this function with
   ``type=ROS, scoring=PPR, position=ALL|QB|RB|WR|TE|K|DST`` and the
   ``x-api-key`` header.
2. **Normalization** -- :func:`build_rows` turns payloads into CSV rows:
   canonical team codes and ``"<Nickname> D/ST"`` defense names (via
   ``teams.py``), a dense overall ``Rank``, and a ``Position Rank`` derived
   from that same order.

What this deliberately does NOT write: projected points. The pages carry an
``r2p_pts`` field that is strictly monotonic in position rank at every
position (0 exceptions across 454 players on 2026-09-22) and does not exist in
the API spec -- it is a rank-to-points conversion, not a projection. Feeding
it to the optimizer would silently turn the points-based lineup/VOR logic back
into rank-based logic. Points come from ``PROJECTIONS_CSV`` (PFF's
``projections.csv``) until the API's ``/nfl/{season}/projections?ros=true``
(real stat projections with ``points_ppr``) is wired in.

Usage (from the repository root):

    python tools/fetch_fantasypros.py              # writes FantasyPros_ROS_rankings.csv
    python tools/fetch_fantasypros.py --dry-run    # report only, write nothing
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teams import canonical_dst_name, canonical_team  # noqa: E402

POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")
OVERALL = "ALL"

_BASE = "https://www.fantasypros.com/nfl/rankings"
_SCRAPE_URLS = {
    OVERALL: f"{_BASE}/ros-ppr-overall.php",
    "QB": f"{_BASE}/ros-qb.php",
    "RB": f"{_BASE}/ros-ppr-rb.php",
    "WR": f"{_BASE}/ros-ppr-wr.php",
    "TE": f"{_BASE}/ros-ppr-te.php",
    "K": f"{_BASE}/ros-k.php",
    "DST": f"{_BASE}/ros-dst.php",
}
_HEADERS = {"User-Agent": "Mozilla/5.0 (PFF-fantasy-ranking/fantasypros-client)"}
_ECR_RE = re.compile(r"var ecrData = (\{.*?\});\s*\n", re.DOTALL)

# Positions where PPR vs STD changes the ranking. K/DST pages only exist as STD.
_PPR_POSITIONS = {OVERALL, "RB", "WR", "TE"}

RANKINGS_FIELDS = [
    "Rank", "Position", "Name", "Team", "Position Rank",
    "Bye Week", "ADP", "Tier", "Projected Points",
    "Projected Points Low", "Projected Points High",
    "FPID",  # FantasyPros player_id -- what the API's projections endpoint filters on
]


# --------------------------------------------------------------------------- transport


def fetch_consensus(position: str, timeout: float = 30.0) -> Dict[str, Any]:
    """Return a consensus-rankings payload (API ``NFLRankingsResponse`` shape) for one position.

    ``position`` is ``"ALL"`` for the cross-position board or one of POSITIONS.
    Current transport: public-page scrape. API replacement goes here.
    """
    url = _SCRAPE_URLS[position]
    resp = httpx.get(url, headers=_HEADERS, timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    match = _ECR_RE.search(resp.text)
    if not match:
        raise RuntimeError(f"Could not find embedded ecrData on {url!r} -- page layout may have changed.")
    payload = json.loads(match.group(1))
    _check_payload(payload, position)
    return payload


def _check_payload(payload: Dict[str, Any], position: str) -> None:
    """Fail loudly if a page/endpoint silently serves the wrong board."""
    rtype = str(payload.get("ranking_type_name", "")).lower()
    if rtype and rtype != "ros":
        raise RuntimeError(f"{position}: expected a ROS ranking, got {rtype!r}")
    pos_id = str(payload.get("position_id", "")).upper()
    if pos_id and pos_id != position:
        raise RuntimeError(f"{position}: payload is for position {pos_id!r}")
    scoring = str(payload.get("scoring", "")).upper()
    if position in _PPR_POSITIONS and scoring and scoring != "PPR":
        raise RuntimeError(f"{position}: expected PPR scoring, got {scoring!r}")
    if not payload.get("players"):
        raise RuntimeError(f"{position}: payload has no players")


# --------------------------------------------------------------------------- normalization


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _pos_rank(player: Dict[str, Any]) -> Optional[int]:
    """``pos_rank`` is a string like ``"RB12"`` in both the scrape and the API."""
    raw = player.get("pos_rank")
    if isinstance(raw, str):
        digits = re.sub(r"\D", "", raw)
        if digits:
            return int(digits)
    return _int_or_none(player.get("rank_ecr"))


def build_rows(overall: Dict[str, Any], by_position: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge position boards with the overall board into CSV rows.

    - Player universe = union of the position boards (the overall board only
      covers its top ~400; deep K/DST/WR5s are missing from it).
    - ``Rank``: overall-board order first, then players missing from the
      overall board ordered by their position rank; renumbered densely 1..N
      (``data.py`` needs an integer Rank on every row).
    - ``Position Rank``: recomputed from that final order, so the two columns
      never disagree. The position boards use a different expert panel than
      the overall board, and taking their ranks verbatim produced 108
      rank/position-rank inversions on 2026-09-22. ``data.py`` recomputes
      PosRank from Rank anyway; this keeps the CSV itself honest.
    """
    overall_rank: Dict[Any, int] = {}
    for p in overall.get("players", []):
        pid, rank = p.get("player_id"), _int_or_none(p.get("rank_ecr"))
        if pid is not None and rank is not None:
            overall_rank[pid] = rank

    rows: List[Dict[str, Any]] = []
    seen = set()
    for position, payload in by_position.items():
        for p in payload.get("players", []):
            pid, name = p.get("player_id"), p.get("player_name")
            if not name or pid in seen:
                continue
            seen.add(pid)
            team = canonical_team(p.get("player_team_id"))
            if position == "DST":
                name = canonical_dst_name(name, team) or name
            rows.append(
                {
                    "_overall": overall_rank.get(pid),
                    "_pos": _pos_rank(p),
                    "Position": position,
                    "Name": name,
                    "Team": team,
                    "Bye Week": p.get("player_bye_week"),
                    "ADP": None,  # mid-season ROS boards have no real ADP
                    "Tier": p.get("tier"),
                    "Projected Points": None,  # see module docstring
                    "Projected Points Low": None,
                    "Projected Points High": None,
                    "FPID": pid,
                }
            )

    rows.sort(key=lambda r: (r["_overall"] is None, r["_overall"] or 0, r["_pos"] or 10**6))
    pos_counter: Counter = Counter()
    for i, row in enumerate(rows, start=1):
        row["Rank"] = i
        pos_counter[row["Position"]] += 1
        row["Position Rank"] = pos_counter[row["Position"]]
    return rows


def write_rankings_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RANKINGS_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- CLI


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rankings-out", default="FantasyPros_ROS_rankings.csv")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be written without touching files.")
    args = parser.parse_args(argv)

    overall = fetch_consensus(OVERALL)
    by_position = {pos: fetch_consensus(pos) for pos in POSITIONS}
    rows = build_rows(overall, by_position)

    counts = Counter(r["Position"] for r in rows)
    unranked = sum(1 for r in rows if r["_overall"] is None)
    print(f"fetched {len(rows)} players (FantasyPros ROS PPR consensus; {overall.get('total_experts')} experts on the overall board)")
    print("  by position:", dict(sorted(counts.items())))
    print(f"  {unranked} not on the overall board (placed after it by position rank)")
    print(f"  top 3: {', '.join(r['Name'] for r in rows[:3])}")

    if args.dry_run:
        print("dry run: no files written")
        return 0

    out = REPO_ROOT / args.rankings_out
    write_rankings_csv(rows, out)
    print(f"wrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
