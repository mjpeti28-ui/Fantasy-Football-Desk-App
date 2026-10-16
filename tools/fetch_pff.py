"""Fetch PFF fantasy rankings and projections into the CSVs this project consumes.

PFF's site is backed by a JSON API at ``consumer-api.pff.com``. It takes a static
``api-key`` header -- a public client key shipped in PFF's own frontend bundle and
identical for every visitor, not a per-user credential -- so no login or cookie is
needed. This replaces the manual "export CSV from the rankings page" step.

Note that ``www.pff.com/api/fantasy/*`` is a *different*, older surface that
truncates responses to 10 records for anonymous callers. Use the consumer API.

Usage (from the repository root):

    python tools/fetch_pff.py                      # writes both CSVs
    python tools/fetch_pff.py --scoring REDRAFT_HALF_PPR
    python tools/fetch_pff.py --dry-run            # report only, write nothing
    python tools/fetch_pff.py --ros                # in-season ROS board -> PFF_ROS_rankings.csv

In-season, ``/fantasy/rankings`` is frozen at the preseason draft board (verified
2026-10-16). ``--ros`` reads ``/fantasy/ros-rankings`` instead, the endpoint behind
pff.com/fantasy/rankings/ros. It writes ranks plus PFF's projected points, which are
for the NEXT game only (a player on bye has none), so they are labelled as such and
not written to projections.csv. Each ROS board is also archived under
history/rankings/<season>/ for ``backtest/pff_ros.py``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent

CONSUMER_API = "https://consumer-api.pff.com/football/v1"
PFF_PUBLIC_CLIENT_KEY = "0f6ca1f4-79d4-11ee-b962-0242ac120002"
HEADERS = {
    "Accept": "application/json",
    "Origin": "https://www.pff.com",
    "Referer": "https://www.pff.com/",
    "api-key": PFF_PUBLIC_CLIENT_KEY,
    "User-Agent": "PFF-Fantasy-Ranking/pff-client",
}

# position -> query spellings to try. "DST" is rejected (400, 2026-10-16); the others are guesses.
ROS_POSITIONS = {"QB": ("QB",), "RB": ("RB",), "WR": ("WR",), "TE": ("TE",), "K": ("K",), "DST": ("D", "DEF", "DST")}
ROS_REQUIRED = {"QB", "RB", "WR", "TE"}
# rank.current is the shared flex board for these; for QB/K it just repeats the position rank
FLEX = {"RB", "WR", "TE"}

# PFF labels team defenses "D"; the rest of this project calls them "DST".
POSITION_ALIASES = {"D": "DST"}


def fetch_rankings(scoring: str, league_type: str = "standard", timeout: float = 60.0) -> List[Dict[str, Any]]:
    """Return every ranked player for the given scoring format.

    The endpoint accepts a ``page`` parameter but ignores it -- page 2 returns the
    same payload as page 1 -- so a single request is the complete set.
    """

    resp = httpx.get(
        f"{CONSUMER_API}/fantasy/rankings",
        params={"page": 1, "leagueType": league_type, "scoringType": scoring},
        headers=HEADERS,
        timeout=timeout,
    )
    resp.raise_for_status()
    rankings = resp.json().get("rankings") or []
    if not rankings:
        raise RuntimeError(f"PFF returned no rankings for scoringType={scoring!r}")
    return rankings


def fetch_ros(position: str, scoring: str = "ppr", timeout: float = 60.0) -> Dict[str, Any]:
    """One position's rest-of-season board (the payload, with ``currentGameWeek``)."""

    resp = httpx.get(
        f"{CONSUMER_API}/fantasy/ros-rankings",
        params={"scoringType": scoring, "position": position},
        headers=HEADERS,
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    if not payload.get("rankings"):
        raise RuntimeError(f"PFF returned no ROS rankings for position={position!r}")
    return payload


def build_ros_rows(payloads: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten per-position ROS boards, flex positions first by PFF's flex rank."""

    rows: List[Dict[str, Any]] = []
    for position, payload in payloads.items():
        week = payload.get("currentGameWeek")
        for entry in payload["rankings"]:
            pos = POSITION_ALIASES.get(entry.get("position"), entry.get("position")) or position
            rows.append(
                {
                    "Flex Rank": _nested(entry, "rank", "current") if pos in FLEX else None,
                    "Position": pos,
                    "Name": _full_name(entry),
                    "Team": entry.get("teamAbbreviation"),
                    "Position Rank": _nested(entry, "rank", "position"),
                    "Bye Week": entry.get("byeWeek"),
                    "PFF ID": entry.get("playerId"),
                    "Next Week": week,
                    "Next Week Points": _nested(entry, "projection", "points", "mid"),
                    "Next Week Points Low": _nested(entry, "projection", "points", "low"),
                    "Next Week Points High": _nested(entry, "projection", "points", "high"),
                }
            )
    rows = [r for r in rows if r["Position Rank"] is not None and r["Name"]]
    order = list(ROS_POSITIONS)
    rows.sort(key=lambda r: (r["Flex Rank"] is None, r["Flex Rank"] or 0, order.index(r["Position"]) if r["Position"] in order else 99,
                             r["Position Rank"]))
    return rows


ROS_FIELDS = [
    "Flex Rank", "Position", "Name", "Team", "Position Rank", "Bye Week", "PFF ID",
    "Next Week", "Next Week Points", "Next Week Points Low", "Next Week Points High",
]


def write_ros_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_ros(args: argparse.Namespace) -> int:
    import shutil
    from collections import Counter
    from datetime import datetime, timezone

    payloads: Dict[str, Dict[str, Any]] = {}
    for pos, spellings in ROS_POSITIONS.items():
        for i, spelling in enumerate(spellings):
            try:
                payloads[pos] = fetch_ros(spelling)
                break
            except Exception as exc:  # noqa: BLE001
                if i < len(spellings) - 1:
                    continue
                if pos in ROS_REQUIRED:
                    raise
                print(f"  skipped {pos} (tried {', '.join(spellings)}): {exc}")
    rows = build_ros_rows(payloads)
    counts = Counter(r["Position"] for r in rows)
    week = next(iter(payloads.values())).get("currentGameWeek")
    print(f"fetched {len(rows)} PFF ROS-ranked players (as of week {week})")
    print("  by position:", dict(sorted(counts.items())))
    print(f"  top 3: {', '.join(r['Name'] for r in rows[:3])}")
    if args.dry_run:
        print("dry run: no files written")
        return 0

    out = REPO_ROOT / args.ros_out
    write_ros_csv(rows, out)
    now = datetime.now(timezone.utc)
    arch = REPO_ROOT / "history" / "rankings" / str(now.year) / f"{now:%Y%m%dT%H%M%S}_PFF_ROS_rankings.csv"
    arch.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out, arch)
    print(f"wrote {out.name}; archived {arch.relative_to(REPO_ROOT)}")
    return 0


def _full_name(row: Dict[str, Any]) -> str:
    """Display name for a ranked entry.

    Team defenses carry no first/last name -- their identity is ``teamName`` --
    so they are labelled to match ESPN's "Seahawks D/ST" form, which is what the
    roster aliaser sees on the other side.
    """

    name = " ".join(part for part in (row.get("firstName"), row.get("lastName")) if part).strip()
    if name:
        return name
    if row.get("position") == "D" and row.get("teamName"):
        return f"{row['teamName']} D/ST"
    return ""


def _nested(row: Dict[str, Any], *path: str) -> Any:
    node: Any = row
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def build_rows(rankings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten PFF's nested payload into one record per player."""

    rows: List[Dict[str, Any]] = []
    for entry in rankings:
        position = entry.get("position")
        rows.append(
            {
                "Rank": _nested(entry, "rank", "current"),
                "Position": POSITION_ALIASES.get(position, position),
                "Name": _full_name(entry),
                "Team": entry.get("teamAbbreviation"),
                "Position Rank": _nested(entry, "rank", "position"),
                "Bye Week": entry.get("byeWeek"),
                "ADP": entry.get("adp"),
                "Tier": entry.get("tier"),
                "Projected Points": _nested(entry, "projection", "points", "mid"),
                "Projected Points Low": _nested(entry, "projection", "points", "low"),
                "Projected Points High": _nested(entry, "projection", "points", "high"),
            }
        )
    rows = [r for r in rows if r["Rank"] is not None and r["Name"]]
    rows.sort(key=lambda r: r["Rank"])
    return rows


def write_rankings_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    """Write the rankings feed.

    ``data._prepare_rankings_dataframe`` maps these headers onto its internal
    schema, so the column names here are load-bearing -- keep them in sync.
    """

    fields = [
        "Rank", "Position", "Name", "Team", "Position Rank",
        "Bye Week", "ADP", "Tier", "Projected Points",
        "Projected Points Low", "Projected Points High",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_projections_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    """Write the projections feed.

    ``data.load_projections`` requires exactly ``playerName`` and ``fantasyPoints``;
    the extra columns are carried for reference and ignored by the loader.
    """

    fields = ["fantasyPointsRank", "playerName", "teamName", "position", "byeWeek", "fantasyPoints"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            points = row["Projected Points"]
            if points is None:
                continue
            writer.writerow(
                {
                    "fantasyPointsRank": row["Rank"],
                    "playerName": row["Name"],
                    "teamName": row["Team"],
                    "position": (row["Position"] or "").lower(),
                    "byeWeek": row["Bye Week"],
                    "fantasyPoints": points,
                }
            )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scoring", default="REDRAFT_PPR",
                        help="PFF scoring format. Known good: REDRAFT_PPR, REDRAFT_HALF_PPR.")
    parser.add_argument("--league-type", default="standard")
    parser.add_argument("--rankings-out", default="PFF_rankings.csv")
    parser.add_argument("--projections-out", default="projections.csv")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be written without touching files.")
    parser.add_argument("--ros", action="store_true", help="Fetch the in-season ROS board instead of the draft board.")
    parser.add_argument("--ros-out", default="PFF_ROS_rankings.csv")
    args = parser.parse_args(argv)
    if args.ros:
        return run_ros(args)

    rankings = fetch_rankings(args.scoring, args.league_type)
    rows = build_rows(rankings)

    from collections import Counter
    counts = Counter(r["Position"] for r in rows)
    print(f"fetched {len(rows)} ranked players ({args.scoring})")
    print("  by position:", dict(sorted(counts.items())))
    print(f"  top 3: {', '.join(r['Name'] for r in rows[:3])}")

    if args.dry_run:
        print("dry run: no files written")
        return 0

    rankings_path = REPO_ROOT / args.rankings_out
    projections_path = REPO_ROOT / args.projections_out
    write_rankings_csv(rows, rankings_path)
    write_projections_csv(rows, projections_path)
    print(f"wrote {rankings_path.name} and {projections_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
