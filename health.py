"""Pre-flight checks on the evaluation's inputs.

Silent join failures did real damage before (every rostered D/ST went
unmatched after the FantasyPros switch and nothing complained). Run this with
every evaluation; it answers "can I trust today's numbers?" before anyone
reads them.

Checks:
- roster names that match nothing in the rankings (ERROR -- those players are
  simply missing from the evaluation)
- rostered players whose projection is estimated rather than sourced
- stale input files
- rostered players ESPN lists as out / IR / suspended, whom the model still
  values at full rest-of-season projection
- rostered players with no stat line for consecutive completed weeks while
  not listed as out (the Josh Jacobs case: "day-to-day", 0 games played)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent
STALE_DAYS = 7
OUT_STATUSES = {"OUT", "INJURY_RESERVE", "SUSPENSION", "IR", "PUP"}


@dataclass
class Finding:
    level: str  # ERROR | WARN | INFO
    check: str
    message: str
    players: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        tail = f": {', '.join(self.players)}" if self.players else ""
        return f"[{self.level}] {self.check} -- {self.message}{tail}"


def _file_age_days(path: Path) -> Optional[float]:
    return (time.time() - path.stat().st_mtime) / 86400 if path.exists() else None


def run_health_check(
    rankings_path: Optional[str] = None,
    projections_path: Optional[str] = None,
    *,
    rosters: Optional[Dict[str, Dict[str, List[str]]]] = None,
    use_espn: bool = True,
    dnp_weeks: int = 2,
) -> List[Finding]:
    import alias
    import data
    from main import DEFAULT_PROJECTIONS, DEFAULT_RANKINGS

    rankings_path = rankings_path or str(DEFAULT_RANKINGS)
    projections_path = projections_path or str(DEFAULT_PROJECTIONS)
    findings: List[Finding] = []

    # -- freshness
    for label, path in (("rankings", Path(rankings_path)), ("projections", Path(projections_path))):
        age = _file_age_days(path)
        if age is None:
            findings.append(Finding("ERROR", "freshness", f"{label} file missing: {path.name}"))
        elif age > STALE_DAYS:
            findings.append(Finding("WARN", "freshness", f"{label} file {path.name} is {age:.0f} days old"))
        else:
            findings.append(Finding("INFO", "freshness", f"{label}: {path.name}, updated {age:.1f} days ago"))

    # -- projection basis: a full-season export masquerading as rest-of-season
    # (happened in the 2025 archive: the Nov 5 "projections" had 18 games for everyone).
    try:
        proj_raw = pd.read_csv(projections_path)
        if "games" in proj_raw.columns and not proj_raw.empty:
            import projection_tracking as pt
            from datetime import datetime, timezone

            week = pt.first_week_after(datetime.now(timezone.utc))
            left = pt.REGULAR_SEASON_WEEKS - week + 1
            max_games = float(proj_raw["games"].max())
            if max_games > left:
                findings.append(Finding(
                    "ERROR", "projections",
                    f"{Path(projections_path).name} projects up to {max_games:.0f} games but only {left} weeks remain "
                    "-- this looks like a full-season file, not rest-of-season",
                ))
    except Exception:
        pass

    df = data.load_rankings(rankings_path, projections_path)
    rosters = rosters if rosters is not None else data.load_rosters()
    roster_names = [n for team in rosters.values() for names in team.values() for n in names]
    amap = alias.build_alias_map(roster_names, df)

    # -- matching
    unmatched = sorted({k for k, v in amap.items() if v is None})
    if unmatched:
        findings.append(Finding("ERROR", "matching", f"{len(unmatched)} rostered players match nothing in the rankings and are left out of the evaluation", unmatched))
    else:
        findings.append(Finding("INFO", "matching", f"all {len(amap)} rostered players matched"))

    # -- projection provenance
    if "ProjSource" in df.columns:
        src = df.set_index("Name")["ProjSource"]
        by_kind: Dict[str, List[str]] = {}
        for roster_name, canon in amap.items():
            if canon is None:
                continue
            kind = src.get(canon)
            if isinstance(kind, pd.Series):
                kind = kind.iloc[0]
            by_kind.setdefault(kind or "unknown", []).append(canon)
        for kind, level, msg in (
            ("median", "WARN", "projection filled with an overall median (no rank curve available)"),
            ("imputed", "WARN", "no projection in the source; estimated from their rank"),
            ("fallback", "INFO", "projection matched by last name + team + position (name spelled differently)"),
            ("none", "ERROR", "no projections loaded at all -- evaluation is rank-only"),
        ):
            if by_kind.get(kind):
                findings.append(Finding(level, "projections", msg, sorted(set(by_kind[kind]))))

    if not use_espn:
        return findings

    # -- ESPN status and availability
    try:
        from espn_client import fetch_league_state, get_last_league_state

        league = get_last_league_state() or fetch_league_state()
    except Exception as exc:  # pragma: no cover - network
        findings.append(Finding("WARN", "espn", f"could not load ESPN league state ({exc}); injury/availability checks skipped"))
        return findings

    status: Dict[str, str] = {}
    owner: Dict[str, str] = {}
    for team in league.teams.values():
        for p in team.roster:
            st = (p.injury_status or "").upper()
            if p.on_ir:
                st = st or "INJURY_RESERVE"
            status[p.full_name] = st
            owner[p.full_name] = team.name

    out = sorted(f"{n} ({s}, {owner[n]})" for n, s in status.items() if s in OUT_STATUSES)
    if out:
        findings.append(Finding("WARN", "availability", "listed out/IR/suspended but valued at full rest-of-season projection", out))

    try:
        import projection_tracking as pt

        weeks = pt.completed_weeks()[-dnp_weeks:]
        if len(weeks) == dnp_weeks:
            played = None
            for w in weeks:
                a = pt.load_actuals(w)
                wk = set(a.loc[a["played"], "name"])
                rostered = set(a["name"])
                missing = rostered - wk
                played = missing if played is None else played & missing
            streak = sorted(
                f"{n} ({status.get(n) or 'no ESPN tag'}, {owner.get(n, '?')})"
                for n in (played or set())
                if n in status and status.get(n) not in OUT_STATUSES
            )
            if streak:
                findings.append(Finding(
                    "WARN", "availability",
                    f"no stat line in weeks {weeks[0]}-{weeks[-1]} but not listed out -- check before trusting their value",
                    streak,
                ))
    except Exception as exc:  # pragma: no cover - network / cache
        findings.append(Finding("WARN", "availability", f"DNP check skipped ({exc})"))

    return findings


def summarize(findings: List[Finding]) -> str:
    order = {"ERROR": 0, "WARN": 1, "INFO": 2}
    lines = [str(f) for f in sorted(findings, key=lambda f: order.get(f.level, 3))]
    errs = sum(f.level == "ERROR" for f in findings)
    warns = sum(f.level == "WARN" for f in findings)
    lines.append(f"health: {errs} error(s), {warns} warning(s)")
    return "\n".join(lines)
