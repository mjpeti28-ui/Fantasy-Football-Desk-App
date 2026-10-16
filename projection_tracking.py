"""Measure projection accuracy against real results.

The evaluation's outputs swing hard depending on whose projections feed it
(PFF vs FantasyPros reshuffled the league on 2026-09-22), and nothing told us
which source is closer to reality. This module fixes that:

1. **Snapshot** every projection source each time it changes
   (``history/projection_snapshots/<season>/<UTC stamp>__<source>.csv``).
2. **Actuals**: pull each completed week's real fantasy points for every
   rostered player from ESPN (league scoring, so they match what counts) and
   cache them (``history/actuals/<season>/week_<N>.csv``).
3. **Score**: for week W, use each source's latest snapshot taken before that
   week's Thursday kickoff, convert its rest-of-season total to a per-game
   number, and compare with what players actually scored. Report per source
   and position: n, MAE, bias, and within-position rank correlation
   (the metric that matters most -- the model mostly cares who is better
   than whom, and the sources differ in overall level by ~11 points).

Honest limits:
- Only rostered players are scored (ESPN's league endpoint only returns
  rostered players), so this measures accuracy on the players you actually
  choose between -- not the whole waiver pool.
- A single week is very noisy. Treat anything under ~4 weeks as anecdote.
- Snapshots before a week can only exist from when tracking started; the one
  exception is the archived preseason PFF projections (Aug 24), which cover
  weeks 1-2 as a full-season projection.
- ``espn_weekly`` is ESPN's single-week projection (joined on ESPN id), scored next
  to the ROS sources; the 2021-2025 backtest (backtest/start_sit.py) found it the
  best start/sit input available.
- FantasyPros terms bar using *their* historical stats. This stores their
  projections (not stats) and scores them against ESPN's actuals.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from teams import canonical_dst_name

ROOT = Path(__file__).resolve().parent
SEASON = 2026
# Thursday of week 1 (2026 opener). Kickoff cutoff = Friday 00:00 UTC of each
# week's Thursday (8:15pm ET Thursday game ~= 00:15 UTC Friday).
WEEK1_THURSDAY = date(2026, 9, 10)
REGULAR_SEASON_WEEKS = 18

# source label -> file in the repo root holding that source's ROS projections
PROJECTION_SOURCES: Dict[str, str] = {
    "pff": "projections.csv",
    "fantasypros": "FantasyPros_ROS_projections.csv",
}

SNAP_DIR = ROOT / "history" / "projection_snapshots" / str(SEASON)
ACTUALS_DIR = ROOT / "history" / "actuals" / str(SEASON)
LEGACY_PFF_DIR = ROOT / "history" / "projections" / str(SEASON)  # pre-existing archive (PFF only)

_SNAP_RE = re.compile(r"^(\d{8}T\d{6})Z__([a-z0-9_]+)\.csv$")
_LEGACY_RE = re.compile(r"^(\d{8}T\d{6})_projections\.csv$")


# ------------------------------------------------------------------ calendar


def week_cutoff(week: int) -> datetime:
    thursday = WEEK1_THURSDAY + timedelta(weeks=week - 1)
    return datetime(thursday.year, thursday.month, thursday.day, tzinfo=timezone.utc) + timedelta(days=1)


def first_week_after(ts: datetime) -> int:
    """The first week whose kickoff is after ``ts`` (i.e. the week a snapshot projects from)."""
    for week in range(1, REGULAR_SEASON_WEEKS + 1):
        if ts < week_cutoff(week):
            return week
    return REGULAR_SEASON_WEEKS + 1


def games_remaining(from_week: int, bye_week: Optional[float]) -> int:
    weeks = max(0, REGULAR_SEASON_WEEKS - from_week + 1)
    if bye_week is not None and not pd.isna(bye_week) and int(bye_week) >= from_week:
        weeks -= 1
    return max(1, weeks)


# ------------------------------------------------------------------ snapshots


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def list_snapshots() -> pd.DataFrame:
    rows = []
    if SNAP_DIR.exists():
        for f in SNAP_DIR.iterdir():
            m = _SNAP_RE.match(f.name)
            if m:
                ts = datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
                rows.append({"source": m.group(2), "taken_at": ts, "path": f})
    if LEGACY_PFF_DIR.exists():
        for f in LEGACY_PFF_DIR.iterdir():
            m = _LEGACY_RE.match(f.name)
            if m:
                # Legacy archive stamps are naive local time from Max's Mac (US Eastern);
                # an hour or four of slop is irrelevant against weekly cutoffs.
                ts = datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc) + timedelta(hours=4)
                rows.append({"source": "pff", "taken_at": ts, "path": f})
    df = pd.DataFrame(rows, columns=["source", "taken_at", "path"])
    return df.sort_values("taken_at").reset_index(drop=True)


def snapshot_projections(now: Optional[datetime] = None) -> List[Path]:
    """Copy each source's current projections file into the snapshot archive if it changed."""
    now = now or datetime.now(timezone.utc)
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    existing = list_snapshots()
    written: List[Path] = []
    for source, filename in PROJECTION_SOURCES.items():
        src = ROOT / filename
        if not src.exists():
            continue
        prior = existing[existing["source"] == source]
        if not prior.empty and _sha(prior.iloc[-1]["path"]) == _sha(src):
            continue
        dest = SNAP_DIR / f"{now.strftime('%Y%m%dT%H%M%S')}Z__{source}.csv"
        shutil.copyfile(src, dest)
        written.append(dest)
    return written


def snapshot_for_week(snapshots: pd.DataFrame, source: str, week: int) -> Optional[pd.Series]:
    cands = snapshots[(snapshots["source"] == source) & (snapshots["taken_at"] < week_cutoff(week))]
    return None if cands.empty else cands.iloc[-1]


def per_game_projections(snapshot: pd.Series) -> pd.DataFrame:
    """name_norm -> projected points per game, from a ROS-total snapshot."""
    from data import normalize_name

    df = pd.read_csv(snapshot["path"])
    if "playerName" not in df.columns or "fantasyPoints" not in df.columns:
        return pd.DataFrame(columns=["name_norm", "position", "proj_pg"])
    from_week = first_week_after(snapshot["taken_at"])
    pos = df["position"].astype(str).str.upper() if "position" in df.columns else pd.Series([""] * len(df))
    names = []
    for name, p, team in zip(df["playerName"], pos, df.get("teamName", pd.Series([None] * len(df)))):
        if p in ("DST", "D/ST", "DEF"):
            name = canonical_dst_name(name, team) or name
        names.append(normalize_name(str(name)))
    bye = df["byeWeek"] if "byeWeek" in df.columns else pd.Series([None] * len(df))
    out = pd.DataFrame({
        "name_norm": names,
        "position": pos.replace({"D/ST": "DST", "DEF": "DST"}),
        "proj_pg": [pts / games_remaining(from_week, b) for pts, b in zip(df["fantasyPoints"].astype(float), bye)],
    })
    return out.drop_duplicates("name_norm")


# ------------------------------------------------------------------ actuals


def completed_weeks() -> List[int]:
    now = datetime.now(timezone.utc)
    # A week is complete once Tuesday 12:00 UTC after it has passed (MNF done).
    return [w for w in range(1, REGULAR_SEASON_WEEKS + 1) if now >= week_cutoff(w) + timedelta(days=4, hours=12)]


def load_actuals(week: int, refresh: bool = False) -> pd.DataFrame:
    """Real fantasy points (league scoring) for every player rostered that week."""
    path = ACTUALS_DIR / f"week_{week}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path)
    from data import normalize_name
    from espn_client import fetch_league_state_for_week

    league = fetch_league_state_for_week(week)
    rows = []
    for team in league.teams.values():
        for p in team.roster:
            player = (p.raw.get("playerPoolEntry") or {}).get("player") or {}
            for s in player.get("stats", []):
                if s.get("scoringPeriodId") == week and s.get("statSourceId") == 0 and s.get("statSplitTypeId") == 1:
                    pos = "DST" if (p.position or "").upper() in ("D/ST", "DST") else (p.position or "")
                    name = canonical_dst_name(p.full_name) if pos == "DST" else p.full_name
                    rows.append({
                        "week": week,
                        "espn_id": p.player_id,
                        "name": name or p.full_name,
                        "name_norm": normalize_name(name or p.full_name),
                        "position": pos,
                        "fantasy_team": team.name,
                        "points": float(s.get("appliedTotal") or 0.0),
                        # Empty stat line = did not play. Excluded from accuracy
                        # scoring (that's an availability miss, not a projection miss)
                        # but kept so the health check can see DNP streaks.
                        "played": bool(s.get("stats")),
                    })
    df = pd.DataFrame(rows).drop_duplicates("espn_id")
    ACTUALS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


# ------------------------------------------------------------------ scoring


def _join(actual: pd.DataFrame, proj: pd.DataFrame) -> pd.DataFrame:
    """Exact normalized-name join, then last-name + position where unique on both
    sides (catches "Kenny"/"Kenneth Gainwell"-style spelling differences)."""
    exact = actual.merge(proj[["name_norm", "proj_pg"]], on="name_norm", how="inner")
    rest = actual[~actual["name_norm"].isin(exact["name_norm"])].copy()
    if rest.empty:
        return exact
    key = lambda df: df["name_norm"].str.split().str[-1] + "|" + df["position"].astype(str).str.upper()
    rest["_k"] = key(rest)
    p = proj[~proj["name_norm"].isin(exact["name_norm"])].copy()
    p["_k"] = key(p)
    p = p[~p["_k"].duplicated(keep=False)]
    rest = rest[~rest["_k"].duplicated(keep=False)]
    fb = rest.merge(p[["_k", "proj_pg"]], on="_k", how="inner").drop(columns="_k")
    return pd.concat([exact, fb], ignore_index=True)


def _rank_corr(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 3:
        return float("nan")
    return float(a.rank().corr(b.rank()))


def score(weeks: Optional[List[int]] = None, include_blend: bool = True) -> Dict[str, pd.DataFrame]:
    """Return {"detail": per player-week rows, "summary": per source x position metrics}."""
    snapshots = list_snapshots()
    weeks = weeks or completed_weeks()
    detail = []
    for week in weeks:
        actual = load_actuals(week)
        actual = actual[actual["played"]]
        per_source = {}
        for source in PROJECTION_SOURCES:
            snap = snapshot_for_week(snapshots, source, week)
            if snap is None:
                continue
            proj = per_game_projections(snap)
            per_source[source] = proj
            joined = _join(actual, proj)
            joined["source"] = source
            joined["snapshot"] = snap["path"].name
            detail.append(joined)
        if include_blend and len(per_source) >= 2:
            merged = None
            for src, proj in per_source.items():
                p = proj[["name_norm", "proj_pg"]].rename(columns={"proj_pg": src})
                merged = p if merged is None else merged.merge(p, on="name_norm", how="inner")
            merged["proj_pg"] = merged[list(per_source)].mean(axis=1)
            merged["position"] = per_source[next(iter(per_source))].set_index("name_norm")["position"].reindex(merged["name_norm"]).values
            joined = _join(actual, merged)
            joined["source"] = "blend"
            joined["snapshot"] = "+".join(sorted(per_source))
            detail.append(joined)
        # ESPN's own published projection for this exact week (espn_history.py proj ...).
        # A different horizon from the ROS sources above -- it's a single-week number --
        # but it's what start/sit decisions actually use, so it's scored alongside.
        espn_file = ACTUALS_DIR / "espn_weekly_proj.csv"
        if espn_file.exists():
            ew = pd.read_csv(espn_file)
            ew = ew[ew["week"] == week][["espn_id", "espn_proj"]].rename(columns={"espn_proj": "proj_pg"})
            if not ew.empty:
                joined = actual.merge(ew, on="espn_id", how="inner")
                joined["source"] = "espn_weekly"
                joined["snapshot"] = espn_file.name
                detail.append(joined)
    if not detail:
        return {"detail": pd.DataFrame(), "summary": pd.DataFrame()}
    det = pd.concat(detail, ignore_index=True)
    det["error"] = det["proj_pg"] - det["points"]

    rows = []
    for (source, position), g in list(det.groupby(["source", "position"])) + [((s, "ALL"), g) for s, g in det.groupby("source")]:
        # rank correlation computed within each week & position, then averaged,
        # so it measures "who's better than whom", not level or week-to-week scoring swings
        corrs = [
            _rank_corr(gg["proj_pg"], gg["points"])
            for _, gg in g.groupby(["week", "position"]) if len(gg) >= 3
        ]
        corrs = [c for c in corrs if c == c]
        rows.append({
            "source": source,
            "position": position,
            "weeks": g["week"].nunique(),
            "n": len(g),
            "mae": round(g["error"].abs().mean(), 2),
            "bias": round(g["error"].mean(), 2),
            "rank_corr": round(sum(corrs) / len(corrs), 3) if corrs else float("nan"),
        })
    summary = pd.DataFrame(rows).sort_values(["position", "source"]).reset_index(drop=True)
    return {"detail": det, "summary": summary}
