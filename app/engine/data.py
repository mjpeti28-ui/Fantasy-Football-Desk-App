"""Cached inputs for the app's views. Reads files the weekly flow already writes; no network,
no simulation, so every call is fast enough to run on app open."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ACTUALS = ROOT / "history" / "actuals"
LEDGER = ROOT / "history" / "ledger"
NFLVERSE = ROOT / "history" / "nflverse"
RATIOS = ROOT / "history" / "backtest" / "multi" / "ratio_dist.json"

# ESPN abbreviations -> nflverse abbreviations (only the ones that differ)
ESPN_TO_NFLVERSE = {"LAR": "LA", "WSH": "WAS"}

# Weekly team score sd about the team's own season mean, 2021-25 (team-outlook analysis,
# 2026-10-06). Used for the matchup margin; it already includes projection error.
TEAM_WEEK_SD = 22.6


def _mtime(p: Path) -> Optional[float]:
    return p.stat().st_mtime if p.exists() else None


@dataclass
class Inputs:
    season: int
    week: int                      # the upcoming (unplayed) fantasy week
    settings: dict
    rosters: pd.DataFrame          # rosters_by_week
    players_weekly: pd.DataFrame
    team_weekly: pd.DataFrame
    espn_proj: pd.DataFrame        # week, espn_id, espn_proj
    ledger_players: pd.DataFrame   # latest wkNN_players.csv
    ledger_teams: Dict[int, pd.DataFrame]  # week -> wkNN_teams.csv
    pro_teams: pd.DataFrame        # pro_team_id, abbrev, bye_week
    nfl_sched: pd.DataFrame        # nflverse schedule for the season
    league_sched: pd.DataFrame     # fantasy matchups (schedule.csv)
    ids: pd.DataFrame              # nflverse id map
    ratios: Dict[str, np.ndarray]
    ages: Dict[str, Optional[float]] = field(default_factory=dict)

    def team_name(self, team_id: int) -> str:
        return " ".join(self.settings.get("team_names", {}).get(str(team_id), f"Team {team_id}").split())


def _ledger_weeks(season: int) -> Dict[int, Path]:
    out = {}
    for p in (LEDGER / str(season)).glob("wk*_teams.csv"):
        m = re.match(r"wk(\d+)_teams", p.stem)
        if m:
            out[int(m.group(1))] = p
    return out


@lru_cache(maxsize=2)
def load(season: int = 2026) -> Inputs:
    a = ACTUALS / str(season)
    settings = json.loads((a / "settings.json").read_text())
    tw = pd.read_csv(a / "team_weekly.csv")
    proj = pd.read_csv(a / "espn_weekly_proj.csv")
    lw = _ledger_weeks(season)
    latest = max(lw)
    tw = tw[tw["points"] > 0]  # team_weekly lists future weeks with 0 points
    played_weeks = set(int(w) for w in tw["week"].unique())
    week = latest if latest not in played_weeks else max(played_weeks) + 1
    nfl = pd.read_csv(NFLVERSE / "schedules.csv")
    ratios = {k: np.asarray(v_, float) for k, v_ in json.loads(RATIOS.read_text()).items()} if RATIOS.exists() else {}
    ages = {
        "rosters": _mtime(a / "rosters_by_week.csv"),
        "espn_proj": _mtime(a / "espn_weekly_proj.csv"),
        "ledger": _mtime(lw[latest]),
        "scores": _mtime(a / "team_weekly.csv"),
    }
    return Inputs(
        season=season, week=int(week), settings=settings,
        rosters=pd.read_csv(a / "rosters_by_week.csv"),
        players_weekly=pd.read_csv(a / "players_weekly.csv"),
        team_weekly=tw, espn_proj=proj,
        ledger_players=pd.read_csv(LEDGER / str(season) / f"wk{latest:02d}_players.csv"),
        ledger_teams={w: pd.read_csv(p) for w, p in sorted(lw.items())},
        pro_teams=pd.read_csv(a / "byes.csv"),
        nfl_sched=nfl[nfl["season"] == season].copy(),
        league_sched=pd.read_csv(ROOT / "schedule.csv"),
        ids=pd.read_csv(NFLVERSE / "ids.csv"),
        ratios=ratios, ages=ages,
    )


def ages_key() -> tuple:
    """Changes whenever an input file the views read is rewritten (used to invalidate caches)."""
    return tuple(sorted((k, v) for k, v in load().ages.items()))
