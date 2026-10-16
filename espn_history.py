"""Cache this league's past seasons from ESPN for backtesting.

Writes history/actuals/<season>/:
- players_weekly.csv  every player's weekly fantasy points under that season's league
                      scoring (week>=1), plus ESPN's season projection as week 0
                      (``kind=espn_season_proj``; it is stored static -- it did not move
                      during 2025 -- so it is treated as preseason, with that caveat)
- rosters_by_week.csv every team's roster each week (lineup slot, injury status)
- team_weekly.csv     each team's actual score each week
- draft.csv           draft picks (round, pick, team, player) -- a no-lookahead prior
- byes.csv            NFL bye week per pro team
- trades.csv          executed trades
- settings.json       team count, lineup slot counts, regular-season length

League history (ESPN): 2021-2025 were 12 teams; 2026 is 10 teams. Lineups
changed too (2021 used 2 RB / 2 WR / 1 FLEX; 2022+ uses RB, WR, TE, RB/WR,
WR/TE, FLEX), so always read settings.json rather than assuming.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import httpx
import pandas as pd

import espn_client as ec

ROOT = Path(__file__).resolve().parent
POS = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "DST"}
SLOT_NAMES = {0: "QB", 2: "RB", 3: "RB/WR", 4: "WR", 5: "WR/TE", 6: "TE", 16: "DST", 17: "K", 20: "BE", 21: "IR", 23: "FLEX"}


def out_dir(season: int) -> Path:
    d = ROOT / "history" / "actuals" / str(season)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _client(season: int) -> ec.ESPNClient:
    return ec.ESPNClient(dataclasses.replace(ec.ESPNConfig.from_environment(), season=season))


def fetch_players(season: int) -> pd.DataFrame:
    c = _client(season)
    flt = {"players": {"limit": 2500, "sortPercOwned": {"sortPriority": 1, "sortAsc": False},
                       "filterStatsForTopScoringPeriodIds": {"value": 18, "additionalValue": [f"00{season}", f"10{season}"]}}}
    k = c.fetch_endpoint(view=["kona_player_info"], headers={"X-Fantasy-Filter": json.dumps(flt)})
    rows = []
    for ent in k.get("players", []):
        x = ent["player"]
        pos = POS.get(x.get("defaultPositionId"))
        if not pos:
            continue
        base = {"espn_id": x["id"], "name": x["fullName"], "position": pos, "pro_team_id": x.get("proTeamId")}
        for s in x.get("stats", []):
            if s.get("statSourceId") == 0 and s.get("statSplitTypeId") == 1 and s.get("seasonId", season) == season:
                rows.append({**base, "week": s["scoringPeriodId"], "points": float(s.get("appliedTotal") or 0.0),
                             "played": bool(s.get("stats")), "kind": "actual"})
            elif s.get("id") == f"10{season}":
                rows.append({**base, "week": 0, "points": float(s.get("appliedTotal") or 0.0), "played": None,
                             "kind": "espn_season_proj"})
    return pd.DataFrame(rows)


def fetch_weekly_projections(season: int, weeks) -> pd.DataFrame:
    """ESPN's own projection for each player for each week, as published for that
    week (stat id ``11<season><week>``). One call per week; this is what managers saw
    in the app, so it is also our proxy for how the other side values players."""
    c = _client(season)
    rows = []
    for w in weeks:
        flt = {"players": {"limit": 2500, "sortPercOwned": {"sortPriority": 1, "sortAsc": False}}}
        k = c.fetch_endpoint(view=["kona_player_info"], params={"scoringPeriodId": w},
                             headers={"X-Fantasy-Filter": json.dumps(flt)})
        want = f"11{season}{w}"
        for ent in k.get("players", []):
            x = ent["player"]
            for st in x.get("stats", []):
                if st.get("id") == want:
                    rows.append({"week": w, "espn_id": x["id"], "espn_proj": float(st.get("appliedTotal") or 0.0)})
        time.sleep(0.3)
    return pd.DataFrame(rows)


def build_weekly_projections(season: int, weeks) -> Path:
    d = out_dir(season)
    path = d / "espn_weekly_proj.csv"
    new = fetch_weekly_projections(season, weeks)
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[~old["week"].isin(new["week"].unique())], new], ignore_index=True)
    new.sort_values(["week", "espn_id"]).to_csv(path, index=False)
    return path


def fetch_league(season: int, last_week: int = 17) -> Dict[str, pd.DataFrame]:
    c = _client(season)
    rosters, sched, trades = [], {}, []
    current = None
    for w in range(1, last_week + 1):
        p = c.fetch_endpoint(view=["mRoster", "mTeam", "mMatchupScore", "mTransactions2"], params={"scoringPeriodId": w})
        if current is None:
            st = p.get("status", {}) or {}
            current = int(st.get("latestScoringPeriod") or st.get("currentMatchupPeriod") or last_week)
        # ESPN (since ~wk3 2026) returns today's roster for future scoring periods too; those
        # aren't history and make "latest week" = 17 downstream, so stop at the current period.
        if w > current:
            break
        for t in p.get("teams", []):
            for en in t.get("roster", {}).get("entries", []):
                pl = en["playerPoolEntry"]["player"]
                rosters.append({"week": w, "team_id": t["id"], "espn_id": pl["id"], "name": pl["fullName"],
                                "position": POS.get(pl.get("defaultPositionId")),
                                "lineup_slot_id": en.get("lineupSlotId"),
                                "lineup_slot": SLOT_NAMES.get(en.get("lineupSlotId")),
                                "injury_status": pl.get("injuryStatus")})
        for m in p.get("schedule", []):
            for side in ("home", "away"):
                s = m.get(side)
                if s and s.get("totalPoints") is not None:
                    sched[(m["matchupPeriodId"], s["teamId"])] = {
                        "week": m["matchupPeriodId"], "team_id": s["teamId"], "points": s["totalPoints"],
                        "playoff": m.get("playoffTierType") not in (None, "NONE")}
        for tx in p.get("transactions", []) or []:
            if tx.get("type") == "TRADE_ACCEPT" and tx.get("status") == "EXECUTED":
                for it in tx.get("items", []):
                    trades.append({"week": w, "tx_id": tx.get("id"), "espn_id": it.get("playerId"),
                                   "from_team": it.get("fromTeamId"), "to_team": it.get("toTeamId")})
        time.sleep(0.3)
    season_payload = c.fetch_endpoint(view=["mDraftDetail", "mTeam", "mSettings"])
    draft = pd.DataFrame([
        {"round": pk.get("roundId"), "pick": pk.get("overallPickNumber"), "team_id": pk.get("teamId"), "espn_id": pk.get("playerId"),
         "keeper": pk.get("keeper")}
        for pk in season_payload.get("draftDetail", {}).get("picks", [])
    ])
    s = season_payload["settings"]
    settings = {
        "season": season,
        "teams": len(season_payload.get("teams", [])),
        "team_names": {str(t["id"]): (t.get("name") or "").strip() for t in season_payload.get("teams", [])},
        "lineup_slot_counts": {SLOT_NAMES.get(int(k), k): v for k, v in s["rosterSettings"]["lineupSlotCounts"].items() if v},
        "regular_season_weeks": s["scheduleSettings"]["matchupPeriodCount"],
        "playoff_teams": s["scheduleSettings"]["playoffTeamCount"],
        "seeding": s["scheduleSettings"].get("playoffSeedingRule"),
    }
    byes = httpx.get(f"{ec.ESPN_FANTASY_BASE}/ffl/seasons/{season}", params={"view": "proTeamSchedules_wl"}, timeout=30).json()
    byes_df = pd.DataFrame([{"pro_team_id": t["id"], "abbrev": t.get("abbrev"), "bye_week": t.get("byeWeek")}
                            for t in byes.get("settings", {}).get("proTeams", [])])
    return {"rosters_by_week": pd.DataFrame(rosters), "team_weekly": pd.DataFrame(list(sched.values())),
            "trades": pd.DataFrame(trades), "draft": draft, "byes": byes_df, "settings": settings}


def build_season(season: int, last_week: int = 17) -> None:
    d = out_dir(season)
    fetch_players(season).to_csv(d / "players_weekly.csv", index=False)
    lg = fetch_league(season, last_week=last_week)
    for name in ("rosters_by_week", "team_weekly", "trades", "draft", "byes"):
        lg[name].to_csv(d / f"{name}.csv", index=False)
    (d / "settings.json").write_text(json.dumps(lg["settings"], indent=2))


if __name__ == "__main__":
    import sys

    from env import load_local_env

    load_local_env()
    args = sys.argv[1:]
    if args and args[0] == "proj":  # python espn_history.py proj 2024 1 9  -> weeks 1..9
        yr, lo, hi = int(args[1]), int(args[2]), int(args[3])
        t0 = time.time()
        build_weekly_projections(yr, range(lo, hi + 1))
        print(f"{yr} weekly projections {lo}-{hi}: done in {time.time() - t0:.0f}s")
    else:
        for yr in [int(a) for a in args]:
            t0 = time.time()
            build_season(yr)
            print(f"{yr}: done in {time.time() - t0:.0f}s")
