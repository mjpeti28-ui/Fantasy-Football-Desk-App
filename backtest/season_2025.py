"""Backtest the evaluation stack against the 2025 season (this league, 12 teams).

Data (all built from ESPN + archived PFF files, see backtest/README in the
project guide):
- history/projections/2025/*_projections.csv -- 11 in-season PFF ROS projection
  snapshots (Sep 24 -> Nov 5), each with a ``games`` (remaining) column.
- history/actuals/2025/espn_players_weekly.csv -- every player's weekly fantasy
  points under this league's scoring (ESPN), weeks 1-18.
- history/actuals/2025/rosters_by_week.csv -- every team's roster each week.
- history/actuals/2025/team_weekly.csv -- each team's actual weekly score.

Checkpoints: the last snapshot taken before each week's Thursday kickoff gives
a model state at weeks 4-10. At each checkpoint we run the real evaluation
(``main.evaluate_league``) on that week's actual rosters and compare what it
predicted to what happened afterward.

Rankings: the archived 2025 ranking files are a stale draft board and a week-2
ROS board that only cover ~190-240 players, and ``data.load_rankings`` drops
anyone not ranked. So each checkpoint's rankings are synthesized from that
snapshot's own projections (rank by projected points). The model runs on
projected points anyway; this isolates the projection + VOR logic, which is
what's under test.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import normalize_name  # noqa: E402

SEASON = 2025
WEEK1_THURSDAY = date(2025, 9, 4)
REG_SEASON_LAST_WEEK = 14
FANTASY_LAST_WEEK = 17
ACT = ROOT / "history" / "actuals" / "2025"
PROJ_DIR = ROOT / "history" / "projections" / "2025"
OUT = ROOT / "history" / "backtest" / "2025"
SKILL = ["QB", "RB", "WR", "TE"]


# ------------------------------------------------------------------ inputs


def week_cutoff(week: int) -> datetime:
    thu = WEEK1_THURSDAY + timedelta(weeks=week - 1)
    return datetime(thu.year, thu.month, thu.day, tzinfo=timezone.utc) + timedelta(days=1)


def checkpoints() -> List[dict]:
    """[{week, snapshot}] using the latest snapshot before each week's kickoff."""
    snaps = []
    for f in sorted(PROJ_DIR.glob("*_projections.csv")):
        # archive stamps are naive US-Eastern local time
        ts = datetime.strptime(f.name[:15], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc) + timedelta(hours=4)
        snaps.append((ts, f))
    out = []
    for week in range(1, FANTASY_LAST_WEEK + 1):
        prior = [(ts, f) for ts, f in snaps if ts < week_cutoff(week)]
        if not prior:
            continue
        ts, f = prior[-1]
        # Guard: a "projection" whose games column exceeds the games actually left is a
        # full-season export (the Nov 5 2025 archive is one), not rest-of-season.
        games = pd.read_csv(f, usecols=lambda c: c == "games")
        if not games.empty and games["games"].max() > 18 - week + 1:
            continue
        # only use a snapshot for the week immediately after it was taken
        if week_cutoff(week - 1) <= ts if week > 1 else True:
            out.append({"week": week, "snapshot": f, "taken_at": ts})
    return out


def load_weekly() -> pd.DataFrame:
    df = pd.read_csv(ACT / "espn_players_weekly.csv")
    return df[df["week"] > 0].copy()


def rosters_at(week: int) -> pd.DataFrame:
    r = pd.read_csv(ACT / "rosters_by_week.csv")
    return r[r["week"] == week].copy()


def team_weekly() -> pd.DataFrame:
    return pd.read_csv(ACT / "team_weekly.csv")


def synth_rankings(snapshot: Path, dest: Path) -> Path:
    p = pd.read_csv(snapshot)
    p = p.sort_values("fantasyPoints", ascending=False).reset_index(drop=True)
    pos = p["position"].astype(str).str.upper().replace({"DST": "DST", "D/ST": "DST"})
    out = pd.DataFrame({
        "Rank": np.arange(1, len(p) + 1),
        "Position": pos,
        "Name": p["playerName"],
        "Team": p["teamName"],
    })
    out.to_csv(dest, index=False)
    return dest


def custom_rosters(roster_rows: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
    rosters: Dict[str, Dict[str, List[str]]] = {}
    for _, r in roster_rows.iterrows():
        if r["position"] not in ("QB", "RB", "WR", "TE", "K", "DST"):
            continue
        team = f"T{int(r['team_id'])}"
        label = r["name"] + (" (IR)" if r["lineup_slot_id"] == 21 else "")
        rosters.setdefault(team, {p: [] for p in ("QB", "RB", "WR", "TE", "DST", "K")})[r["position"]].append(label)
    return rosters


def realized_ros(weekly: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    w = weekly[(weekly["week"] >= start) & (weekly["week"] <= end)]
    agg = w.groupby(["espn_id", "name", "position"]).agg(
        ros_points=("points", "sum"), ros_games=("played", "sum")
    ).reset_index()
    agg["name_norm"] = agg["name"].apply(normalize_name)
    return agg


# ------------------------------------------------------------------ one checkpoint


def run_checkpoint(cp: dict) -> Dict[str, pd.DataFrame]:
    import main

    week = cp["week"]
    OUT.mkdir(parents=True, exist_ok=True)
    rk = synth_rankings(cp["snapshot"], OUT / f"_rankings_w{week}.csv")
    rows = rosters_at(week)
    res = main.evaluate_league(str(rk), projections_path=str(cp["snapshot"]),
                               custom_rosters=custom_rosters(rows), save_snapshot=False)
    df = res["df"]
    weekly = load_weekly()
    ros = realized_ros(weekly, week, FANTASY_LAST_WEEK)

    # --- player level: every rostered skill player with model VOR
    name_to_id = {n: int(i) for n, i in zip(rows["name"], rows["espn_id"])}
    team_of = {n: f"T{int(t)}" for n, t in zip(rows["name"], rows["team_id"])}
    prow = []
    for team, r in res["results"].items():
        for p in r["starters"]:
            prow.append({"team": team, "name": p["name"], "pos": p["pos"], "proj": p.get("proj"), "vor": p.get("vor"), "role": "starter"})
    for team, tbl in res["bench_tables"].items():
        for p in (tbl if isinstance(tbl, list) else tbl.to_dict("records")):
            prow.append({"team": team, "name": p["name"], "pos": p["pos"], "proj": p.get("proj"), "vor": p.get("vor"), "role": "bench"})
    players = pd.DataFrame(prow)
    players = players[players["pos"].isin(SKILL)].copy()
    players["name"] = players["name"].str.replace(" (IR)", "", regex=False)
    players["espn_id"] = players["name"].map(name_to_id)
    players = players.merge(ros[["espn_id", "ros_points", "ros_games"]], on="espn_id", how="left")
    players["ros_points"] = players["ros_points"].fillna(0.0)
    rank_of = df.set_index("Name")["RankOriginal"].to_dict() if "RankOriginal" in df else {}

    # --- waiver pool at this checkpoint: unrostered players with a real projection
    rostered_norm = {normalize_name(n) for n in rows["name"]}
    pool = df[(df["Position"].isin(SKILL)) & (df.get("ProjSource", "exact") == "exact")].copy()
    pool = pool[~pool["name_norm"].isin(rostered_norm)]
    pool = pool.merge(ros[["name_norm", "position", "ros_points"]].rename(columns={"position": "Position"}),
                      on=["name_norm", "Position"], how="left")
    pool["espn_matched"] = pool["ros_points"].notna()
    pool["ros_points"] = pool["ros_points"].fillna(0.0)
    waiver = {}
    for pos in SKILL:
        # "who you'd realistically pick up": the 5 best unrostered by projection at the time
        top = pool[(pool["Position"] == pos) & pool["espn_matched"]].sort_values("ProjPoints", ascending=False).head(5)
        waiver[pos] = {
            "proj": float(top["ProjPoints"].mean()) if len(top) else 0.0,
            "realized": float(top["ros_points"].mean()) if len(top) else 0.0,
            "names": ", ".join(top["Name"]),
        }
    players["model_repl"] = players["pos"].map(res["replacement_points"])
    players["waiver_repl_proj"] = players["pos"].map({p: v["proj"] for p, v in waiver.items()})
    players["waiver_repl_real"] = players["pos"].map({p: v["realized"] for p, v in waiver.items()})
    players["realized_vor"] = players["ros_points"] - players["waiver_repl_real"]
    players["vor_waiver"] = players["proj"] - players["waiver_repl_proj"]
    players["week"] = week

    # --- team level
    tw = team_weekly()
    ros_team = tw[(tw["week"] >= week) & (tw["week"] <= REG_SEASON_LAST_WEEK)].groupby("team_id")["points"].sum()
    todate = tw[tw["week"] < week].groupby("team_id")["points"].mean()
    teams = pd.DataFrame({
        "team": list(res["starters_totals"].keys()),
    })
    teams["team_id"] = teams["team"].str[1:].astype(int)
    teams["starter_vor"] = teams["team"].map(res["starters_totals"])
    teams["bench_total"] = teams["team"].map(res["bench_totals"])
    teams["starter_proj"] = teams["team"].map({t: r.get("StarterProjection") for t, r in res["results"].items()})
    teams["combined"] = teams["team"].map(res["combined_scores"])
    teams["ppg_to_date"] = teams["team_id"].map(todate)
    teams["ros_actual"] = teams["team_id"].map(ros_team)
    teams["week"] = week

    repl = pd.DataFrame([
        {"week": week, "pos": p, "model_repl_proj": res["replacement_points"].get(p),
         "waiver_repl_proj": waiver[p]["proj"], "waiver_repl_realized": waiver[p]["realized"],
         "waiver_top3": waiver[p]["names"]}
        for p in SKILL
    ])
    return {"players": players, "teams": teams, "repl": repl}


def run_all(weeks: Optional[List[int]] = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for cp in checkpoints():
        if weeks and cp["week"] not in weeks:
            continue
        out = run_checkpoint(cp)
        for k, v in out.items():
            v.to_csv(OUT / f"{k}_w{cp['week']}.csv", index=False)
        print(f"week {cp['week']}: {cp['snapshot'].name} -> {len(out['players'])} players, {len(out['teams'])} teams")


if __name__ == "__main__":
    weeks = [int(a) for a in sys.argv[1:]] or None
    run_all(weeks)
