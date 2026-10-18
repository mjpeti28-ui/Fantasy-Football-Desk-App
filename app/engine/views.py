"""JSON-ready views for each screen. Pure functions of `data.Inputs`; read-only."""
from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import data as d

STARTER_SLOTS = ["QB", "RB", "WR", "TE", "RB/WR", "WR/TE", "FLEX", "DST", "K"]
SLOT_ELIG = {
    "QB": {"QB"}, "RB": {"RB"}, "WR": {"WR"}, "TE": {"TE"}, "K": {"K"}, "DST": {"DST"},
    "RB/WR": {"RB", "WR"}, "WR/TE": {"WR", "TE"}, "FLEX": {"RB", "WR", "TE"},
}
FLEX_SLOTS = ["RB/WR", "WR/TE", "FLEX"]
POS_ORDER = {"QB": 0, "RB": 1, "WR": 2, "TE": 3, "K": 5, "DST": 4}


def _f(x) -> Optional[float]:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else round(x, 2)


def headshot(espn_id: int, pos: str, abbrev: Optional[str]) -> Optional[str]:
    if pos == "DST":
        return f"https://a.espncdn.com/i/teamlogos/nfl/500/{(abbrev or '').lower()}.png" if abbrev else None
    return f"https://a.espncdn.com/i/headshots/nfl/players/full/{int(espn_id)}.png"


def _pro_team(inp: d.Inputs) -> Dict[int, str]:
    """espn_id -> ESPN team abbrev, from the most recent week each player appears."""
    pw = inp.players_weekly.sort_values("week").drop_duplicates("espn_id", keep="last")
    ab = dict(zip(inp.pro_teams["pro_team_id"], inp.pro_teams["abbrev"]))
    return {int(i): ab.get(int(t)) for i, t in zip(pw["espn_id"], pw["pro_team_id"]) if pd.notna(t)}


def _nfl_game(inp: d.Inputs, abbrev: Optional[str], week: int) -> Optional[dict]:
    if not abbrev or abbrev == "FA":
        return None
    t = d.ESPN_TO_NFLVERSE.get(abbrev, abbrev)
    g = inp.nfl_sched[(inp.nfl_sched["week"] == week) & ((inp.nfl_sched["home_team"] == t) | (inp.nfl_sched["away_team"] == t))]
    if g.empty:
        return {"bye": True}
    g = g.iloc[0]
    home = g["home_team"] == t
    opp = g["away_team"] if home else g["home_team"]
    spread, total = _f(g.get("spread_line")), _f(g.get("total_line"))
    implied = None
    if spread is not None and total is not None:  # nflverse spread_line > 0 means home favored
        implied = round((total + spread) / 2 if home else (total - spread) / 2, 1)
    return {"bye": False, "opp": opp, "home": bool(home), "day": str(g.get("weekday", ""))[:3],
            "date": str(g.get("gameday", "")), "time": str(g.get("gametime", "")),
            "implied_total": implied, "game_total": total}


def _range(inp: d.Inputs, pos: str, center: Optional[float]) -> Optional[List[float]]:
    """80% range for a single week: ESPN's number times the empirical actual/forecast ratio
    spread for the position (2021-25, weeks 3-14; QB/RB/WR/TE measured against ESPN's number,
    tools/ratio_dist_espn.py; K/DST still against our forecast)."""
    r = inp.ratios.get(pos)
    if center is None or r is None or len(r) == 0 or center <= 0:
        return None
    lo, hi = np.quantile(r, [0.1, 0.9])
    return [round(center * lo, 1), round(center * hi, 1)]


def player_rows(inp: d.Inputs, team_id: int, week: Optional[int] = None) -> List[dict]:
    week = week or inp.week
    r = inp.rosters[(inp.rosters["week"] == week) & (inp.rosters["team_id"] == team_id)]
    if r.empty:  # rosters for the new week not pulled yet: fall back to the last known week
        last = inp.rosters[inp.rosters["team_id"] == team_id]["week"].max()
        r = inp.rosters[(inp.rosters["week"] == last) & (inp.rosters["team_id"] == team_id)]
    lp = inp.ledger_players.set_index("espn_id")
    pr = inp.espn_proj[inp.espn_proj["week"] == week].set_index("espn_id")["espn_proj"]
    teams = _pro_team(inp)
    pw = inp.players_weekly[inp.players_weekly["week"] < week]
    rows = []
    for _, x in r.iterrows():
        pid, pos = int(x["espn_id"]), x["position"]
        L = lp.loc[pid] if pid in lp.index else None
        ab = teams.get(pid)
        espn = _f(pr.get(pid))
        hist = pw[pw["espn_id"] == pid].set_index("week")
        weekly = [(_f(hist.at[w, "points"]) if w in hist.index and bool(hist.at[w, "played"]) else None)
                  for w in range(1, week)]
        played = [p for p in weekly if p is not None]
        rows.append({
            "id": pid, "name": x["name"], "pos": pos, "team": ab,
            "headshot": headshot(pid, pos, ab),
            "slot": x["lineup_slot"], "injury": (x["injury_status"] if pd.notna(x.get("injury_status")) else "ACTIVE"),
            "espn_week": espn, "week_range": _range(inp, pos, espn),
            "est_ppg": _f(L["est_ppg"]) if L is not None else None,
            "avail_now": _f(L["avail_now"]) if L is not None else None,
            "avail_ros": _f(L["avail"]) if L is not None else None,
            "ecr_rank": _f(L["ecr_rank"]) if L is not None else None,
            "status": (L["status"] if L is not None else None),
            "bye": int(L["bye"]) if L is not None and pd.notna(L["bye"]) else None,
            "game": _nfl_game(inp, ab, week),
            "weekly": weekly, "ppg": round(sum(played) / len(played), 1) if played else None,
            "games": len(played),
        })
    rows.sort(key=lambda z: (POS_ORDER.get(z["pos"], 9), -(z["espn_week"] or 0)))
    return rows


def best_lineup(rows: List[dict], key: str = "espn_week") -> Dict[str, int]:
    """Exact best lineup by `key`: fixed slots greedily (they don't compete), then exhaustive
    over the three flex slots among the top remaining candidates."""
    val = {z["id"]: (z.get(key) or 0.0) for z in rows}
    pos = {z["id"]: z["pos"] for z in rows}
    pool = sorted(val, key=lambda i: -val[i])
    out, used = {}, set()
    for s in ["QB", "K", "DST", "RB", "WR", "TE"]:
        c = next((i for i in pool if i not in used and pos[i] in SLOT_ELIG[s]), None)
        if c is not None:
            out[s], used = c, used | {c}
    cand = [i for i in pool if i not in used and pos[i] in {"RB", "WR", "TE"}][:9]
    best, best_v = None, -1.0
    for combo in itertools.permutations(cand, min(3, len(cand))):
        if all(pos[i] in SLOT_ELIG[s] for i, s in zip(combo, FLEX_SLOTS)):
            v_ = sum(val[i] for i in combo)
            if v_ > best_v:
                best, best_v = combo, v_
    for i, s in zip(best or (), FLEX_SLOTS):
        out[s] = i
    return out


def _lineup_points(rows: List[dict], slots: Dict[str, int], key="espn_week") -> float:
    by = {z["id"]: z for z in rows}
    return round(sum((by[i].get(key) or 0.0) for i in slots.values()), 1)


def _opponent(inp: d.Inputs, team_id: int, week: int) -> Optional[int]:
    names = {inp.team_name(int(t)): int(t) for t in inp.settings.get("team_names", {})}
    s = inp.league_sched[inp.league_sched["Week"] == week]
    me = inp.team_name(team_id)
    for _, g in s.iterrows():
        a, h = " ".join(str(g["AwayTeam"]).split()), " ".join(str(g["HomeTeam"]).split())
        if me in (a, h):
            return names.get(h if a == me else a)
    return None


def team_week(inp: d.Inputs, team_id: int = 3) -> dict:
    week = inp.week
    rows = player_rows(inp, team_id, week)
    current = {z["slot"]: z["id"] for z in rows if z["slot"] in STARTER_SLOTS}
    suggested = best_lineup(rows)
    cur_pts, sug_pts = _lineup_points(rows, current), _lineup_points(rows, suggested)
    by = {z["id"]: z for z in rows}
    swaps = []
    ins = [i for i in suggested.values() if i not in current.values()]
    outs = [i for i in current.values() if i not in suggested.values()]
    for i, o in zip(sorted(ins, key=lambda i: -(by[i]["espn_week"] or 0)), sorted(outs, key=lambda i: (by[i]["espn_week"] or 0))):
        swaps.append({"start": by[i]["name"], "bench": by[o]["name"],
                      "gain": round((by[i]["espn_week"] or 0) - (by[o]["espn_week"] or 0), 1)})
    opp_id = _opponent(inp, team_id, week)
    matchup = None
    sd = d.TEAM_WEEK_SD * math.sqrt(2)
    if opp_id is not None:
        orows = player_rows(inp, opp_id, week)
        ocur = {z["slot"]: z["id"] for z in orows if z["slot"] in STARTER_SLOTS}
        obest = _lineup_points(orows, best_lineup(orows))
        opts = _lineup_points(orows, ocur) or obest
        my = sug_pts
        win = lambda o: round(0.5 * (1 + math.erf((my - o) / (sd * math.sqrt(2)))), 3)
        oby = {z["id"]: z for z in orows}
        holes = [oby[i]["name"] for s, i in ocur.items() if not (oby[i]["espn_week"] or 0)]
        matchup = {"opp_id": opp_id, "opp_name": inp.team_name(opp_id), "my_proj": my, "opp_proj": opts,
                   "p_win": win(opts), "margin_sd": round(sd, 1),
                   # their best lineup from their own roster: the win chance if they fix empty or 0-point slots
                   "opp_best": obest, "p_win_best": win(obest), "opp_holes": holes,
                   "opp_starters": [{"id": z["id"], "name": z["name"], "pos": z["pos"], "team": z["team"],
                                     "headshot": z["headshot"], "espn_week": z["espn_week"], "slot": z["slot"],
                                     "game": z["game"], "injury": z["injury"]}
                                    for z in orows if z["slot"] in STARTER_SLOTS]}
    return {
        "season": inp.season, "week": week, "team_id": team_id, "team_name": inp.team_name(team_id),
        "players": rows, "lineup": {"current": current, "suggested": suggested,
                                     "current_pts": cur_pts, "suggested_pts": sug_pts, "swaps": swaps},
        "matchup": matchup,
        "notes": {
            "lineup_rule": "Suggested lineup = highest ESPN weekly projection per slot; in 1,020 past team-weeks "
                           "this beat managers by about 1.3 pts/week. Worth about a point a week.",
            "range": "80% range = ESPN number times the 2021-25 spread of actual/forecast by position.",
            "p_win": f"Normal margin, sd {sd:.0f} (team weekly sd {d.TEAM_WEEK_SD} each side).",
        },
    }


def _records(inp: d.Inputs) -> Dict[int, dict]:
    tw = inp.team_weekly
    pts = {(int(w), int(t)): float(p) for w, t, p in zip(tw["week"], tw["team_id"], tw["points"])}
    names = {inp.team_name(int(t)): int(t) for t in inp.settings.get("team_names", {})}
    rec = {int(t): {"w": 0, "l": 0, "t": 0, "pa": 0.0} for t in inp.settings.get("team_names", {})}
    for _, g in inp.league_sched.iterrows():
        w = int(g["Week"])
        a, h = names.get(" ".join(str(g["AwayTeam"]).split())), names.get(" ".join(str(g["HomeTeam"]).split()))
        if a is None or h is None or (w, a) not in pts or (w, h) not in pts:
            continue
        pa_, ph = pts[(w, a)], pts[(w, h)]
        rec[a]["pa"] += ph
        rec[h]["pa"] += pa_
        for t, mine, theirs in ((a, pa_, ph), (h, ph, pa_)):
            rec[t]["w" if mine > theirs else "l" if mine < theirs else "t"] += 1
    return rec


def power_table(inp: d.Inputs) -> dict:
    tw = inp.team_weekly[inp.team_weekly["week"] < inp.week]
    weeks_played = sorted(tw["week"].unique())
    r = inp.rosters[inp.rosters["week"].isin(weeks_played) & inp.rosters["lineup_slot"].isin(STARTER_SLOTS)]
    pr = inp.espn_proj.rename(columns={"espn_proj": "p"})
    pj_w = r.merge(pr, on=["week", "espn_id"], how="left").groupby(["team_id", "week"])["p"].sum()
    proj = pj_w.groupby("team_id").sum()
    rec = _records(inp)
    latest_w = max(inp.ledger_teams)
    lt = inp.ledger_teams[latest_w].set_index("team_id")
    hist = {w: df.set_index("team_id") for w, df in inp.ledger_teams.items()}
    teams = []
    for t in sorted(int(x) for x in inp.settings.get("team_names", {})):
        g = tw[tw["team_id"] == t].sort_values("week")
        pf = float(g["points"].sum())
        pj = float(proj.get(t, np.nan))
        L = lt.loc[t] if t in lt.index else None
        teams.append({
            "id": t, "name": inp.team_name(t), "record": rec.get(t),
            "pf": round(pf, 1), "pa": round(rec.get(t, {}).get("pa", 0.0), 1),
            "weekly": [round(float(p), 1) for p in g["points"]],
            "proj_pf": _f(pj), "luck": _f(pf - pj),
            "weekly_proj": [_f(pj_w.get((t, w))) for w in g["week"]],
            "ros_value": _f(L["ros_value"]) if L is not None else None,
            "proj_final_pts": _f(L["reg_pts"]) if L is not None else None,
            "odds": {k: _f(L[k]) for k in ("seed1", "top2", "playoffs", "title")} if L is not None else None,
            "odds_history": [{"week": w, **{k: _f(h.at[t, k]) for k in ("seed1", "top2", "playoffs", "title")}}
                             for w, h in hist.items() if t in h.index],
        })
    n = len(weeks_played)
    pfs = np.array([z["pf"] for z in teams])
    for z in teams:
        z["pf_rank"] = int(1 + (pfs > z["pf"]).sum())
    teams.sort(key=lambda z: -(z["proj_final_pts"] or z["pf"]))
    for k, z in enumerate(teams):
        z["proj_rank"] = k + 1
    return {
        "season": inp.season, "through_week": max(weeks_played) if weeks_played else 0, "weeks_played": n,
        "odds_as_of_week": latest_w, "teams": teams,
        "notes": {
            "seeding": "8 of 10 make the playoffs, seeded by total points; record doesn't seed.",
            "luck": "Luck = actual points minus ESPN's pre-kickoff projection for the lineup actually started. "
                    "In 2021-25, early luck had ~0 carry-over to later weeks.",
            "odds": "Seed/playoff odds carry signal from about week 6; title odds are close to a lottery (Brier skill <= .07).",
        },
    }


def ratio_quantiles(inp: d.Inputs, n: int = 50) -> Dict[str, List[float]]:
    """Per-position quantiles of actual / forecast points (2021-25, weeks 3-14). The UI scales
    them by a player's projection to compare two players' weekly outcomes."""
    qs = np.linspace(0.01, 0.99, n)
    return {pos: [round(float(x), 4) for x in np.quantile(r, qs)] for pos, r in inp.ratios.items() if len(r)}
