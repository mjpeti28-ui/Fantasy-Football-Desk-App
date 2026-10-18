"""Draft recap for Desk's League > Draft tab.

Value is points over replacement (VOR: points minus games played x the per-game rate of the first non-starter
at the position), so a late QB's raw points don't read as a steal. Points so far are scored against the VOR that
draft slot usually returns in this league by the same week (2021-25 drafts). Rest of season is our ROS
forecast's VOR against this year's picks around the same slot: the forecast runs ~20% under what
drafted players historically scored after week 5, so comparing it to history would mark every pick down. The market view is the preseason PFF board's ADP (the only
draft-day ADP Desk has saved). A history block says how much a draft grade this early tends to hold,
and whether draft results repeat year to year, so luck vs skill is labelled from data, not vibes.
Read-only: reads files the weekly flow and Desk's refresh already write.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .analytics import ACT, LAST, LEDGER, ROOT, _r, complete_week, n_teams, tiers

HIST = range(2021, 2026)
WIN = 7          # +/- picks pooled for a slot's expected points
BOARD = ROOT / "PFF_rankings.csv"
ADP_CLIP = 30    # picks; late-round ADPs run past the draft, so team averages clip each pick's gap


def _norm(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]+", "", str(s).lower().replace("-", " "))
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s.strip())
    return re.sub(r"\s+", " ", s)


def _season_pts(season: int) -> pd.DataFrame:
    P = pd.read_csv(ACT / str(season) / "players_weekly.csv")
    return P[(P["kind"] == "actual") & P["week"].between(1, LAST)]


def _starters(n: int) -> Dict[str, int]:
    return {**tiers(n)["start"], "K": n, "DST": n}


def _repl(ppg: pd.Series, pos: pd.Series, n: int) -> Dict[str, float]:
    """Replacement level per position: points per game of the first player past the league's starters."""
    out = {}
    for p, k in _starters(n).items():
        v = ppg[pos.reindex(ppg.index) == p].sort_values(ascending=False)
        out[p] = float(v.iloc[k]) if len(v) > k else 0.0
    return out


def _vor(season: int, P: pd.DataFrame, through: int) -> pd.DataFrame:
    """Per player through week ``through``: points, games, and VOR = points - games x replacement per
    game (a missed game is filled by a replacement, so it adds 0, not minus a replacement's points).
    Floored at 0: a player below replacement is worth what a replacement is, so a bust costs its slot's
    expectation and no more."""
    Q = P[P["week"] <= through]
    tot = Q.groupby("espn_id")["points"].sum()
    g = Q[Q["played"].astype(bool)].groupby("espn_id").size().reindex(tot.index).fillna(0)
    pos = P.drop_duplicates("espn_id").set_index("espn_id")["position"]
    ppg = (tot / g)[g >= max(1, (through + 1) // 2)]
    rp = _repl(ppg, pos, n_teams(season))
    vor = (tot - g * pos.reindex(tot.index).map(rp).fillna(0.0)).clip(lower=0.0)
    return pd.DataFrame({"pts": tot, "gp": g, "vor": vor})


def _slot_curve(through: int) -> np.ndarray:
    """Mean VOR by overall pick (1..N) through week ``through``, pooled over 2021-25 drafts with a
    +/-WIN pick window. Index 0 is pick 1. Undrafted-in-data players count as replacement (0)."""
    rows = []
    for y in HIST:
        dr = pd.read_csv(ACT / str(y) / "draft.csv")
        P = _season_pts(y)
        now = _vor(y, P, through)["vor"]
        rows.append(pd.DataFrame({"pick": dr["pick"], "now": dr["espn_id"].map(now).fillna(0.0)}))
    H = pd.concat(rows)
    m = H.groupby("pick")["now"].mean().reindex(range(1, int(H["pick"].max()) + 1))
    return m.rolling(2 * WIN + 1, center=True, min_periods=WIN).mean().to_numpy()


def _history(through: int) -> dict:
    """How much a draft ranking at week ``through`` holds to week 14, and whether a team slot's draft
    value repeats the next season (team ids are slots; owners may have changed)."""
    hold, nxt, vs_pts = [], [], []
    by_season = {}
    for y in HIST:
        dr = pd.read_csv(ACT / str(y) / "draft.csv").sort_values("pick")
        P = _season_pts(y)
        v = {}
        for W in (through, 14):
            pts = dr["espn_id"].map(_vor(y, P, W)["vor"]).fillna(0.0)
            exp = pts.rolling(2 * WIN + 1, center=True, min_periods=WIN).mean()
            v[W] = (pts - exp).groupby(dr["team_id"]).sum()
        by_season[y] = v[14]
        hold.append(v[through].rank().corr(v[14].rank()))
        tw = pd.read_csv(ACT / str(y) / "team_weekly.csv")
        reg = tw[~tw["playoff"].astype(bool)].groupby("team_id")["points"].sum()
        vs_pts.append(v[14].rank().corr(reg.reindex(v[14].index).rank()))
    for y in list(HIST)[:-1]:
        a, b = by_season[y], by_season[y + 1]
        ix = a.index.intersection(b.index)
        nxt.append(a[ix].rank().corr(b[ix].rank()))
    f = lambda xs: [_r(x, 2) for x in xs]
    return {"through": through, "seasons": list(HIST), "hold": f(hold), "next_year": f(nxt), "vs_points": f(vs_pts)}


def _board() -> Dict[str, dict]:
    if not BOARD.exists():
        return {}
    B = pd.read_csv(BOARD)
    out = {}
    for _, r in B.iterrows():
        adp = pd.to_numeric(r.get("ADP"), errors="coerce")
        out.setdefault((_norm(r["Name"]), str(r["Position"]).upper()),
                       {"adp": None if pd.isna(adp) else float(adp), "rank": int(r["Rank"]),
                        "proj": _r(r.get("Projected Points"), 1)})
    return out


_CACHE: dict = {}


def recap(season: int = 2026) -> dict:
    files = [ACT / str(season) / f for f in ("draft.csv", "players_weekly.csv", "rosters_by_week.csv", "trades.csv")]
    files += sorted((LEDGER / str(season)).glob("wk*_players.csv"))[-1:] + [BOARD]
    key = (season, tuple((f.name, f.stat().st_mtime if f.exists() else None) for f in files))
    if _CACHE.get("key") != key:
        _CACHE.update(key=key, out=_recap(season))
    return _CACHE["out"]


def _recap(season: int) -> dict:
    d = ACT / str(season)
    settings = json.loads((d / "settings.json").read_text())
    names = {int(k): v.strip() for k, v in settings["team_names"].items()}
    dr = pd.read_csv(d / "draft.csv").sort_values("pick")
    P = _season_pts(season)
    done = complete_week(season, int(P["week"].max()) if len(P) else 0)
    P = P[P["week"] <= done]
    meta = pd.read_csv(d / "players_weekly.csv").drop_duplicates("espn_id").set_index("espn_id")
    byes = pd.read_csv(d / "byes.csv").set_index("pro_team_id")["bye_week"]
    V = _vor(season, P, done)

    # ROS forecast VOR: remaining weeks of expected points x availability (latest ledger), minus replacement
    led = sorted((LEDGER / str(season)).glob("wk*_players.csv"))
    ros = pd.Series(dtype=float)
    if led:
        L = pd.read_csv(led[-1]).drop_duplicates("espn_id").set_index("espn_id")
        wk = [w for w in range(done + 1, LAST + 1) if f"xp_w{w}" in L]
        if wk:
            r = L["position"].map(_repl(L["est_ppg"], L["position"], len(names))).fillna(0.0)
            ros = sum(L[f"xa_w{w}"].fillna(0) * (L[f"xp_w{w}"].fillna(0) - r).clip(lower=0) for w in wk)
    curve = _slot_curve(done)
    f_by_pick = dr["espn_id"].map(ros).fillna(0.0) if len(ros) else None
    ros_curve = None if f_by_pick is None else \
        f_by_pick.rolling(2 * WIN + 1, center=True, min_periods=WIN).mean().to_numpy()
    board = _board()

    R = pd.read_csv(d / "rosters_by_week.csv")
    last_wk = int(R["week"].max()) if len(R) else 0
    own_now = R[R["week"] == last_wk].set_index("espn_id")["team_id"].to_dict()
    try:
        traded = set(pd.read_csv(d / "trades.csv")["espn_id"].astype(int))
    except (pd.errors.EmptyDataError, KeyError):
        traded = set()
    # started points by (player, team) and the week a drafter last had him
    wpts = P.set_index(["espn_id", "week"])["points"].to_dict()
    started = R[~R["lineup_slot"].isin(["BE", "IR"])]
    st_for: Dict[tuple, float] = {}
    for i, t, w in zip(started["espn_id"], started["team_id"], started["week"]):
        if w <= done:
            st_for[(int(i), int(t))] = st_for.get((int(i), int(t)), 0.0) + wpts.get((int(i), int(w)), 0.0)
    held = R.groupby(["espn_id", "team_id"])["week"].max().to_dict()

    picks = []
    for rd, pk, t, i in zip(dr["round"], dr["pick"], dr["team_id"], dr["espn_id"]):
        i, t, pk = int(i), int(t), int(pk)
        m = meta.loc[i] if i in meta.index else None
        pos = None if m is None else m["position"]
        name = None if m is None else m["name"]
        bye = byes.get(int(m["pro_team_id"])) if m is not None and pd.notna(m["pro_team_id"]) else None
        possible = done - (1 if bye is not None and pd.notna(bye) and bye <= done else 0)
        exp_now = curve[pk - 1] if pk <= len(curve) else np.nan
        exp_ros = ros_curve[pk - 1] if ros_curve is not None else np.nan
        got = float(V["pts"].get(i, 0.0))
        g, vor = int(V["gp"].get(i, 0)), float(V["vor"].get(i, 0.0))
        # split VOR-vs-slot into games played vs the slot's per-game rate, and games missed (mostly luck)
        slot_pg = exp_now / possible if possible else 0.0
        missed = max(possible - g, 0)
        v_now = vor - exp_now
        v_abs = -missed * slot_pg
        f_ros = float(ros.get(i, 0.0)) if len(ros) else None
        v_ros = None if f_ros is None else f_ros - exp_ros
        b = board.get((_norm(name), "DST" if pos == "D/ST" else str(pos))) if name else None
        owner = own_now.get(i)
        status = "kept" if owner == t else ("traded" if i in traded else "dropped")
        picks.append({
            "id": i, "name": name, "pos": pos, "team_id": t, "round": int(rd), "pick": pk,
            "pts": _r(got, 1), "vor": _r(vor, 1), "gp": g, "possible": possible, "exp_now": _r(exp_now, 1), "v_now": _r(v_now, 1),
            "v_rate": _r(v_now - v_abs, 1), "v_absence": _r(v_abs, 1),
            "ros": _r(f_ros, 1), "exp_ros": _r(exp_ros, 1), "v_ros": _r(v_ros, 1),
            "v_total": _r(v_now + (v_ros or 0.0), 1),
            "adp": None if not b or b["adp"] is None else _r(b["adp"], 1),
            "vs_adp": None if not b or b["adp"] is None or pos in ("K", "DST") else _r(pk - b["adp"], 1),
            "board_rank": None if not b else b["rank"],
            "status": status, "owner_now": None if owner is None else int(owner),
            "left_after": None if status == "kept" else held.get((i, t)),
            "started_for_drafter": _r(st_for.get((i, t), 0.0), 1),
            "started_elsewhere": _r(sum(v for (pi, tt), v in st_for.items() if pi == i and tt != t), 1),
        })
    X = pd.DataFrame(picks)
    teams = []
    for t, g in X.groupby("team_id"):
        a = g[g["vs_adp"].notna()]
        teams.append({
            "id": int(t), "name": names.get(int(t), str(t)),
            **{k: _r(g[k].sum(), 1) for k in ("v_now", "v_rate", "v_absence", "v_ros", "v_total", "pts",
                                               "started_for_drafter", "started_elsewhere")},
            "vs_adp": _r(a["vs_adp"].clip(-ADP_CLIP, ADP_CLIP).mean(), 1), "adp_n": int(len(a)),
            "kept": int((g["status"] == "kept").sum()), "dropped": int((g["status"] == "dropped").sum()),
            "traded": int((g["status"] == "traded").sum()),
            "by_pos": {p: _r(gg["v_total"].sum(), 1) for p, gg in g.groupby("pos")},
        })
    for k in ("v_now", "v_ros", "v_total"):
        order = sorted(teams, key=lambda z: -(z[k] or 0))
        for n, z in enumerate(order, 1):
            z[f"{k}_rank"] = n
    a = X[X["vs_adp"].notna()]
    league_adp = a["vs_adp"].clip(-ADP_CLIP, ADP_CLIP).mean()
    for z in teams:
        z["vs_adp_rel"] = _r(z["vs_adp"] - league_adp, 1) if z["vs_adp"] is not None else None
    reach_paid = _r(a["vs_adp"].corr(a["v_total"]), 2) if len(a) > 5 else None
    return {"season": season, "through_week": done, "ros_from_week": done + 1, "n_teams": len(names),
            "rounds": int(dr["round"].max()), "picks": picks, "teams": teams,
            "reach_corr": reach_paid, "league_vs_adp": _r(league_adp, 1), "adp_matched": int(len(a)), "history": _history(done)}
