"""Season simulator: rest-of-season points, playoff seeding, and title odds.

Why: seeding is by total points (so the regular season is a points race) but the
title is three single-week knockout games, where week-to-week noise (~23 pts SD per
team) dwarfs the gap between teams' averages (~8 pts SD). Expected points alone
can't tell you how much a move changes your *championship* odds; this can.

Model:
- Each player-week: plays with his availability (valuation.player_table); if he
  plays, points = forecast ppg x a ratio drawn from the empirical distribution of
  actual/forecast for his position (fit on past seasons). Boom/bust is barely a
  repeatable player trait (CV repeat corr ~0.1-0.3 within position), so one
  distribution per position.
- Each team starts its best lineup *by expectation* among players available that
  week (managers can't see the outcome), plus a waiver streamer at every position.
- Seeds: top ``playoff_teams`` by total regular-season points (points so far +
  simulated). Bracket: 1v8, 4v5, 2v7, 3v6 in week 15, winners meet in 16, final 17.

League size is read from the data (12 teams 2021-2025, 10 in 2026; 8 playoff teams).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import valuation as v

POS_ORDER = v.POSITIONS


def fit_ratio_dist(seasons: List[int], params_for, weeks=range(3, 15)) -> Dict[str, np.ndarray]:
    """actual / forecast ppg for players who played, by position (forecast made that week)."""
    out: Dict[str, list] = {p: [] for p in POS_ORDER}
    for yr in seasons:
        s = v.load_season(yr)
        p = params_for(yr)
        wk = s.weekly[s.weekly["played"]]
        for W in weeks:
            t = v.player_table(s, W, p)
            a = wk[wk["week"] == W].set_index("espn_id")["points"]
            j = t.join(a.rename("actual"), how="inner")
            j = j[j["est_ppg"] > 2]
            for pos, g in j.groupby("position"):
                out[pos].extend((g["actual"] / g["est_ppg"]).clip(lower=0, upper=6).tolist())
    return {p: np.array(x) for p, x in out.items() if x}


def _lineup(avail_ids: List[int], pos: Dict[int, str], exp: Dict[int, float], slots) -> List[int]:
    from backtest.start_sit import choose

    return choose([(i, pos[i], exp[i]) for i in avail_ids], slots)


def simulate(
    s: v.Season, week: int, params: v.ForecastParams, ratios: Dict[str, np.ndarray], *,
    rosters: Optional[Dict[int, List[int]]] = None, n_sims: int = 1000, seed: int = 0,
    tbl: Optional[pd.DataFrame] = None, replacement: Optional[Dict[str, float]] = None,
    levels: bool = False,
) -> pd.DataFrame:
    """``levels``: also draw each player's persistent rest-of-season level once per simulated
    season (lognormal, mean 1, sd from tbl['level_sd']) and shrink the weekly ratio noise so
    the per-game variance is unchanged -- weeks for the same player become correlated."""
    rng = np.random.default_rng(seed)
    tbl = tbl if tbl is not None else v.player_table(s, week, params)
    if tbl.empty or float(tbl["est_ppg"].fillna(0).sum()) == 0.0:
        # e.g. 2023 week 1: ESPN's season and week-1 projections are both corrupt, and no
        # games have been played yet -- there is nothing to simulate from
        raise ValueError(f"no usable forecasts for {s.season} week {week}")
    if rosters is None:
        r = s.roster_at(week)
        rosters = {int(t): [int(i) for i in g["espn_id"] if int(i) in tbl.index] for t, g in r.groupby("team_id")}
    repl = replacement if replacement is not None else v.waiver_replacement(
        tbl, [i for ids in rosters.values() for i in ids], scale=params.waiver_scale)
    teams = sorted(rosters)
    reg_end = s.reg_weeks
    n_playoff = int(s.settings.get("playoff_teams", 8))
    tw = s.team_weekly
    so_far = {t: float(tw[(tw["team_id"] == t) & (tw["week"] < week)]["points"].sum()) for t in teams}

    pos = {i: tbl.at[i, "position"] for ids in rosters.values() for i in ids}
    est = {i: float(tbl.at[i, "est_ppg"]) for ids in rosters.values() for i in ids}
    # streamers: one per position per team, always available, deterministic points
    streamer = {p: -(k + 1) for k, p in enumerate(POS_ORDER)}
    for p, sid in streamer.items():
        pos[sid], est[sid] = p, repl.get(p, 0.0)
    avail_by_week = {w: dict(zip(tbl.index, v._avail_for_week(tbl, w))) for w in range(week, v.FANTASY_LAST_WEEK + 1)}

    ids_all = [i for ids in rosters.values() for i in ids]
    lsd = {i: float(tbl.at[i, "level_sd"]) if levels and "level_sd" in tbl.columns and pd.notna(tbl.at[i, "level_sd"]) else 0.0
           for i in ids_all}
    rvar = {p: float(np.var(r)) if r is not None and len(r) else 0.0 for p, r in ratios.items()}
    rmean = {p: float(np.mean(r)) if r is not None and len(r) else 1.0 for p, r in ratios.items()}
    shrink = {i: (np.sqrt(max(0.0, 1 - lsd[i] ** 2 / rvar[pos[i]])) if rvar.get(pos[i], 0) > 0 else 1.0) for i in ids_all}
    level = {i: 1.0 for i in ids_all}

    def draw_levels():
        for i in ids_all:
            sd = min(lsd[i], np.sqrt(rvar.get(pos[i], 0.0)))
            if sd > 0:
                s2 = np.log1p(sd ** 2)
                level[i] = float(np.exp(-s2 / 2 + np.sqrt(s2) * rng.standard_normal()))

    def week_score(t: int, w: int) -> float:
        ids = [i for i in rosters[t] if rng.random() < avail_by_week[w].get(i, 0.0)] + list(streamer.values())
        picks = _lineup(ids, pos, est, s.slots)
        tot = 0.0
        for i in picks:
            if i < 0:
                tot += est[i]
            else:
                rs = ratios.get(pos[i])
                r = rs[rng.integers(len(rs))] if rs is not None and len(rs) else 1.0
                if levels:
                    m = rmean.get(pos[i], 1.0)
                    r = level[i] * (m + (r - m) * shrink[i])
                tot += est[i] * r
        return tot

    counts = {t: {"seed1": 0, "top2": 0, "playoffs": 0, "title": 0, "final": 0, "reg_pts": 0.0} for t in teams}
    for _ in range(n_sims):
        if levels:
            draw_levels()
        total = {t: so_far[t] + sum(week_score(t, w) for w in range(week, reg_end + 1)) for t in teams}
        order = sorted(teams, key=lambda t: -total[t])
        seeds = order[:n_playoff]
        for rank, t in enumerate(order):
            counts[t]["reg_pts"] += total[t]
            if rank == 0:
                counts[t]["seed1"] += 1
            if rank < 2:
                counts[t]["top2"] += 1
        for t in seeds:
            counts[t]["playoffs"] += 1
        # bracket (seed index 0-based): (0,7),(3,4) | (1,6),(2,5)
        pw = reg_end + 1
        def game(a, b, w):
            return a if week_score(a, w) >= week_score(b, w) else b
        if len(seeds) >= 8:
            a = game(game(seeds[0], seeds[7], pw), game(seeds[3], seeds[4], pw), pw + 1)
            b = game(game(seeds[1], seeds[6], pw), game(seeds[2], seeds[5], pw), pw + 1)
        else:
            a, b = seeds[0], seeds[1]
        counts[a]["final"] += 1
        counts[b]["final"] += 1
        counts[game(a, b, pw + 2)]["title"] += 1
    out = pd.DataFrame(counts).T
    for c in ("seed1", "top2", "playoffs", "final", "title"):
        out[c] = out[c] / n_sims
    out["reg_pts"] = out["reg_pts"] / n_sims
    return out.sort_values("title", ascending=False)
