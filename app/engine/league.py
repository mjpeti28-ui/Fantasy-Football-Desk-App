"""League screen: live rest-of-season odds and injury what-ifs, on the existing season simulator.

Odds come from a common-random-numbers version of season_sim.simulate (see simulate_crn),
cached until rosters change. A what-if removes players for N weeks (or the season) and reruns
on the same random draws, so the difference against the base is the scenario, not noise.
"""
from __future__ import annotations

import hashlib
import json
import threading
from typing import Dict, List, Optional

import numpy as np

from . import data as d, store

_lock = threading.Lock()
_ctx_lock = threading.Lock()
_cache: dict = {}
SEED = 7
FIELDS = ("seed1", "top2", "playoffs", "title", "reg_pts")


def _ctx():
    import trade_value as tv
    key = ("ctx", d.load().week, d.ages_key())
    with _ctx_lock:  # the warm-up thread and a request can arrive together; load it once
        if _cache.get("ctx_key") != key:
            _cache.clear()
            _cache["ctx"], _cache["ctx_key"] = tv.load_context(d.load().season, d.load().week), key
        return _cache["ctx"]


def _ratios():
    if "ratios" not in _cache:
        r = json.loads(d.RATIOS.read_text())
        _cache["ratios"] = {k: np.asarray(v, float) for k, v in r.items()}
    return _cache["ratios"]


def simulate_crn(out: Dict[int, Optional[int]], sims: int, extras: Optional[dict] = None,
                 rosters: Optional[Dict[int, List[int]]] = None) -> Dict[int, dict]:
    """season_sim.simulate (levels off), re-implemented with common random numbers: every
    player's play/no-play draw and points ratio is fixed per (simulated season, week, player).
    A scenario then changes only what the scenario touches, so deltas aren't swamped by
    Monte Carlo noise. `out` = {player_id: weeks out, or None for the rest of the season}."""
    import season_sim as ss
    import valuation as v
    ctx = _ctx()
    s, week, tbl, repl = ctx.season, ctx.week, ctx.tbl, ctx.replacement
    base_rosters = ctx.rosters
    rosters = rosters or base_rosters
    ratios = _ratios()
    teams = sorted(rosters)
    reg_end, last = s.reg_weeks, v.FANTASY_LAST_WEEK
    n_playoff = int(s.settings.get("playoff_teams", 8))
    tw = s.team_weekly
    so_far = {t: float(tw[(tw["team_id"] == t) & (tw["week"] < week)]["points"].sum()) for t in teams}
    pos = {i: tbl.at[i, "position"] for ids in list(rosters.values()) + list(base_rosters.values()) for i in ids}
    est = {i: float(tbl.at[i, "est_ppg"]) for ids in list(rosters.values()) + list(base_rosters.values()) for i in ids}
    streamer = {p: -(k + 1) for k, p in enumerate(ss.POS_ORDER)}
    for p, sid in streamer.items():
        pos[sid], est[sid] = p, repl.get(p, 0.0)
    weeks = list(range(week, last + 1))
    avail = {w: dict(zip(tbl.index, v._avail_for_week(tbl, w))) for w in weeks}
    for pid, n in out.items():
        for w in weeks:
            if n is None or w < week + n:
                avail[w][pid] = 0.0
    # random draws are keyed to the base rosters' players so a roster change (trade, drop)
    # leaves every other player's draws untouched
    players = sorted({i for ids in base_rosters.values() for i in ids} | {i for ids in rosters.values() for i in ids})
    col = {i: k for k, i in enumerate(players)}
    rng = np.random.default_rng(SEED)
    U_play = rng.random((sims, len(weeks), len(players)), dtype=np.float32)
    U_ratio = rng.random((sims, len(weeks), len(players)), dtype=np.float32)
    rs_by_pos = {p: np.asarray(r) for p, r in ratios.items()}

    counts = {t: {"seed1": 0, "top2": 0, "playoffs": 0, "title": 0, "reg_pts": 0.0} for t in teams}
    # extras for the League analysis: rank distribution, mean points by week, starter points by
    # position (regular season), and how often a waiver streamer had to start at each position
    nT, nW = len(teams), len(weeks)
    rank_ct = np.zeros((nT, nT))
    wk_pts = np.zeros((nT, nW))
    POS = list(ss.POS_ORDER)
    pos_pts = np.zeros((nT, len(POS)))
    stream_ct = np.zeros((nT, len(POS)))
    tix = {t: k for k, t in enumerate(teams)}
    for n in range(sims):
        memo: dict = {}

        def score(t: int, w: int) -> float:
            if (t, w) in memo:
                return memo[(t, w)]
            wi = w - week
            up, ur = U_play[n, wi], U_ratio[n, wi]
            ids = [i for i in rosters[t] if up[col[i]] < avail[w].get(i, 0.0)] + list(streamer.values())
            tot = 0.0
            reg = w <= reg_end and extras is not None
            for i in ss._lineup(ids, pos, est, s.slots):
                if i < 0:
                    pts = est[i]
                    if reg:
                        stream_ct[tix[t], POS.index(pos[i])] += 1
                else:
                    r = rs_by_pos.get(pos[i])
                    pts = est[i] * (float(r[int(ur[col[i]] * len(r))]) if r is not None and len(r) else 1.0)
                if reg:
                    pos_pts[tix[t], POS.index(pos[i])] += pts
                tot += pts
            memo[(t, w)] = tot
            return tot

        total = {t: so_far[t] + sum(score(t, w) for w in range(week, reg_end + 1)) for t in teams}
        order = sorted(teams, key=lambda t: -total[t])
        if extras is not None:
            for rank, t in enumerate(order):
                rank_ct[tix[t], rank] += 1
            for t in teams:
                for wi, w in enumerate(weeks):
                    wk_pts[tix[t], wi] += score(t, w)
        seeds = order[:n_playoff]
        for rank, t in enumerate(order):
            counts[t]["reg_pts"] += total[t]
            if rank == 0:
                counts[t]["seed1"] += 1
            if rank < 2:
                counts[t]["top2"] += 1
        for t in seeds:
            counts[t]["playoffs"] += 1
        pw = reg_end + 1
        game = lambda a, b, w: a if score(a, w) >= score(b, w) else b
        if len(seeds) >= 8:
            fa = game(game(seeds[0], seeds[7], pw), game(seeds[3], seeds[4], pw), pw + 1)
            fb = game(game(seeds[1], seeds[6], pw), game(seeds[2], seeds[5], pw), pw + 1)
        else:
            fa, fb = seeds[0], seeds[1]
        counts[game(fa, fb, pw + 2)]["title"] += 1
    if extras is not None:
        n_reg = max(1, reg_end - week + 1)
        extras.update({
            "weeks": weeks, "positions": POS, "reg_end": reg_end, "playoff_teams": n_playoff,
            "seed_dist": {int(t): [round(x / sims, 4) for x in rank_ct[tix[t]]] for t in teams},
            "week_pts": {int(t): [round(x / sims, 1) for x in wk_pts[tix[t]]] for t in teams},
            # starter points per regular-season week, by the position of the player who started
            "pos_ppw": {int(t): {p: round(pos_pts[tix[t], k] / sims / n_reg, 2) for k, p in enumerate(POS)} for t in teams},
            "stream_rate": {int(t): {p: round(stream_ct[tix[t], k] / sims / n_reg, 3) for k, p in enumerate(POS)} for t in teams},
        })
    return {int(t): {k: round(counts[t][k] / sims, 4) for k in FIELDS} for t in teams}


def _run(out: Dict[int, Optional[int]], sims: int) -> Dict[int, dict]:
    with _lock:
        return simulate_crn(out, sims)


def base(sims: int = 2000) -> Dict[int, dict]:
    _ctx()
    key = ("base", sims)
    with _lock:
        if key not in _cache:
            # Same draws (fixed seed) on the same inputs give the same answer, so a saved run with
            # this exact key is reused across app launches.
            dk = ("league-base", sims, SEED, d.load().week, d.ages_key())
            hit = store.get(f"league_base_{sims}", dk)
            if hit is not None:
                res, ex = hit[0]
            else:
                ex = {}
                res = simulate_crn({}, sims, extras=ex)
                store.put(f"league_base_{sims}", dk, (res, ex))
            _cache[("extras", sims)] = ex
            _cache[key] = res
    return _cache[key]


def analysis(sims: int = 2000) -> dict:
    """Simulation extras for the base case plus actual points by position so far."""
    base(sims)
    ex = dict(_cache[("extras", sims)])
    inp = d.load()
    pw = inp.players_weekly[inp.players_weekly["week"] < inp.week][["week", "espn_id", "points", "position"]]
    r = inp.rosters[(inp.rosters["week"] < inp.week) & ~inp.rosters["lineup_slot"].isin(["BE", "IR"])][["week", "team_id", "espn_id"]]
    m = r.merge(pw, on=["week", "espn_id"], how="left")
    m["position"] = m["position"].replace({"D/ST": "DST"})
    so_far = m.groupby(["team_id", "position"])["points"].sum()
    ex["pos_so_far"] = {int(t): {str(p): round(float(v), 1) for p, v in g.droplevel(0).items()} for t, g in so_far.groupby(level=0)}
    return ex


def whatif(out: List[dict], sims: int = 2000) -> dict:
    o = {int(x["id"]): (None if x.get("weeks") in (None, 0, "season") else int(x["weeks"])) for x in out}
    b = base(sims)
    k = ("wi", sims, hashlib.md5(json.dumps(sorted(o.items(), key=lambda z: z[0])).encode()).hexdigest())
    if k not in _cache:
        _cache[k] = _run(o, sims)
    s = _cache[k]
    return {"sims": sims, "base": b, "scenario": s,
            "delta": {t: {f: round(s[t][f] - b[t][f], 4) for f in FIELDS} for t in s},
            # Monte Carlo standard error of a probability near 50% at this many sims
            "noise_pp": round(100 * 0.5 / np.sqrt(sims), 1)}


def rosters() -> Dict[int, List[dict]]:
    """Every team's roster with the model's numbers, for the League inspector."""
    from .views import _pro_team, headshot
    ctx = _ctx()
    t = ctx.tbl
    nfl = _pro_team(d.load())
    out = {}
    for team, ids in ctx.rosters.items():
        rows = []
        for i in ids:
            r = t.loc[i]
            rows.append({"id": int(i), "name": str(r["name"]), "pos": str(r["position"]),
                         "est_ppg": round(float(r["est_ppg"]), 2) if r["est_ppg"] == r["est_ppg"] else None,
                         "avail": round(float(r["avail"]), 3) if r["avail"] == r["avail"] else None,
                         "status": str(r.get("status", "")), "ir": int(i) in ctx.ir_slot,
                         "team": nfl.get(int(i)),
                         "headshot": headshot(int(i), str(r["position"]), nfl.get(int(i)))})
        rows.sort(key=lambda z: -(z["est_ppg"] or 0))
        out[int(team)] = rows
    return out
