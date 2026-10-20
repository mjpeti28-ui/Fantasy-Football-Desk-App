"""Trades screen: partner history, two-view evaluation, season-odds deltas, idea search."""
from __future__ import annotations

import itertools
import threading
import time
from typing import Dict, List, Optional

import pandas as pd

from . import data as d, league

ME = 3
_ideas = {"state": "idle", "partner": None, "player": None, "started": None, "finished": None, "rows": [], "error": None}
_ctx_cache: dict = {}


def _ctx(espn_view: bool = False):
    import trade_value as tv
    key = ("espn" if espn_view else "model", d.load().week, d.ages_key())
    if not espn_view:
        return tv, league._ctx()  # same context the league and waivers screens use
    if key not in _ctx_cache:
        params = tv.source_params("espn") if espn_view else None
        _ctx_cache.clear() if len(_ctx_cache) > 4 else None
        _ctx_cache[key] = tv.load_context(d.load().season, d.load().week, params=params)
    return tv, _ctx_cache[key]


def history() -> Dict[int, dict]:
    """Your past proposals to each current team (2021-25): sent, accepted, declined, canceled.
    Assumes an ESPN team id is the same manager across seasons."""
    t = pd.read_csv(d.ACTUALS / "trade_transactions_all.csv")
    p = t[t["type"] == "TRADE_PROPOSAL"]
    resp = t[t["type"].isin(["TRADE_ACCEPT", "TRADE_DECLINE"])].groupby("related")["type"].first()
    out: Dict[int, dict] = {}
    for tx, g in p.groupby("tx_id"):
        if int(g["proposer"].iloc[0]) != ME:
            continue
        teams = set(g["from_team"].dropna().astype(int)) | set(g["to_team"].dropna().astype(int))
        other = [x for x in teams if x != ME]
        if not other:
            continue
        h = out.setdefault(int(other[0]), {"sent": 0, "accepted": 0, "declined": 0, "canceled": 0})
        h["sent"] += 1
        r = resp.get(tx)
        if r == "TRADE_ACCEPT":
            h["accepted"] += 1
        elif r == "TRADE_DECLINE":
            h["declined"] += 1
        elif (g["status"] == "CANCELED").any():
            h["canceled"] += 1
    return out


def evaluate(partner: int, give: List[int], get: List[int]) -> dict:
    tv, ctx = _ctx(False)
    for team, ids in ((ME, give), (partner, get)):
        bad = [i for i in ids if i not in ctx.rosters.get(team, [])]
        if bad:
            raise ValueError(f"players {bad} are not on team {team}")
    model = tv.evaluate(ctx, ME, give, partner, get, sims=300)
    tv2, ctx2 = _ctx(True)
    espn = tv2.evaluate(ctx2, ME, give, partner, get, sims=300)
    return {"model": model, "espn": espn, "calibration": ctx.calibration.get("sign_accuracy_by_abs_pred_edge", [])}


def odds(partner: int, give: List[int], get: List[int], sims: int = 2000) -> dict:
    """Season odds before and after the trade, on the same simulated seasons."""
    import trade_value as tv
    ctx = league._ctx()
    after = dict(ctx.rosters)
    after[ME] = tv._after(ctx, ctx.rosters[ME], give, get)
    after[partner] = tv._after(ctx, ctx.rosters[partner], get, give)
    base = dict(league.base(sims))  # copy: the league cache is shared
    ex: dict = {}
    with league._lock:
        res = league.simulate_crn({}, sims, rosters=after, extras=ex)
    med = lambda dist: next(k + 1 for k, c in enumerate(itertools.accumulate(dist)) if c >= 0.5)  # median seed, 1 = best
    sd0, sd1 = league._cache[("extras", sims)]["seed_dist"], ex["seed_dist"]
    keys = ("seed1", "top2", "playoffs", "title", "reg_pts", "seed_med")
    for t in (ME, partner):
        base[t] = {**base[t], "seed_med": med(sd0[t])}
        res[t] = {**res[t], "seed_med": med(sd1[t])}
    return {"sims": sims, "noise_pp": round(100 * 0.5 / sims ** 0.5, 1), "week": ctx.week,
            "teams": {t: {"before": base[t], "after": res[t], "delta": {k: round(res[t][k] - base[t][k], 4) for k in keys}}
                      for t in (ME, partner)}}


def _run_ideas(partner: Optional[int], player: Optional[int] = None) -> None:
    try:
        tv, ctx = _ctx(False)
        if player is not None:  # deals built around one player: land him, or shop him
            rows = tv.around(ctx, _ctx(True)[1], ME, player).to_dict("records")
            _ideas.update(state="done", rows=[{k: (float(v) if isinstance(v, float) else v) for k, v in r.items()} for r in rows], finished=time.time())
            return
        df = tv.find(ctx, ME, partners=[partner] if partner else None)
        names = {(t, ctx.tbl.at[i, "name"]): i for t, ids in ctx.rosters.items() for i in ids}
        team_of = {ctx.team_name(t): t for t in ctx.rosters}
        rows = []
        for r in df.to_dict("records"):
            other = team_of.get(r["with"])
            rows.append({**{k: (float(v) if isinstance(v, (int, float)) else v) for k, v in r.items()}, "partner": other,
                         "give_ids": [names.get((ME, n.strip())) for n in r["give"].split(",")],
                         "get_ids": [names.get((other, n.strip())) for n in r["get"].split(",")]})
        _ideas.update(state="done", rows=rows, finished=time.time())
    except Exception as e:  # surface the reason in the UI
        _ideas.update(state="failed", error=str(e), finished=time.time())


def start_ideas(partner: Optional[int], player: Optional[int] = None) -> dict:
    if _ideas["state"] == "running":
        return _ideas
    _ideas.update(state="running", partner=partner, player=player, started=time.time(), finished=None, rows=[], error=None)
    threading.Thread(target=_run_ideas, args=(partner, player), daemon=True).start()
    return _ideas


def ideas_status() -> dict:
    return _ideas
