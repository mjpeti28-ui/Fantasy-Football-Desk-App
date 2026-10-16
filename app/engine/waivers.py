"""Waivers screen: free-agent shortlist with best drop and team gain, plus usage evidence.

Same method as tools/waivers.py (v2 rest-of-season forecast, expected lineup points weeks now-17,
try every drop). Backtest: v2 picks ~83.6 ROS pts per pickup vs 79.0 for what managers added;
managers see role news the model can't, so this is a shortlist to research, not a verdict.
"""
from __future__ import annotations

import json
import threading
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from . import data as d, league, store

ME = 3
POS = ["QB", "RB", "WR", "TE", "K", "DST"]
OUT_TAGS = {"INJURY_RESERVE", "OUT", "SUSPENSION", "PUP"}
_cache: dict = {}
_lock = threading.Lock()


def _usage(ids: List[int], week: int) -> Dict[int, dict]:
    """Last few weeks of usage from nflverse (evidence for judgment; not a model input)."""
    inp = d.load()
    try:
        idm = inp.ids.dropna(subset=["espn_id"])
        e2g = dict(zip(idm["espn_id"].astype(int), idm["gsis_id"]))
        e2p = dict(zip(idm["espn_id"].astype(int), idm["pfr_id"]))
        ps = pd.read_csv(d.NFLVERSE / str(inp.season) / "player_stats.csv")
        sn = pd.read_csv(d.NFLVERSE / str(inp.season) / "snaps.csv")
    except Exception:
        return {}
    weeks = list(range(max(1, week - 4), week))
    out = {}
    for i in ids:
        g, p = e2g.get(i), e2p.get(i)
        s = ps[ps["player_id"] == g].set_index("week") if g else pd.DataFrame()
        n = sn[sn["pfr_player_id"] == p].set_index("week") if p else pd.DataFrame()
        def col(df, c, scale=1.0):
            return [None if df.empty or w not in df.index or pd.isna(df.at[w, c]) else round(float(df.at[w, c]) * scale, 2) for w in weeks]
        out[i] = {"weeks": weeks, "targets": col(s, "targets"), "carries": col(s, "carries"),
                  "target_share": col(s, "target_share", 100), "snap_pct": col(n, "offense_pct", 100)}
    return out


def _tags() -> Dict[int, str]:
    try:
        import sys
        sys.path.insert(0, str(d.ROOT / "tools"))
        import waivers as w  # tools/waivers.py
        return w.fa_injury_tags()
    except Exception:
        return {}


def _tags_stamp() -> str:
    """When the free-agent injury tags would change: the latest espn_live snapshot if it's fresh
    (tools/waivers.fa_injury_tags uses it for 2 days), otherwise today (it pulls live)."""
    import time
    snaps = sorted((d.ROOT / "history" / "snapshots" / str(d.load().season)).glob("*/espn_live.csv"))
    if snaps and time.time() - snaps[-1].stat().st_mtime < 2 * 86400:
        return f"snap:{snaps[-1].stat().st_mtime:.0f}"
    return "day:" + time.strftime("%Y-%m-%d")


# Drops tried per candidate: the K rostered players whose loss costs your lineup least, plus
# everyone at the candidate's position. Against trying every drop (week 5, 36 candidates) this
# picked the same drop every time at about a third of the cost.
DROP_K = 3


# Players you won't drop. Saved by the engine (app/cache.nosync/prefs/), so every window and the
# warm-up build the same board; the drop search just never offers them.
def _keep_file():
    return store.local_dir("prefs") / "waiver_keep.json"


def load_keep() -> List[int]:
    try:
        return sorted({int(x) for x in json.loads(_keep_file().read_text())["ids"]})
    except Exception:
        return []


def save_keep(ids: Iterable[int]) -> List[int]:
    ids = sorted({int(i) for i in ids})
    p = _keep_file()
    tmp = p.with_suffix(".tmpw")
    tmp.write_text(json.dumps({"ids": ids}))
    tmp.replace(p)
    return ids


def board(top: int = 6) -> dict:
    keep = tuple(load_keep())
    key = ("board", top, d.load().week, d.ages_key(), _tags_stamp(), keep)
    with _lock:
        if key in _cache:
            return _cache[key]
        hit = store.get("waivers", key)
        if hit is not None:
            _cache.clear()
            _cache[key] = {**hit[0], "computed_at": hit[1]}
            return _cache[key]
        out = _build(top, keep)
        store.put("waivers", key, out)
        _cache.clear()
        _cache[key] = out
        return out


def _build(top: int, keep: Iterable[int] = ()) -> dict:
    import time
    import trade_value as tv
    import valuation as v
    from .views import _pro_team, headshot
    ctx = league._ctx()
    t, weeks = ctx.tbl, range(ctx.week, v.FANTASY_LAST_WEEK + 1)
    rostered = {i for ids in ctx.rosters.values() for i in ids}
    pool = t[~t.index.isin(rostered) & t["position"].isin(POS)].copy()
    pool["ros"] = v.expected_ros_points(pool, ctx.week, v.FANTASY_LAST_WEEK)
    pool["p_start"], pool["p_boom"] = v.hit_probability(pool, "start"), v.hit_probability(pool, "boom")
    tags = _tags()
    pool["tag"] = pool.index.map(lambda i: tags.get(int(i), ""))
    med = pool.groupby("position")["ros"].transform("median")
    hurt = pool[pool["tag"].isin(OUT_TAGS) & (pool["ros"] > med)]
    pool = pool[~pool["tag"].isin(OUT_TAGS)]
    mine = ctx.rosters[ME]
    TV = lambda r: v.team_value(r, t, ctx.season.slots, weeks, ctx.replacement, sims=200, seed=3)
    base = TV(mine)
    nfl = _pro_team(d.load())
    keep = {int(i) for i in keep}
    elig = [i for i in mine if i not in ctx.ir_slot]
    # what each of your players is worth: season lineup points lost if he's simply gone
    cost = {int(i): float(base - TV([j for j in mine if j != i])) for i in elig}
    droppable = [i for i in elig if int(i) not in keep]  # the ones you'd actually let go
    cheap = sorted(droppable, key=lambda i: cost[int(i)])[:DROP_K]
    cands = []
    for pos, g in pool.groupby("position"):
        tries = list(dict.fromkeys(cheap + [i for i in droppable if t.at[i, "position"] == pos]))
        if not tries:
            continue
        for pid, r in g.nlargest(top, "ros").iterrows():
            best = None
            for dr in tries:
                val = TV([i for i in mine if i != dr] + [pid])
                if best is None or val > best[0]:
                    best = (val, dr)
            pid = int(pid)
            cands.append({
                "id": pid, "name": str(r["name"]), "pos": pos, "team": nfl.get(pid),
                "headshot": headshot(pid, pos, nfl.get(pid)), "espn_wk": round(float(r.get("espn_now") or 0), 1),
                "est_ppg": round(float(r["est_ppg"]), 2), "avail": round(float(r["avail"]), 3), "ros": round(float(r["ros"]), 1),
                "p_start": None if pd.isna(r["p_start"]) else round(float(r["p_start"]), 3),
                "p_boom": None if pd.isna(r["p_boom"]) else round(float(r["p_boom"]), 3),
                "ecr_rank": None if pd.isna(r.get("ecr_rank")) else int(r["ecr_rank"]), "tag": r["tag"] or "",
                "why": tv.why(ctx, pid) if pid in ctx.tbl.index else "",
                "gain": round(float(best[0] - base), 1), "drop_id": int(best[1]), "drop_name": str(t.at[best[1], "name"]),
            })
    usage = _usage([c["id"] for c in cands] + [int(i) for i in mine], ctx.week)
    for c in cands:
        c["usage"] = usage.get(c["id"])
    depth = {p: [] for p in POS}
    for i in mine:
        p = t.at[i, "position"]
        if p in depth:
            depth[p].append({"id": int(i), "name": str(t.at[i, "name"]), "est_ppg": round(float(t.at[i, "est_ppg"]), 2),
                             "avail": round(float(t.at[i, "avail"]), 3), "ir": int(i) in ctx.ir_slot, "team": nfl.get(int(i)),
                             "usage": usage.get(int(i)), "drop_cost": None if int(i) not in cost else round(cost[int(i)], 1),
                             "kept": int(i) in keep})
    out = {"week": ctx.week, "candidates": sorted(cands, key=lambda c: -c["gain"]), "depth": depth,
           "keep": sorted(int(i) for i in mine if int(i) in keep),
           "replacement": {p: round(float(ctx.replacement.get(p, 0.0)), 2) for p in POS},
           "left_off": [{"name": str(n), "tag": str(g)} for n, g in zip(hurt["name"], hurt["tag"])],
           "noise": 10, "drop_cost": {str(k): round(x, 1) for k, x in cost.items()},
           "pool_size": {p: int((pool["position"] == p).sum()) for p in POS},
           "computed_at": time.time()}
    return out
