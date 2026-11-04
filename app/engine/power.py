"""League › Power: one summary number per team (Power Score, expected pts/wk from here on), tiers,
and the in-season context behind it. Method and every trust note come from the Sleeper backtests in
backtest/ (1,000 public leagues, 2021-25) and the face-validity run on this league (2021-25).

Power Score = w * strength + (1 - w) * PF/g, where strength = points the best legal lineup from the
roster would score on ESPN's next-week projections (byes/absent filled with the player's mean
projection so far), averaged over the last 3 weeks of rosters. Read-only; reuses league.simulate_crn.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from . import analytics, data as d, league, store
from .views import _records, best_lineup, power_table

TIER_GAP = 5.5        # pts/wk: P(leader finishes ahead) = 60% at this gap
LOGIT_B = 0.074       # P(A finishes ahead ROS) = 1/(1+exp(-b*gap)); fits 60/70/80% at 5.5/11.4/18.7 (Sleeper)
P_CAP = 0.70          # shown as "70%+": beyond ~11 pts/wk this league ran less sure than the curve (2021-25)
GOOD_TRADE = 2.6      # pts/wk, Sleeper median of the winning side of a trade
BONUS = 5.0           # pts/wk added in the sim for "what +1 pt/wk buys" (scaled back to 1)
SMOOTH = 3            # weeks of strength averaged (same accuracy, about 35% less churn than 1 week)

NOTES = {
    "power": "Expected points per week from here on. In 1,000 Sleeper leagues it ordered pairs of teams right 68% of "
             "the time (rest-of-season scoring), vs 67% for points per game and 65% for record.",
    "tiers": f"A tier is every team within {TIER_GAP} pts/wk of its top team: inside a tier, the top team finishes "
             "ahead less than 60% of the time.",
    "pairwise": "Chance your rest-of-season points per game beat theirs, from the Power gap. Capped at 70%+ because "
                "big gaps were less sure in this league than in the Sleeper sample.",
    "luck": "Schedule luck and over/under-performance didn't carry over in 1,000 leagues (r about 0). Teams far above "
            "their projections scored less than their Power said afterwards.",
    "lineup_cost": "Points per week vs the lineup ESPN's projections would have started from the same roster. The one "
                   "manager habit that carried over (r .14).",
    "fragility": "Share of lineup strength lost if the top starter is out. Barely predictive (r -.01); context only.",
    "trajectory": "Change in roster strength over 3 weeks. Didn't predict anything beyond the current level; it says "
                  "what changed, not where it's going.",
    "trades": f"Gap in 'good trades': {GOOD_TRADE} pts/wk is what the winning side of a typical trade gains.",
    "title": "Title odds come from three single-week playoff games: close to a lottery. +1 pt/wk = title odds added "
             "by one more point every remaining week, playoffs included.",
}


def strength_weight(week: int) -> float:
    """Weight on strength vs PF/g after `week` games. Smooth version of the Sleeper leave-one-season-out
    fit (.70 at week 4 drifting to .40 by week 10)."""
    return float(np.clip(0.70 - 0.05 * (week - 4), 0.40, 0.80))


def p_ahead(gap: float) -> float:
    return 1.0 / (1.0 + math.exp(-LOGIT_B * gap))


def tiers(scores: Dict[int, float]) -> Dict[int, int]:
    """Tier leader = best team not yet tiered; every team within TIER_GAP of the leader joins (no chaining)."""
    left = sorted(scores, key=lambda t: -scores[t])
    out, k = {}, 0
    while left:
        k += 1
        lead = scores[left[0]]
        for t in [t for t in left if lead - scores[t] < TIER_GAP]:
            out[t] = k
            left.remove(t)
    return out


def _pairs(inp: d.Inputs) -> Dict[int, List[tuple]]:
    names = {inp.team_name(int(t)): int(t) for t in inp.settings.get("team_names", {})}
    out: Dict[int, List[tuple]] = {}
    for _, g in inp.league_sched.iterrows():
        a, h = names.get(" ".join(str(g["AwayTeam"]).split())), names.get(" ".join(str(g["HomeTeam"]).split()))
        if a is not None and h is not None:
            out.setdefault(int(g["Week"]), []).append((a, h))
    return out


def _values(inp: d.Inputs, week: int) -> Dict[int, float]:
    """Player values known before `week`: ESPN's projection for `week`, else the mean projection so far."""
    pj = inp.espn_proj
    mean = pj[pj["week"] < week].groupby("espn_id")["espn_proj"].mean()
    v = {int(i): float(x) for i, x in mean.items()}
    v.update({int(i): float(x) for i, x in zip(pj[pj["week"] == week]["espn_id"], pj[pj["week"] == week]["espn_proj"])})
    return v


def _rows(inp: d.Inputs, roster_week: int, team: int, V: Dict[int, float]) -> List[dict]:
    r = inp.rosters[(inp.rosters["week"] == roster_week) & (inp.rosters["team_id"] == team) & (inp.rosters["lineup_slot"] != "IR")]
    return [{"id": int(i), "pos": "DST" if p in ("D/ST", "DST") else p, "name": n, "v": V.get(int(i), 0.0)}
            for i, p, n in zip(r["espn_id"], r["position"], r["name"])]


def _lineup(rows: List[dict]) -> float:
    by = {z["id"]: z["v"] for z in rows}
    return sum(by[i] for i in best_lineup(rows, key="v").values())


def view(sims: int = 2000) -> dict:
    inp = d.load()
    pt = power_table(inp)
    teams = [t["id"] for t in pt["teams"]]
    W = int(pt["through_week"])
    if W < 3:
        return {"season": inp.season, "through_week": W, "teams": [], "history": [], "notes": NOTES,
                "message": "Power needs three played weeks."}
    tw = inp.team_weekly[inp.team_weekly["week"] <= W]
    pts = {(int(w), int(t)): float(p) for w, t, p in zip(tw["week"], tw["team_id"], tw["points"])}
    roster_now = int(inp.rosters["week"].max())

    # strength by week: roster during week w, values known before week w+1
    st: Dict[int, Dict[int, float]] = {}
    for w in range(1, W + 1):
        V = _values(inp, w + 1)
        st[w] = {t: _lineup(_rows(inp, w, t, V)) for t in teams}
    Vnow = _values(inp, W + 1)
    now_rows = {t: _rows(inp, roster_now, t, Vnow) for t in teams}
    st_now = {t: _lineup(now_rows[t]) for t in teams}

    def power_at(w: int, cur: Optional[Dict[int, float]] = None) -> Dict[int, dict]:
        a = strength_weight(w)
        out = {}
        for t in teams:
            s = np.mean([(cur or st[w])[t]] + [st[w - j][t] for j in range(1, SMOOTH) if w - j >= 1])
            pf = np.mean([pts[(k, t)] for k in range(1, w + 1)])
            out[t] = {"strength": float(s), "pf_g": float(pf), "power": float(a * s + (1 - a) * pf)}
        return out
    hist = {w: power_at(w) for w in range(3, W)}
    hist[W] = power_at(W, st_now)
    cur, prev = hist[W], hist.get(W - 1)
    tier = tiers({t: cur[t]["power"] for t in teams})
    rank = {t: k + 1 for k, t in enumerate(sorted(teams, key=lambda t: -cur[t]["power"]))}
    prank = {t: k + 1 for k, t in enumerate(sorted(teams, key=lambda t: -prev[t]["power"]))} if prev else None

    # luck ledger
    rec = _records(inp)
    pairs = _pairs(inp)
    weeks = list(range(1, W + 1))
    ap = {t: 0.0 for t in teams}
    for w in weeks:
        sc = [pts[(w, t)] for t in teams]
        for t in teams:
            ap[t] += (sum(pts[(w, t)] > x for x in sc) + 0.5 * (sum(pts[(w, t)] == x for x in sc) - 1)) / (len(teams) - 1)
    opp_vs = {t: [] for t in teams}
    pf_g = {t: cur[t]["pf_g"] for t in teams}
    for w in weeks:
        for a, h in pairs.get(w, []):
            if (w, a) in pts and (w, h) in pts:
                opp_vs[a].append(pts[(w, h)] - pf_g[h])
                opp_vs[h].append(pts[(w, a)] - pf_g[a])
    ptab = {t["id"]: t for t in pt["teams"]}
    try:
        alg = {x["id"]: x for x in analytics.league().get("teams", [])}
    except Exception:  # noqa: BLE001 - lineup cost is optional context
        alg = {}

    # odds + what one more pt/wk buys (same draws as the base sim)
    odds = league.base(sims)
    ctx = league._ctx()
    rem = list(ctx.reg_weeks) + list(ctx.playoff_weeks)
    dk = ("power-bonus", "seed-by-record", sims, league.SEED, BONUS, inp.week, d.ages_key())
    hit = store.get(f"power_bonus_{sims}", dk)
    if hit is not None:
        per_pt = {int(t): x for t, x in hit[0].items()}
    else:
        per_pt = {}
        with league._lock:
            for t in teams:
                b = league.simulate_crn({}, sims, bonus={(t, w): BONUS for w in rem})[t]["title"]
                per_pt[t] = round((b - odds[t]["title"]) / BONUS, 4)
        store.put(f"power_bonus_{sims}", dk, per_pt)

    out = []
    for t in teams:
        c = cur[t]
        rows = now_rows[t]
        tot = _lineup(rows)
        chosen = best_lineup(rows, key="v")
        top = max(chosen.values(), key=lambda i: next(z["v"] for z in rows if z["id"] == i)) if chosen else None
        worst = 1 - _lineup([z for z in rows if z["id"] != top]) / tot if tot > 0 and top is not None else 0.0
        ids = lambda wk: {int(i) for i in inp.rosters[(inp.rosters["week"] == wk) & (inp.rosters["team_id"] == t)]["espn_id"]}
        old, new = ids(max(1, roster_now - 3)), ids(roster_now)  # whole rosters, IR included: a move to IR isn't a move
        nm = dict(zip(inp.rosters["espn_id"].astype(int), inp.rosters["name"]))
        moves = sorted([{"name": nm.get(i, str(i)), "dir": "in", "v": round(Vnow.get(i, 0.0), 1)} for i in new - old] +
                       [{"name": nm.get(i, str(i)), "dir": "out", "v": round(Vnow.get(i, 0.0), 1)} for i in old - new],
                       key=lambda m: -m["v"])[:3]
        r = rec.get(t, {"w": 0, "l": 0, "t": 0})
        n = max(1, len(weeks))
        proj_g = (ptab[t]["proj_pf"] or 0.0) / n
        a = strength_weight(W)
        d_rank = (prank[t] - rank[t]) if prank else 0
        if prank and d_rank:
            ds, dp = a * (c["strength"] - prev[t]["strength"]), (1 - a) * (c["pf_g"] - prev[t]["pf_g"])
            driver = "roster" if abs(ds) > abs(dp) else "scoring"
        else:
            driver = None
        traj = c["strength"] - hist[W - 3][t]["strength"] if (W - 3) in hist else None
        o = odds.get(t, {})
        out.append({
            "id": t, "name": inp.team_name(t), "power": round(c["power"], 1), "rank": rank[t], "tier": tier[t],
            "rank_change": d_rank, "driver": driver,
            "strength": round(c["strength"], 1), "pf_g": round(c["pf_g"], 1), "weight": round(a, 2),
            "record": {"w": r["w"], "l": r["l"], "t": r["t"]},
            "ledger": {"wins": r["w"] + 0.5 * r["t"], "allplay_wins": round(ap[t], 2),
                       "sched_luck": round(r["w"] + 0.5 * r["t"] - ap[t], 2),
                       "proj_g": round(proj_g, 1), "over_g": round(c["pf_g"] - proj_g, 1),
                       "opp_over": round(float(np.mean(opp_vs[t])) if opp_vs[t] else 0.0, 1),
                       "lineup_cost_g": round(alg[t]["vs_espn"] / n, 1) if t in alg else None},
            "fragility": {"share": round(worst, 3), "player": nm.get(top) if top is not None else None},
            "trajectory": round(traj, 1) if traj is not None else None, "moves": moves,
            "odds": {"playoffs": o.get("playoffs"), "seed1": o.get("seed1"), "title": o.get("title"),
                     "reg_wins": o.get("reg_wins"), "title_per_pt": per_pt.get(t)},
            "history": [{"week": w, "power": round(hist[w][t]["power"], 1)} for w in sorted(hist)],
        })
    out.sort(key=lambda z: z["rank"])
    return {"season": inp.season, "through_week": W, "roster_week": roster_now, "sims": sims,
            "weight": round(strength_weight(W), 2), "tier_gap": TIER_GAP, "good_trade": GOOD_TRADE,
            "logit_b": LOGIT_B, "p_cap": P_CAP, "teams": out, "notes": NOTES}
