"""Points-based trade evaluation and search for the live season (experimental).

Currency: change in each team's expected starting-lineup points for the rest of
the season (valuation.py), split into regular season (weeks now-14, drives
points-based seeding) and fantasy playoffs (15-17). Backtested on 1,600
hypothetical trades across 2021-2025:
- realized changes averaged ~0.90x the predicted change when realized lineups are
  picked by ESPN projections (1.00x with hindsight-perfect lineups), so every number
  shown is also given "calibrated";
- how often the predicted direction was right depends on the size of the edge
  (see ``win_probability``) -- small edges are close to coin flips.

Acceptance: not a fitted model (only ~20 answered proposals exist, 2021-2025).
History (backtest/trade_proposals.py, re-run after the 2026-09-23 data audit):
3 of 4 accepted proposals gave the *responder* a gain by this model (median +2.6)
vs 2 of 8 declined ones (median -5.2); one-sided Fisher p = 0.15 -- a weak hint,
not evidence. The finder requires the partner to gain at least a little
(``min_their_gain``); treat bigger partner gains as more likely to be accepted
only as an assumption.

League size: 10 teams in 2026 (12 in 2021-2025). Replacement level comes from
the live waiver wire, so the shallower league is built in; the calibration
numbers come from 12-team seasons and are an approximation here.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import pandas as pd

import valuation as v
from data import normalize_name

SKILL = ("QB", "RB", "WR", "TE")


@dataclass
class Context:
    season: v.Season
    week: int
    params: v.ForecastParams
    tbl: pd.DataFrame
    rosters: Dict[int, List[int]]
    replacement: Dict[str, float]
    calibration: dict
    # players sitting in an IR lineup slot: they don't use a bench spot, so they're
    # neither counted against the roster limit nor dropped to make room
    ir_slot: frozenset = frozenset()

    @property
    def reg_weeks(self) -> range:
        return range(self.week, self.season.reg_weeks + 1)

    @property
    def playoff_weeks(self) -> range:
        return range(max(self.week, self.season.reg_weeks + 1), v.FANTASY_LAST_WEEK + 1)

    def team_name(self, team_id: int) -> str:
        return self.season.settings.get("team_names", {}).get(str(team_id), f"Team {team_id}")


def load_context(season: int = 2026, week: Optional[int] = None,
                 params: Optional[v.ForecastParams] = None) -> Context:
    v.load_season.cache_clear()
    s = v.load_season(season)
    week = week or int(s.rosters["week"].max())
    params = params or v.ForecastParams.load()
    r = s.roster_at(week)
    tbl = v.player_table(s, week, params, live_injuries=True, rostered=set(r["espn_id"]))
    rosters = {int(t): [int(i) for i in g["espn_id"] if int(i) in tbl.index] for t, g in r.groupby("team_id")}
    repl = v.waiver_replacement(tbl, [i for ids in rosters.values() for i in ids], scale=params.waiver_scale)
    cal = json.loads((v.ROOT / "valuation_params.json").read_text()).get("calibration", {})
    ir = frozenset(int(i) for i in r.loc[r["lineup_slot"] == "IR", "espn_id"]) if "lineup_slot" in r else frozenset()
    return Context(s, week, params, tbl, rosters, repl, cal, ir)


def win_probability(ctx: Context, edge: float) -> float:
    for lo, hi, acc in ctx.calibration.get("sign_accuracy_by_abs_pred_edge", []):
        if lo <= abs(edge) < hi:
            return float(acc)
    return 0.5


# Typical per-player miss on rest-of-season points per game (RMSE 3.6-4.2 at weeks 3-13 in the
# 2021-25 backtests, rolling target). Shown as a plain error band next to forecasts.
TYPICAL_MISS_PPG = 4.0


def edge_label(edge: float) -> str:
    """Backtest: edges < 10 are right 47-57% of the time (noise), 10-20 ~61% (thin)."""
    e = abs(edge)
    return "noise" if e < 10 else "thin edge" if e < 20 else "yes"


def why(ctx: Context, pid: int) -> str:
    """One line on where a player's number comes from: the three sources the forecast averages
    (our prior+observed rate, ESPN's weekly projection, the FantasyPros rank), the blended
    per-game estimate, and how often the model expects him to play."""
    r = ctx.tbl.loc[pid]
    parts = []
    if pd.notna(r.get("ours_ppg")):
        parts.append(f"ours {r['ours_ppg']:.1f}")
    if pd.notna(r.get("espn_now")) and r.get("espn_now", 0) > 0:
        parts.append(f"ESPN {r['espn_now']:.1f}")
    elif r.get("espn_now") == 0:
        parts.append("ESPN 0 (out?)")
    if pd.notna(r.get("ecr_rank")):
        parts.append(f"experts {r['position']}{int(r['ecr_rank'])}")
    s = " / ".join(parts)
    band = f" (+/-{TYPICAL_MISS_PPG:.0f})" if r["position"] in SKILL else ""  # band measured on skill players only
    s += f" -> {r['est_ppg']:.1f} ppg{band}, plays ~{100 * float(r['avail']):.0f}%"
    if r["position"] in ctx.replacement:  # what a waiver pickup at his position would give instead
        s += f"; waiver {r['position']} ~{ctx.replacement[r['position']]:.1f}"
    if str(r.get("status", "active")) != "active":
        s += f" [{r['status']}]"
    return s


def resolve(ctx: Context, team_id: int, names: Sequence[str]) -> List[int]:
    out = []
    for n in names:
        key = normalize_name(n)
        hits = [i for i in ctx.rosters[team_id] if normalize_name(ctx.tbl.at[i, "name"]) == key]
        if not hits:
            raise ValueError(f"{n!r} is not on {ctx.team_name(team_id)}'s roster")
        out.append(hits[0])
    return out


def _after(ctx: Context, roster: List[int], give: List[int], get: List[int]) -> List[int]:
    """Roster after a trade. If it now exceeds the roster limit, drop the cheapest players:
    never one just acquired or one in an IR slot (dropping him frees no bench spot); a
    backup K or D/ST (worth one bye week) goes first, then the lowest-forecast players."""
    new = [i for i in roster if i not in give] + list(get)
    ir = ctx.ir_slot
    extra = sum(i not in ir for i in new) - sum(i not in ir for i in roster)
    if extra > 0:  # roster limit
        exp = v.expected_ros_points(ctx.tbl.loc[new], ctx.week, v.FANTASY_LAST_WEEK)
        pos = {i: ctx.tbl.at[i, "position"] for i in new}
        best = {}
        for i in new:
            if pos[i] in ("K", "DST") and exp[i] > exp.get(best.get(pos[i]), -1.0):
                best[pos[i]] = i

        def cost(i):
            return 0.0 if pos[i] in ("K", "DST") else float(exp[i])

        droppable = [i for i in new if i not in get and i not in ir
                     and (pos[i] not in ("K", "DST") or best[pos[i]] != i)]
        for pid in sorted(droppable, key=cost)[:extra]:
            new.remove(pid)
    return new


def _value(ctx: Context, roster: List[int], weeks, sims: int) -> float:
    if len(weeks) == 0:
        return 0.0
    return v.team_value(roster, ctx.tbl, ctx.season.slots, weeks, ctx.replacement, sims=sims, seed=7)


def evaluate(ctx: Context, team_a: int, give_a: List[int], team_b: int, give_b: List[int], sims: int = 300) -> dict:
    out = {"week": ctx.week, "teams": {}}
    exp = v.expected_ros_points(ctx.tbl.loc[list(give_a) + list(give_b)], ctx.week, v.FANTASY_LAST_WEEK)
    for team, give, get in ((team_a, give_a, give_b), (team_b, give_b, give_a)):
        before = ctx.rosters[team]
        after = _after(ctx, before, give, get)
        reg = _value(ctx, after, ctx.reg_weeks, sims) - _value(ctx, before, ctx.reg_weeks, sims)
        po = _value(ctx, after, ctx.playoff_weeks, sims) - _value(ctx, before, ctx.playoff_weeks, sims)
        total = reg + po
        dropped = [i for i in before if i not in after and i not in give]
        out["teams"][team] = {
            "name": ctx.team_name(team),
            "gives": [ctx.tbl.at[i, "name"] for i in give],
            "gets": [ctx.tbl.at[i, "name"] for i in get],
            "would_drop": [ctx.tbl.at[i, "name"] for i in dropped],
            "delta_regular_season": round(reg, 1),
            "delta_playoffs": round(po, 1),
            "delta_total": round(total, 1),
            # the players' own rest-of-season points swapped, ignoring who would start; the rest of
            # delta_total is lineup fit (bench, byes, drops, waiver streaming)
            "raw_swap": round(float(sum(exp[i] for i in get) - sum(exp[i] for i in give)), 1),
            "delta_total_calibrated": round(total * ctx.calibration.get("realized_over_predicted", 1.0), 1),
            "p_direction_right": win_probability(ctx, total),
            "edge_label": edge_label(total),
            "why": {ctx.tbl.at[i, "name"]: why(ctx, i) for i in list(give) + list(get)},
        }
    return out


def source_params(source: str) -> v.ForecastParams:
    """Forecast params using one talent source alone for QB/RB/WR/TE (a stress test, not a
    validated forecaster): "ours" (prior + observed), "espn" (ESPN's weekly number), "ecr"
    (FantasyPros consensus rank mapped onto our scale); anything else = the normal blend."""
    import copy
    p = copy.deepcopy(v.ForecastParams.load())
    sk = {x: 0.0 for x in SKILL}
    if source == "ours":
        p.espn_weight, p.ecr_weight = {**p.espn_weight, **sk}, {}
    elif source == "espn":
        p.espn_weight, p.ecr_weight = {**p.espn_weight, **{x: 1.0 for x in SKILL}}, {}
    elif source == "ecr":
        p.ecr_weight = {x: 1.0 for x in SKILL}
    return p


WIDE_SHAPES = ("1-1", "2-1", "1-2", "2-2", "3-1", "3-2")


def find(ctx: Context, team: int, *, shapes=("1-1", "2-1", "1-2"), sims: int = 40, rescore_sims: int = 300,
         top: int = 15, min_edge: float = 20.0, min_their_gain: float = 0.0,
         partners: Optional[Sequence[int]] = None) -> pd.DataFrame:
    """Search trades for ``team``. Keeps trades where both sides gain by our numbers,
    ranks by our calibrated gain. ``min_edge`` flags trades big enough to trust
    (backtest: <20 predicted points is right only ~60% of the time).

    ``WIDE_SHAPES`` adds 2-for-2, 3-for-1 and 3-for-2 (pruned: you must get someone clearly
    better than anyone you give, 10+ ROS points; 3-player packages exclude your 250+ point
    stars). Those shapes are slower (minutes) and outside the calibration backtest, which
    covered 1-for-1 and 2-for-1 only (2-for-1 calibrated like 1-for-1)."""
    exp = v.expected_ros_points(ctx.tbl, ctx.week, v.FANTASY_LAST_WEEK)
    mine = [i for i in ctx.rosters[team] if ctx.tbl.at[i, "position"] in SKILL]
    base = {t: _value(ctx, ids, range(ctx.week, v.FANTASY_LAST_WEEK + 1), sims) for t, ids in ctx.rosters.items()}
    cands = []
    for other, theirs_all in ctx.rosters.items():
        if other == team or (partners and other not in partners):
            continue
        theirs = [i for i in theirs_all if ctx.tbl.at[i, "position"] in SKILL]
        for shape in shapes:
            if shape == "2-1":  # consolidate: two of mine for one better player
                for pair in itertools.combinations(mine, 2):
                    cands.extend((other, list(pair), [g]) for g in theirs if exp[g] > max(exp[pair[0]], exp[pair[1]]))
            if shape == "1-2":
                for g in mine:
                    cands.extend((other, [g], list(pair)) for pair in itertools.combinations(theirs, 2)
                                 if exp[g] > max(exp[pair[0]], exp[pair[1]]))
            if shape == "2-2":  # upgrade plus a filler coming back
                for mp in itertools.combinations(mine, 2):
                    mx, ms = max(exp[i] for i in mp), sum(exp[i] for i in mp)
                    for tp in itertools.combinations(theirs, 2):
                        if max(exp[i] for i in tp) >= mx + 10 and -90 < sum(exp[i] for i in tp) - ms < 30:
                            cands.append((other, list(mp), list(tp)))
            if shape in ("3-1", "3-2"):
                pool = [i for i in mine if exp[i] < 250]
                for mt in itertools.combinations(pool, 3):
                    mx, ms = max(exp[i] for i in mt), sum(exp[i] for i in mt)
                    if shape == "3-1":
                        cands.extend((other, list(mt), [g]) for g in theirs if exp[g] > mx + 15)
                        continue
                    for tp in itertools.combinations(theirs, 2):
                        if (max(exp[i] for i in tp) >= mx + 20 and min(exp[i] for i in tp) <= 150
                                and -170 < sum(exp[i] for i in tp) - ms < -40):
                            cands.append((other, list(mt), list(tp)))
        if "1-1" in shapes:
            cands.extend((other, [g], [t]) for g in mine for t in theirs)
    rows = []
    weeks = range(ctx.week, v.FANTASY_LAST_WEEK + 1)
    for other, give, get in cands:
        da = _value(ctx, _after(ctx, ctx.rosters[team], give, get), weeks, sims) - base[team]
        if da <= 0:
            continue
        db = _value(ctx, _after(ctx, ctx.rosters[other], get, give), weeks, sims) - base[other]
        if db <= 0:
            continue
        rows.append((da, other, give, get))
    rows.sort(key=lambda r: -r[0])
    out = []
    for _, other, give, get in rows[: top * 3]:
        ev = evaluate(ctx, team, give, other, get, sims=rescore_sims)
        me, them = ev["teams"][team], ev["teams"][other]
        if me["delta_total"] <= 0 or them["delta_total"] <= max(0.0, min_their_gain):
            continue
        out.append({
            "with": them["name"], "give": ", ".join(me["gives"]), "get": ", ".join(me["gets"]),
            "you_drop": ", ".join(me["would_drop"]),
            "your_gain": me["delta_total"], "your_gain_calibrated": me["delta_total_calibrated"],
            "your_gain_playoffs": me["delta_playoffs"], "p_right": me["p_direction_right"],
            "their_gain": them["delta_total"], "trust": edge_label(me["delta_total"]),
            "why": " | ".join(f"{n}: {w}" for n, w in me["why"].items()),
        })
        if len(out) >= top:
            break
    return pd.DataFrame(out)


def around(ctx: Context, espn: Context, team: int, pid: int, *, sims: int = 40, rescore_sims: int = 300,
           top: int = 12) -> pd.DataFrame:
    """Deals built around one player. If ``pid`` is on ``team``: shop him to every other team
    (him for 1 or 2 of theirs, or him plus a filler for one better player). Otherwise: what
    ``team`` could send his owner to land him (1 or 2 players, or a 2-for-2 with a filler coming
    back). Keeps deals the partner gains from on ESPN's numbers (``espn``, what their app shows;
    accepted offers in this league have looked fair there), ranked by ``team``'s gain on ours,
    which may be negative: then it is the cheapest price, not a recommendation."""
    exp = v.expected_ros_points(ctx.tbl, ctx.week, v.FANTASY_LAST_WEEK)
    weeks = range(ctx.week, v.FANTASY_LAST_WEEK + 1)
    skill = lambda t: [i for i in ctx.rosters[t] if ctx.tbl.at[i, "position"] in SKILL]
    mine = skill(team)
    cands = []
    if pid in ctx.rosters[team]:
        rest = [i for i in mine if i != pid]
        for other in ctx.rosters:
            if other == team:
                continue
            theirs = skill(other)
            cands += [(other, [pid], [g]) for g in theirs]
            cands += [(other, [pid], list(pr)) for pr in itertools.combinations(theirs, 2) if exp[pid] > max(exp[i] for i in pr)]
            cands += [(other, [pid, x], [g]) for x in rest for g in theirs if exp[g] > max(exp[pid], exp[x])]
    else:
        other = next(t for t, ids in ctx.rosters.items() if pid in ids)
        filler = sorted((i for i in skill(other) if i != pid and exp[i] < exp[pid]), key=lambda i: -exp[i])[:6]
        cands += [(other, [g], [pid]) for g in mine]
        cands += [(other, list(pr), [pid]) for pr in itertools.combinations(mine, 2)]
        cands += [(other, list(pr), [pid, f]) for pr in itertools.combinations(mine, 2) for f in filler
                  if max(exp[i] for i in pr) < exp[pid]]
    base = {}

    def gain(c: Context, t: int, give, get):
        if (id(c), t) not in base:
            base[(id(c), t)] = _value(c, c.rosters[t], weeks, sims)
        return _value(c, _after(c, c.rosters[t], give, get), weeks, sims) - base[(id(c), t)]

    rows = []
    for other, give, get in cands:
        if gain(espn, other, get, give) < 0:
            continue
        rows.append((gain(ctx, team, give, get), other, give, get))
    rows.sort(key=lambda r: -r[0])
    seen: Dict[tuple, int] = {}  # at most two deals per package sent, so fillers don't crowd the list
    rows = [r for r in rows if seen.setdefault((r[1], tuple(r[2])), 0) < 2 and not seen.update({(r[1], tuple(r[2])): seen[(r[1], tuple(r[2]))] + 1})]
    out = []
    for _, other, give, get in rows[: top * 2]:
        ev, ee = evaluate(ctx, team, give, other, get, sims=rescore_sims), evaluate(espn, team, give, other, get, sims=rescore_sims)
        me, them, them_e = ev["teams"][team], ev["teams"][other], ee["teams"][other]
        if them_e["delta_total"] < 0:
            continue
        out.append({
            "with": them["name"], "partner": other, "give": ", ".join(me["gives"]), "get": ", ".join(me["gets"]),
            "give_ids": list(give), "get_ids": list(get), "you_drop": ", ".join(me["would_drop"]),
            "your_gain": me["delta_total"], "your_gain_calibrated": me["delta_total_calibrated"],
            "your_gain_playoffs": me["delta_playoffs"], "p_right": me["p_direction_right"],
            "their_gain": them["delta_total"], "their_gain_espn": them_e["delta_total"], "trust": edge_label(me["delta_total"]),
            "why": " | ".join(f"{n}: {w}" for n, w in me["why"].items()),
        })
    out.sort(key=lambda r: -r["your_gain"])
    return pd.DataFrame(out[:top])


def _shape_cands(exp, mine: List[int], theirs: List[int], shape: str) -> List[tuple]:
    """(give, get) packages of one shape, pruned the way ``find`` prunes: in uneven deals the
    side sending more players must get the best one back."""
    a, b = (int(x) for x in shape.split("-"))
    gives, gets = list(itertools.combinations(mine, a)), list(itertools.combinations(theirs, b))
    out = []
    for g in gives:
        gx, gs = max(exp[i] for i in g), sum(exp[i] for i in g)
        for t in gets:
            tx, ts = max(exp[i] for i in t), sum(exp[i] for i in t)
            if a > b and tx <= gx + (15 if a == 3 else 0):
                continue
            if b > a and gx <= tx:
                continue
            if a == b > 1 and not (tx >= gx + 10 and -90 < ts - gs < 30):
                continue
            out.append((list(g), list(t)))
    return out


def search(ctx: Context, espn: Context, team: int, *, shapes=("1-1", "2-1", "1-2"), give_pool: Optional[Sequence[int]] = None,
           keep: Sequence[int] = (), must_give: Sequence[int] = (), must_get: Sequence[int] = (),
           get_pos: Sequence[str] = SKILL, partners: Optional[Sequence[int]] = None, horizon: str = "ros",
           desk_weight: float = 0.5, min_gain: float = 0.0, min_their: float = 0.0, their_view: str = "espn",
           sort: str = "gain", sims: int = 20, rescore_sims: int = 300, top: int = 25, max_cands: int = 12000) -> dict:
    """Trade finder with user-set filters and valuation. Every deal is scored on our model
    (``ctx``) and on ESPN's numbers (``espn``, what the partner's app shows). Your gain is
    ``desk_weight`` x ours + (1 - desk_weight) x ESPN's over ``horizon`` ("ros", "reg" or
    "playoffs"). The partner must gain ``min_their`` on ``their_view`` ("espn", "desk" or
    "both"). Sort: "gain" (yours), "their" (partner, on their view) or "fair" (the smaller of
    the two). Descriptive: Desk's gap with ESPN was overweighted ~4x in the weekly backtest, so a
    deal that only wins on Desk deserves suspicion."""
    exp = v.expected_ros_points(ctx.tbl, ctx.week, v.FANTASY_LAST_WEEK)
    weeks = {"ros": range(ctx.week, v.FANTASY_LAST_WEEK + 1), "reg": ctx.reg_weeks, "playoffs": ctx.playoff_weeks}[horizon]
    pos = lambda i: ctx.tbl.at[i, "position"]
    mine = [i for i in ctx.rosters[team] if pos(i) in SKILL and i not in keep and (give_pool is None or i in give_pool or i in must_give)]
    cands = []
    for other, ids in ctx.rosters.items():
        if other == team or (partners and other not in partners) or (must_get and not set(must_get) & set(ids)):
            continue
        theirs = [i for i in ids if pos(i) in SKILL and (pos(i) in get_pos or i in must_get)]
        for shape in shapes:
            cands += [(other, g, t) for g, t in _shape_cands(exp, mine, theirs, shape)
                      if (not must_give or set(must_give) & set(g)) and (not must_get or set(must_get) & set(t))]
    truncated = len(cands) > max_cands
    if truncated:  # keep the closest swaps on raw points: lopsided ones rarely survive the filters
        cands.sort(key=lambda c: abs(sum(exp[i] for i in c[2]) - sum(exp[i] for i in c[1])))
        cands = cands[:max_cands]
    base: Dict[tuple, float] = {}

    def gain(c: Context, t: int, give, get, n: int):
        if (id(c), t, n) not in base:
            base[(id(c), t, n)] = _value(c, c.rosters[t], weeks, n)
        return _value(c, _after(c, c.rosters[t], give, get), weeks, n) - base[(id(c), t, n)]

    def their(d, e):
        return {"espn": e, "desk": d, "both": min(d, e)}[their_view]

    w = desk_weight
    rows = []
    for other, give, get in cands:
        te = gain(espn, other, get, give, sims)
        if their_view != "desk" and te < min_their - 5:  # small slack: the cheap pass is noisy
            continue
        td = gain(ctx, other, get, give, sims) if their_view != "espn" else 0.0
        if their_view != "espn" and their(td, te) < min_their - 5:
            continue
        me = w * gain(ctx, team, give, get, sims) + (1 - w) * gain(espn, team, give, get, sims)
        if me < min_gain - 5:
            continue
        rows.append((me, their(td, te), other, give, get))
    key = {"gain": lambda r: -r[0], "their": lambda r: -r[1], "fair": lambda r: -min(r[0], r[1])}[sort]
    rows.sort(key=lambda r: (r[0] < min_gain or r[1] < min_their, key(r)))  # clear passes first, slack ones after
    hk = {"ros": "delta_total", "reg": "delta_regular_season", "playoffs": "delta_playoffs"}[horizon]
    seen: Dict[tuple, int] = {}  # at most two deals per player(s) you'd get, so fillers don't crowd the list
    rows = [r for r in rows if seen.setdefault((r[2], tuple(r[4])), 0) < 2 and not seen.update({(r[2], tuple(r[4])): seen[(r[2], tuple(r[4]))] + 1})]
    out = []
    for _, _, other, give, get in rows[: top * 2]:
        ev, ee = evaluate(ctx, team, give, other, get, sims=rescore_sims), evaluate(espn, team, give, other, get, sims=rescore_sims)
        me, them, me_e, them_e = ev["teams"][team], ev["teams"][other], ee["teams"][team], ee["teams"][other]
        g = w * me[hk] + (1 - w) * me_e[hk]
        tg = their(them[hk], them_e[hk])
        if g < min_gain or tg < min_their:
            continue
        out.append({
            "with": them["name"], "partner": other, "give_ids": list(give), "get_ids": list(get),
            "give": [{"id": i, "name": ctx.tbl.at[i, "name"], "pos": pos(i), "ros": round(float(exp[i]), 1)} for i in give],
            "get": [{"id": i, "name": ctx.tbl.at[i, "name"], "pos": pos(i), "ros": round(float(exp[i]), 1)} for i in get],
            "shape": f"{len(give)}-{len(get)}", "you_drop": ", ".join(me["would_drop"]),
            "your_gain": round(g, 1), "desk_gain": me[hk], "espn_gain": me_e[hk], "their_gain": round(tg, 1),
            "their_desk": them[hk], "their_espn": them_e[hk], "playoffs_gain": round(w * me["delta_playoffs"] + (1 - w) * me_e["delta_playoffs"], 1),
            # players' own ROS points swapped (Desk), and the rest of Desk's ROS gain: lineup fit
            "players": me["raw_swap"], "fit": round(me["delta_total"] - me["raw_swap"], 1),
            "p_right": win_probability(ctx, g), "trust": edge_label(g),
            "why": " | ".join(f"{n}: {x}" for n, x in me["why"].items()),
        })
    out.sort(key={"gain": lambda r: -r["your_gain"], "their": lambda r: -r["their_gain"],
                  "fair": lambda r: -min(r["your_gain"], r["their_gain"])}[sort])
    return {"rows": out[:top], "candidates": len(cands), "kept": len(rows), "truncated": truncated}
