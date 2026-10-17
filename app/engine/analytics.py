"""Season analytics for Desk's Analysis tabs and the Lab.

Usage vs results, consistency, trends and league context for every QB/RB/WR/TE, plus the league's
team-by-week table. Every stat carries the label the 2021-25 backtest gave it
(backtest/analytics_metrics.py writes analytics_registry.json next to this file): whether it repeats,
whether it predicts later points, and whether it knows anything the model doesn't. Read-only: reads
the files the weekly flow and Desk's refresh already write.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import nflverse_data as nd  # noqa: E402

ACT = ROOT / "history" / "actuals"
NFL = ROOT / "history" / "nflverse"
LEDGER = ROOT / "history" / "ledger"
REG = Path(__file__).with_name("analytics_registry.json")
SKILL = ["QB", "RB", "WR", "TE"]
LAST = 17


# ------------------------------------------------------------------------------- shared with the backtest
def tiers(n_teams: int) -> dict:
    """Weekly finish tiers sized to the league: a startable week, a smash week, a dud week."""
    start = {"QB": n_teams, "TE": n_teams, "RB": round(2.5 * n_teams), "WR": round(2.5 * n_teams)}
    smash = {"QB": 3, "TE": 3, "RB": round(0.25 * start["RB"]), "WR": round(0.25 * start["WR"])}
    dud = {k: 2 * v for k, v in start.items()}
    return {"start": start, "smash": smash, "dud": dud}


def n_teams(season: int) -> int:
    return int(json.loads((ACT / str(season) / "settings.json").read_text())["teams"])


def weekly(season: int) -> pd.DataFrame:
    """One row per player-game played: ESPN points and projection, nflverse usage, weekly finish tier."""
    P = pd.read_csv(ACT / str(season) / "players_weekly.csv")
    P = P[(P["kind"] == "actual") & P["position"].isin(SKILL) & P["week"].between(1, LAST)]
    P = P[["espn_id", "name", "position", "pro_team_id", "week", "points", "played"]].copy()
    P["played"] = P["played"].astype(bool)
    E = pd.read_csv(ACT / str(season) / "espn_weekly_proj.csv")
    P = P.merge(E.drop_duplicates(["week", "espn_id"]), on=["week", "espn_id"], how="left")
    U = nd.usage_table(season)
    U["team_car"] = U.groupby(["team", "week"])["carries"].transform("sum")
    U["csh"] = U["carries"] / U["team_car"].replace(0, np.nan)
    U["fpoe"] = U["fp_opp"] - U["xfp"]
    U["opp"] = U["carries"].fillna(0) + U["targets"].fillna(0)
    cols = ["espn_id", "week", "xfp", "fpoe", "opp", "offense_pct", "target_share", "air_yards_share", "wopr", "csh"]
    P = P.merge(U[cols].drop_duplicates(["espn_id", "week"]), on=["espn_id", "week"], how="left")
    P = P[P["played"]].copy()
    t = tiers(n_teams(season))
    P["rank"] = P.groupby(["week", "position"])["points"].rank(ascending=False, method="min")
    for k in ("start", "smash"):
        P[k] = (P["rank"] <= P["position"].map(t[k])).astype(float)
    P["dud"] = (P["rank"] > P["position"].map(t["dud"])).astype(float)
    P["resid"] = P["points"] - P["espn_proj"]
    P["beat"] = np.where(P["espn_proj"].notna(), (P["points"] > P["espn_proj"]).astype(float), np.nan)
    P["season"] = season
    return P


def summarize(G: pd.DataFrame) -> pd.DataFrame:
    g = G.groupby("espn_id")
    o = pd.DataFrame({
        "n": g.size(), "ppg": g["points"].mean(), "xfp": g["xfp"].mean(), "fpoe": g["fpoe"].mean(),
        "opp": g["opp"].mean(), "snap": g["offense_pct"].mean(), "tsh": g["target_share"].mean(),
        "ays": g["air_yards_share"].mean(), "wopr": g["wopr"].mean(), "csh": g["csh"].mean(),
        "sd": g["points"].std(), "smash": g["smash"].mean(), "start": g["start"].mean(), "dud": g["dud"].mean(),
        "p20": g["points"].quantile(0.2), "p80": g["points"].quantile(0.8),
        "resid": g["resid"].mean(), "beat": g["beat"].mean(),
    })
    o["cv"] = o["sd"] / o["ppg"].clip(lower=1)
    o["spread"] = (o["p80"] - o["p20"]) / o["ppg"].clip(lower=1)
    return o


TREND_COLS = {"xfp": "xfp", "snap": "offense_pct", "tsh": "target_share", "opp": "opp"}


def trends(G: pd.DataFrame, season_summary: pd.DataFrame) -> pd.DataFrame:
    """Last three games minus the season so far, for the usage columns."""
    l3 = G.sort_values("week").groupby("espn_id").tail(3).groupby("espn_id")
    return pd.DataFrame({f"{k}_tr": l3[c].mean() - season_summary[k] for k, c in TREND_COLS.items()})


# ------------------------------------------------------------------------------------------ registry
_REG: dict = {"mtime": None, "v": None}


def registry() -> dict:
    mt = REG.stat().st_mtime if REG.exists() else None
    if _REG["mtime"] != mt:
        _REG.update(mtime=mt, v=json.loads(REG.read_text()) if mt else {})
    return _REG["v"]


READ = {
    "ppg": "Repeats strongly (r {rep}). The model is built on it.",
    "xfp": "Points his usage usually produces. Repeats (r {rep}) and predicts later scoring (r {pred}); the model already reflects it.",
    "fpoe": "Scoring over or under what his usage usually produces barely repeats (r {rep}). About {sticks}% of the gap has carried forward, and the model already discounts it.",
    "opp": "Targets plus carries. Repeats (r {rep}) and predicts later points (r {pred}); already in the model.",
    "snap": "Repeats (r {rep}) and predicts later points (r {pred}); already in the model.",
    "tsh": "Repeats (r {rep}) and predicts later points (r {pred}); already in the model.",
    "ays": "Repeats (r {rep}); predicts later points less than target share does (r {pred}). Already in the model.",
    "wopr": "Target share and air-yards share combined. Repeats (r {rep}); already in the model.",
    "csh": "Share of his team's carries. Repeats (r {rep}); already in the model.",
    "cv": "How much his weekly score swings relative to his average. Repeats moderately (r {rep}); steadier players tend to be better players, which the model already knows.",
    "spread": "Gap between a bad week and a good week, relative to his average. Repeats weakly (r {rep}).",
    "smash": "Share of games in the top tier at his position. Repeats (r {rep}) because it tracks how good he is; already in the model.",
    "start": "Share of games good enough to start in this league. Repeats (r {rep}); already in the model.",
    "dud": "Share of games outside twice the startable pool. Repeats (r {rep}). Players with more of them trailed the model by {beyond} ppg per SD in past seasons, but that didn't hold out of sample.",
    "resid": "Points vs ESPN's projection. Doesn't repeat (r {rep}); no player stays under-projected.",
    "beat": "How often he beat ESPN's projection. Doesn't repeat (r {rep}).",
    "xfp_tr": "Last three games vs his season. About {sticks}% of a change sticks; no reliable edge beyond the model.",
    "snap_tr": "Last three games vs his season. About {sticks}% of a snap change sticks, and rising snaps beat the model by {beyond} ppg per SD, out of sample too. Real but small.",
    "tsh_tr": "Last three games vs his season. About {sticks}% sticks; no edge beyond the model.",
    "opp_tr": "Last three games vs his season. About {sticks}% sticks; no reliable edge beyond the model.",
    "est_tr": "Change in the model's value over three weeks. It tends to keep moving the same way: {beyond} ppg per SD beyond the model, out of sample too. Small.",
}


def _num(e: Optional[dict], k: str = "v") -> Optional[float]:
    return None if not e else e.get(k)


def metrics(pos: str = "ALL") -> Dict[str, dict]:
    """Label, numbers and a plain reading for every player stat, from the backtest."""
    R = registry()
    out = {}
    for key, m in (R.get("players") or {}).items():
        t = dict(m["by_pos"].get(pos) or m["by_pos"].get("ALL") or {})
        top = (m["by_pos"].get("ALL") or {}).get("label", m.get("label"))
        # a position cell's "beyond" was only tested in sample: it can't claim an edge (or a failed one)
        # the pooled stat doesn't have, so it shows the pooled label instead
        if t.get("label") in ("tilt", "edge", "adds") and t.get("label") != top and top not in ("noise",):
            t["label"] = top
        rep = t.get("repeat") or t.get("sticks")
        fill = {"rep": f"{_num(rep):.2f}" if rep else "–", "pred": f"{_num(t.get('predict')):.2f}" if t.get("predict") else "–",
                "beyond": f"{_num(t.get('beyond')):+.2f}" if t.get("beyond") else "–",
                "sticks": f"{100 * _num(t.get('sticks')):.0f}" if t.get("sticks") else "–"}
        out[key] = {"name": m["name"], "label": t.get("label", m.get("label")), "label_text": R.get("labels", {}).get(t.get("label", ""), ""),
                    "repeat": _num(t.get("repeat")), "sticks": _num(t.get("sticks")), "predict": _num(t.get("predict")),
                    "beyond": t.get("beyond"), "weekly": t.get("weekly"), "reading": READ.get(key, "").format(**fill)}
    return out


# ------------------------------------------------------------------------------------------ players
def complete_week(season: int, fallback: int) -> int:
    """Last fantasy week whose NFL games have all finished (partial weeks stay out of the analytics)."""
    try:
        s = pd.read_csv(NFL / "schedules.csv", usecols=["season", "game_type", "week", "home_score"])
        s = s[(s["season"] == season) & (s["game_type"] == "REG")]
        done = s.groupby("week")["home_score"].apply(lambda x: x.notna().all())
        w = 0
        for wk, ok in done.sort_index().items():
            if not ok:
                break
            w = int(wk)
        return max(0, min(w, LAST)) if w else fallback
    except Exception:
        return fallback


POOL = {"QB": 2.0, "TE": 2.0, "RB": 5.0, "WR": 5.0}   # x league size: the fantasy-relevant pool for percentiles
PCT_KEYS = ["ppg", "xfp", "fpoe", "opp", "snap", "tsh", "ays", "wopr", "csh", "cv", "smash", "start", "dud", "resid", "snap_tr", "xfp_tr", "est_tr"]
HOT = 3.0          # ppg over/under usage that earns a flag
SNAP_T = 0.10      # snap-share change (last 3 vs season) that earns a flag
EST_T = 1.0        # model value change (ppg, three weeks) that earns a flag


def _r(x, nd_=2):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else round(x, nd_)


def _ledger_est(season: int) -> tuple:
    files = sorted((LEDGER / str(season)).glob("wk*_players.csv"))
    if not files:
        return pd.Series(dtype=float), pd.Series(dtype=float), None, None
    cur = pd.read_csv(files[-1])
    wk = int(files[-1].stem[2:4])
    prev_f = LEDGER / str(season) / f"wk{wk - 3:02d}_players.csv"
    prev = pd.read_csv(prev_f) if prev_f.exists() else None
    c = cur.drop_duplicates("espn_id").set_index("espn_id")
    p = prev.drop_duplicates("espn_id").set_index("espn_id")["est_ppg"] if prev is not None else pd.Series(dtype=float)
    return c, p, wk, (wk - 3 if prev is not None else None)


def _inputs_key(season: int) -> tuple:
    files = [ACT / str(season) / f for f in ("players_weekly.csv", "espn_weekly_proj.csv", "rosters_by_week.csv", "team_weekly.csv")]
    files += [NFL / str(season) / f for f in ("opportunity.csv", "player_stats.csv", "snaps.csv")]
    files += [NFL / "schedules.csv", REG] + sorted((LEDGER / str(season)).glob("wk*_players.csv"))[-1:]
    return tuple((str(f.name), f.stat().st_mtime if f.exists() else None) for f in files)


_CACHE: dict = {}


def players(season: int = 2026) -> dict:
    key = ("players", season, _inputs_key(season))
    if _CACHE.get("pkey") == key:
        return _CACHE["players"]
    out = _players(season)
    _CACHE.update(pkey=key, players=out)
    return out


def _breakout(r, Sp: pd.DataFrame, i: int, done: int):
    """PPG well above last season (wk >= 4, 3+ games now, 6+ then): split the gain into usage (nflverse expected
    points, xfp) and efficiency (points over expected, fpoe). In 2021-25 ~60-70% of a usage gain carried over,
    under half of an efficiency gain (backtest/stabilization.py)."""
    if done < 4 or r["n"] < 3 or i not in Sp.index or Sp.at[i, "n"] < 6 or pd.isna(r["xfp"]) or pd.isna(Sp.at[i, "xfp"]):
        return None
    pv = Sp.at[i, "ppg"]
    if r["ppg"] - pv < max(3.0, 0.2 * pv):
        return None
    du, de = r["xfp"] - Sp.at[i, "xfp"], r["fpoe"] - Sp.at[i, "fpoe"]
    if du + de <= 0:
        return None
    u = min(1.0, max(0.0, du / (du + de)))
    return {"prev_ppg": _r(pv, 1), "gain": _r(r["ppg"] - pv, 1), "usage_pct": round(100 * u), "eff_pct": round(100 * (1 - u))}


def _players(season: int) -> dict:
    from . import data as d
    inp = d.load(season)
    nt = n_teams(season)
    done = complete_week(season, inp.week - 1)
    P = weekly(season)
    P = P[P["week"] <= done]
    S = summarize(P).join(trends(P, summarize(P)))
    led, prev, wk_now, wk_prev = _ledger_est(season)
    S["est_ppg"] = led["est_ppg"].reindex(S.index) if len(led) else np.nan
    S["est_tr"] = (S["est_ppg"] - prev.reindex(S.index)) if len(prev) else np.nan
    last = P.sort_values("week").drop_duplicates("espn_id", keep="last").set_index("espn_id")
    S["pos"], S["name"], S["pro_team_id"] = last["position"], last["name"], last["pro_team_id"]
    try:  # last season, for the breakout split
        Sp = summarize(weekly(season - 1))
    except Exception:
        Sp = pd.DataFrame(columns=["n", "ppg", "xfp", "fpoe"])
    ab = dict(zip(inp.pro_teams["pro_team_id"], inp.pro_teams["abbrev"]))
    ro = inp.rosters[inp.rosters["week"] == inp.rosters["week"].max()]
    owner = dict(zip(ro["espn_id"].astype(int), ro["team_id"].astype(int)))
    # fantasy-relevant pool for percentiles: top POOL x league size at the position by ppg, min games
    min_g = min(3, max(1, done))
    S["pool"] = False
    for pos, k in POOL.items():
        g = S[(S["pos"] == pos) & (S["n"] >= min_g)].sort_values("ppg", ascending=False)
        S.loc[g.index[: int(k * nt)], "pool"] = True
    pct = {}
    for c in PCT_KEYS:
        col = pd.Series(np.nan, index=S.index)
        for pos in SKILL:
            ref = S.loc[S["pool"] & (S["pos"] == pos), c].dropna().sort_values().to_numpy()
            if len(ref) < 5:
                continue
            m = (S["pos"] == pos) & S[c].notna()
            col[m] = 100 * np.searchsorted(ref, S.loc[m, c].to_numpy(), side="right") / len(ref)
        pct[c] = col
    # z within position among players the model values with 3+ games (as in the backtest) -> size of the small edges
    M = metrics()
    edge_v = {k: (M.get(k, {}).get("beyond") or {}).get("v") for k in ("snap_tr", "est_tr")}
    zs = {}
    for k in ("snap_tr", "est_tr"):
        z = pd.Series(np.nan, index=S.index)
        for pos in SKILL:
            g = S[(S["pos"] == pos) & (S["n"] >= 3) & S["est_ppg"].notna()][k]
            if g.notna().sum() >= 10 and g.std() > 0:
                m = S["pos"] == pos
                z[m] = (S.loc[m, k] - g.mean()) / g.std()
        zs[k] = z
    M_pos = {p: metrics(p) for p in SKILL}
    weeks = list(range(1, done + 1))
    PW = {i: g.set_index("week") for i, g in P.groupby("espn_id")}
    rows = []
    for i, r in S.iterrows():
        if r["pos"] not in SKILL:
            continue
        i = int(i)
        if not (r["pool"] or i in owner or (r["n"] >= 2 and (r["est_ppg"] if pd.notna(r["est_ppg"]) else 0) >= 3)):
            continue
        g = PW.get(i)
        wk = []
        for w in weeks:
            if g is None or w not in g.index:
                wk.append(None)
                continue
            x = g.loc[w]
            tier = "smash" if x["smash"] else "start" if x["start"] else "dud" if x["dud"] else "mid"
            # compact: [pts, ESPN proj, xFP, snap share, target share, opportunities, rank at position, tier]
            wk.append([_r(x["points"], 1), _r(x["espn_proj"], 1), _r(x["xfp"], 1), _r(x["offense_pct"], 3),
                       _r(x["target_share"], 3), _r(x["opp"], 0), int(x["rank"]), tier])
        flags = []
        n = int(r["n"])
        fp = r["fpoe"]
        mp = M_pos[r["pos"]]
        if n >= 3 and pd.notna(fp) and abs(fp) >= HOT:
            st = mp.get("fpoe", {}).get("sticks")
            flags.append({"key": "hot" if fp > 0 else "cold", "tone": "up" if fp > 0 else "down", "metric": "fpoe",
                          "short": f"{'Hot' if fp > 0 else 'Cold'} {fp:+.1f}",
                          "text": f"Scoring {abs(fp):.1f} a game {'over' if fp > 0 else 'under'} what his usage usually produces "
                                  f"({r['ppg']:.1f} vs {r['xfp']:.1f}). About {round(100 * (st or 0.09))}% of gaps like this has carried "
                                  f"forward; the model already discounts it.", "label": "noise"})
        st_ = r.get("snap_tr")
        if n >= 4 and pd.notna(st_) and abs(st_) >= SNAP_T:
            e = (edge_v["snap_tr"] or 0) * (zs["snap_tr"].get(i) if pd.notna(zs["snap_tr"].get(i)) else 0)
            lab = mp.get("snap_tr", {}).get("label", "edge")
            stick = mp.get("snap_tr", {}).get("sticks") or 0.49
            tail = (f"Worth about {e:+.1f} ppg against the model's number (held up out of sample, but small)." if lab == "edge"
                    else f"At {r['pos']} that hasn't beaten the model's number.")
            flags.append({"key": "snaps_up" if st_ > 0 else "snaps_down", "tone": "up" if st_ > 0 else "down", "metric": "snap_tr",
                          "short": f"Snaps {'up' if st_ > 0 else 'down'} {abs(100 * st_):.0f}",
                          "text": f"Snap share {100 * (r['snap'] + st_):.0f}% over his last three vs {100 * r['snap']:.0f}% for the season. "
                                  f"About {100 * stick:.0f}% of changes like this stick. {tail}", "edge": _r(e, 2) if lab == "edge" else None,
                          "label": lab})
        et = r.get("est_tr")
        if pd.notna(et) and abs(et) >= EST_T:
            e = (edge_v["est_tr"] or 0) * (zs["est_tr"].get(i) if pd.notna(zs["est_tr"].get(i)) else 0)
            lab = mp.get("est_tr", {}).get("label", "edge")
            tail = (f"Moves like this have kept going a little: about {e:+.1f} ppg more (held up out of sample, small)." if lab == "edge"
                    else f"At {r['pos']} moves like this haven't kept going.")
            flags.append({"key": "value_up" if et > 0 else "value_down", "tone": "up" if et > 0 else "down", "metric": "est_tr",
                          "short": f"Value {'up' if et > 0 else 'down'} {abs(et):.1f}",
                          "text": f"The model's rest-of-season number moved {et:+.1f} ppg since week {wk_prev}. {tail}",
                          "edge": _r(e, 2) if lab == "edge" else None, "label": lab})
        cvp = pct["cv"].get(i)
        if n >= 4 and pd.notna(cvp) and (cvp >= 85 or cvp <= 15):
            hi = cvp >= 85
            flags.append({"key": "volatile" if hi else "steady", "tone": "info", "metric": "cv",
                          "short": "Boom-bust" if hi else "Steady",
                          "text": f"Weekly scores swing {'more' if hi else 'less'} than {cvp if hi else 100 - cvp:.0f}% of fantasy-relevant "
                                  f"{r['pos']}s. That repeats moderately but doesn't predict points beyond the model.", "label": "trait"})
        rows.append({
            "id": i, "name": r["name"], "pos": r["pos"], "team": ab.get(int(r["pro_team_id"])) if pd.notna(r["pro_team_id"]) else None,
            "owner": owner.get(i), "games": n, "pool": bool(r["pool"]),
            "est_ppg": _r(r["est_ppg"], 1), "ecr_rank": _r(led["ecr_rank"].get(i), 1) if len(led) and "ecr_rank" in led else None,
            **{k: _r(r[k], 3) for k in ("ppg", "xfp", "fpoe", "opp", "snap", "tsh", "ays", "wopr", "csh", "cv", "spread", "sd",
                                        "p20", "p80", "smash", "start", "dud", "resid", "beat", "xfp_tr", "snap_tr", "tsh_tr",
                                        "opp_tr", "est_tr")},
            "pct": {k: _r(pct[k].get(i), 0) for k in PCT_KEYS},
            "duds": int(round((r["dud"] or 0) * n)), "smashes": int(round((r["smash"] or 0) * n)),
            "starts": int(round((r["start"] or 0) * n)),
            "weekly": wk, "flags": flags, "breakout": _breakout(r, Sp, i, done),
        })
    t = tiers(nt)
    # NFL schedule from the next unplayed week through 17 (nflverse abbreviations), for schedule-ahead views
    sc = inp.nfl_sched[(inp.nfl_sched.get("game_type", "REG") == "REG") & inp.nfl_sched["week"].between(done + 1, LAST)]
    sched: Dict[str, Dict[int, dict]] = {}
    for w, h, a_ in zip(sc["week"], sc["home_team"], sc["away_team"]):
        sched.setdefault(str(h), {})[int(w)] = {"opp": str(a_), "home": True}
        sched.setdefault(str(a_), {})[int(w)] = {"opp": str(h), "home": False}
    S_ = registry().get("schedule") or {}
    sched_ev = {"fpa_playoffs": ((S_.get("fpa") or {}).get("by_pos", {}).get("ALL") or {}).get("playoffs"),
                "pass_playoffs": ((S_.get("epa") or {}).get("by_side", {}).get("pass") or {}).get("playoffs"),
                "run_playoffs": ((S_.get("epa") or {}).get("by_side", {}).get("run") or {}).get("playoffs"),
                "pass_next4": ((S_.get("epa") or {}).get("by_side", {}).get("pass") or {}).get("next4"),
                "run_next4": ((S_.get("epa") or {}).get("by_side", {}).get("run") or {}).get("next4"),
                "label": (S_.get("epa") or {}).get("label", "weak")}
    return {"season": season, "through_week": done, "week": inp.week, "n_teams": nt, "tiers": t,
            "schedule": sched, "sched_weeks": list(range(done + 1, LAST + 1)), "sched_ev": sched_ev,
            "pool": {p: int(k * nt) for p, k in POOL.items()}, "est_weeks": [wk_prev, wk_now],
            "metrics": metrics(), "metrics_pos": M_pos, "players": rows,
            "edge_note": (registry().get("edge") or {}).get("single", {})}


# ------------------------------------------------------------------------------------------- league
def league(season: int = 2026) -> dict:
    key = ("league", season, _inputs_key(season))
    if _CACHE.get("lkey") == key:
        return _CACHE["league"]
    out = _league(season)
    _CACHE.update(lkey=key, league=out)
    return out


def _league(season: int) -> dict:
    """Team-by-week table: points, best possible lineup, ESPN's projection of the starters used, and
    where each team's starting points came from (its own draft picks vs players it added)."""
    from . import data as d
    from .views import best_lineup, _opponent
    inp = d.load(season)
    done = complete_week(season, inp.week - 1)
    pw = inp.players_weekly[inp.players_weekly["kind"] == "actual"][["espn_id", "week", "points"]]
    pts_of = {(int(i), int(w)): float(p) for i, w, p in zip(pw["espn_id"], pw["week"], pw["points"])}
    pj = inp.espn_proj.drop_duplicates(["week", "espn_id"])
    proj_of = {(int(i), int(w)): float(p) for i, w, p in zip(pj["espn_id"], pj["week"], pj["espn_proj"])}
    tw = inp.team_weekly
    draft = pd.read_csv(ACT / str(season) / "draft.csv")
    drafted_by = dict(zip(draft["espn_id"].astype(int), draft["team_id"].astype(int)))
    teams = sorted(int(t) for t in tw["team_id"].unique())
    cells = []
    for w in range(1, done + 1):
        R = inp.rosters[inp.rosters["week"] == w]
        for t in teams:
            r = R[R["team_id"] == t]
            if r.empty:
                continue
            rows = [{"id": int(i), "pos": p, "pts": pts_of.get((int(i), w), 0.0), "proj": proj_of.get((int(i), w), 0.0)}
                    for i, p, s in zip(r["espn_id"], r["position"], r["lineup_slot"]) if s != "IR"]
            best = best_lineup(rows, key="pts")
            opt = sum(next(z["pts"] for z in rows if z["id"] == i) for i in best.values())
            # the lineup ESPN's projections would have started, scored with what actually happened
            espn = sum(next(z["pts"] for z in rows if z["id"] == i) for i in best_lineup(rows, key="proj").values())
            st = r[~r["lineup_slot"].isin(["BE", "IR"])]
            got = sum(pts_of.get((int(i), w), 0.0) for i in st["espn_id"])
            proj = sum(proj_of.get((int(i), w), 0.0) for i in st["espn_id"])
            own = sum(pts_of.get((int(i), w), 0.0) for i in st["espn_id"] if drafted_by.get(int(i)) == t)
            missed = sorted(((z["pts"], z["id"]) for z in rows if z["id"] in best.values() and z["id"] not in set(st["espn_id"].astype(int))),
                            reverse=True)
            cells.append({"team_id": t, "week": w, "pts": round(got, 2), "opt": round(opt, 2), "proj": round(proj, 2),
                          "espn": round(espn, 2), "opp": _opponent(inp, t, w),
                          "drafted": round(own, 2), "missed": [int(i) for _, i in missed[:2]]})
    C = pd.DataFrame(cells)
    if C.empty:
        return {"season": season, "through_week": done, "teams": [], "cells": [], "draft": []}
    g = C.groupby("week")["pts"]
    C["z"] = (C["pts"] - g.transform("mean")) / g.transform("std")
    C["rank"] = g.rank(ascending=False, method="min").astype(int)
    C["eff"] = C["pts"] / C["opt"].where(C["opt"] > 0)
    C["left"] = C["opt"] - C["pts"]
    C["luck"] = C["pts"] - C["proj"]
    C["vs_espn"] = C["pts"] - C["espn"]
    # W/L the ESPN-best lineup would have flipped (opponent from the league schedule; unknown -> not counted)
    op = C.set_index(["team_id", "week"])["pts"]
    o = pd.Series([op.get((x, w)) if pd.notna(x) else None for x, w in zip(C["opp"], C["week"])], index=C.index, dtype=float)
    C["flip_won"] = ((C["pts"] < o) & (C["espn"] > o)).astype(int)
    C["flip_lost"] = ((C["pts"] > o) & (C["espn"] < o)).astype(int)
    T = C.groupby("team_id").agg(pts=("pts", "sum"), opt=("opt", "sum"), proj=("proj", "sum"), left=("left", "sum"),
                                 espn=("espn", "sum"), vs_espn=("vs_espn", "sum"), flip_won=("flip_won", "sum"),
                                 flip_lost=("flip_lost", "sum"),
                                 luck=("luck", "sum"), drafted=("drafted", "sum"), sd=("pts", "std"), best=("pts", "max"),
                                 worst=("pts", "min"), top3=("rank", lambda x: int((x <= 3).sum())), eff=("eff", "mean"))
    T["opt_rank"] = T["opt"].rank(ascending=False, method="min").astype(int)
    T["pts_rank"] = T["pts"].rank(ascending=False, method="min").astype(int)
    names = {t: inp.team_name(t) for t in teams}
    # draft: positional draft order vs positional rank by points so far
    tot = pw[pw["week"] <= done].groupby("espn_id")["points"].sum()
    pos_of = inp.players_weekly.drop_duplicates("espn_id").set_index("espn_id")["position"]
    name_of = inp.players_weekly.drop_duplicates("espn_id").set_index("espn_id")["name"]
    dr = draft.copy()
    dr["overall"] = (dr["round"] - 1) * len(teams) + dr["pick"]
    dr["pos"] = dr["espn_id"].map(pos_of)
    dr = dr.dropna(subset=["pos"]).sort_values("overall")
    dr["pos_pick"] = dr.groupby("pos").cumcount() + 1
    ranks = {}
    for pos, gp in pd.DataFrame({"pts": tot, "pos": pos_of.reindex(tot.index)}).groupby("pos"):
        ranks.update(gp["pts"].rank(ascending=False, method="min").astype(int).to_dict())
    picks = [{"id": int(i), "name": name_of.get(i), "pos": p, "team_id": int(t), "round": int(rd), "overall": int(ov),
              "pos_pick": int(pp), "pos_rank_now": int(ranks[i]) if i in ranks else None, "pts": _r(tot.get(i, 0.0), 1),
              "keeper": bool(k)}
             for i, p, t, rd, ov, pp, k in zip(dr["espn_id"], dr["pos"], dr["team_id"], dr["round"], dr["overall"],
                                                 dr["pos_pick"], dr["keeper"])]
    reg = registry().get("teams", {})
    return {
        "season": season, "through_week": done, "n_teams": len(teams),
        "teams": [{"id": int(t), "name": names[int(t)], **{k: _r(v, 2) for k, v in r.items()}} for t, r in T.iterrows()],
        "cells": [{k: (_r(v, 3) if isinstance(v, float) else v) for k, v in c.items()} for c in C.to_dict("records")],
        "draft": picks,
        "labels": {k: {"name": v["name"], "label": v["label"], "repeat": (v["all"].get("repeat") or {}).get("v")} for k, v in reg.items()},
    }
