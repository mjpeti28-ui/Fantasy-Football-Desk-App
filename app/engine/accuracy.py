"""In-season accuracy: the numbers Desk shows before kickoff, checked against what happened this season.

Every check is rebuilt from the archives (ESPN's weekly projections, rosters by week, actual points),
so past weeks are scored with the same formulas the screens use today:
  range  : the 80% outcome band (ESPN projection x the position's actual/forecast spread, ratio_q)
  boom   : P(15+) and P(20+) from the same spread, predicted vs actual rate
  h2h    : P(A outscores B) for same-team, same-position pairs (real start/sit calls), by confidence
  back   : first game back from missed games vs ESPN (the one blind spot the screens flag)
  lineup : your lineup vs the one ESPN's projections would have started, and the best possible
  winp   : pre-game win probability for every league matchup vs the result
  sources: projection error by source (history/projection_scores, written by the weekly run)
Scored on relevant players only (rostered, not on IR, projected), the way the backtests were.
Each check carries n, a 95% range and what 2021-25 says is normal, so a drift shows up as
"outside its range", not as a reason to retune on a few weeks.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import analytics, data as d, views

POS = ("QB", "RB", "WR", "TE")
BACK_EXPECT = {"WR": -1.34, "QB": -1.49, "RB": -0.66, "TE": 0.0}  # pff/creative.md E10, 2021-25 vs ESPN
_C: dict = {"key": None, "v": None}


def wilson(k: float, n: int, z: float = 1.96) -> list:
    if n <= 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(float(c - h), 3), round(float(c + h), 3)]


def _mean_ci(x) -> dict:
    x = np.asarray(x, float)
    n = len(x)
    if not n:
        return {"n": 0, "mean": None, "ci": [None, None]}
    m = float(x.mean())
    se = float(x.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    return {"n": n, "mean": round(m, 2), "ci": [round(m - 1.96 * se, 2), round(m + 1.96 * se, 2)] if n > 1 else [None, None]}


def _rows(inp: d.Inputs, done: int) -> pd.DataFrame:
    """One row per rostered player-week with a projection and a game: projection, band, actual."""
    pw = inp.players_weekly[inp.players_weekly["kind"] == "actual"]
    act = pw.drop_duplicates(["espn_id", "week"]).set_index(["espn_id", "week"])
    pj = inp.espn_proj.drop_duplicates(["week", "espn_id"]).set_index(["espn_id", "week"])["espn_proj"]
    R = inp.rosters[(inp.rosters["week"] <= done) & (inp.rosters["lineup_slot"] != "IR") & inp.rosters["position"].isin(POS)]
    out = []
    for r in R.itertuples():
        k = (int(r.espn_id), int(r.week))
        proj = pj.get(k)
        if proj is None or not proj > 0 or k not in act.index:
            continue
        a = act.loc[k]
        if not bool(a.get("played", True)):
            continue
        out.append({"id": k[0], "week": k[1], "team_id": int(r.team_id), "pos": r.position, "slot": r.lineup_slot,
                    "proj": float(proj), "pts": float(a["points"])})
    return pd.DataFrame(out)


def _range_boom(D: pd.DataFrame, rq: dict) -> tuple[dict, dict]:
    qs = np.linspace(0.01, 0.99, len(next(iter(rq.values()))))
    lo_i, hi_i = int(np.argmin(abs(qs - 0.10))), int(np.argmin(abs(qs - 0.90)))
    inside, p15, p20 = [], [], []
    for r in D.itertuples():
        g = np.asarray(rq[r.pos]) * r.proj
        inside.append(g[lo_i] <= r.pts <= g[hi_i])
        p15.append(float((g >= 15).mean()))
        p20.append(float((g >= 20).mean()))
    D = D.assign(inside=inside, p15=p15, p20=p20)
    rng = {"n": int(len(D)), "rate": round(float(D["inside"].mean()), 3) if len(D) else None, "ci": wilson(D["inside"].sum(), len(D)), "expect": 0.80,
           "by_pos": {p: {"n": int(len(g)), "rate": round(float(g["inside"].mean()), 3), "ci": wilson(g["inside"].sum(), len(g))} for p, g in D.groupby("pos")}}
    boom = {}
    for t, col in ((15, "p15"), (20, "p20")):
        hit = (D["pts"] >= t)
        bins = pd.cut(D[col], [0, 0.1, 0.25, 0.4, 0.6, 1.0], include_lowest=True)
        boom[str(t)] = {"n": int(len(D)), "pred": round(float(D[col].mean()), 3), "actual": round(float(hit.mean()), 3), "ci": wilson(hit.sum(), len(D)),
                        "bins": [{"lo": float(b.left), "hi": float(b.right), "n": int(len(g)), "pred": round(float(g[col].mean()), 3),
                                  "actual": round(float((g["pts"] >= t).mean()), 3), "ci": wilson((g["pts"] >= t).sum(), len(g))}
                                 for b, g in D.groupby(bins, observed=True) if len(g)]}
    return rng, boom


def _p_beats(a: float, pa: str, b: float, pb: str, rq: dict) -> float:
    x = np.asarray(rq[pa]) * a
    y = np.asarray(rq[pb]) * b
    return float((x[:, None] > y[None, :]).mean() + 0.5 * (x[:, None] == y[None, :]).mean())


def _h2h(D: pd.DataFrame, rq: dict) -> dict:
    """Same team, same week, same position: the pairs a start/sit call is actually made between."""
    ps = []
    for (_t, _w, pos), g in D.groupby(["team_id", "week", "pos"]):
        g = g.reset_index(drop=True)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                a, b = g.loc[i], g.loc[j]
                p = _p_beats(a.proj, pos, b.proj, pos, rq)
                fav_p, fav_won = (p, a.pts > b.pts) if p >= 0.5 else (1 - p, b.pts > a.pts)
                if a.pts != b.pts:
                    ps.append((fav_p, bool(fav_won)))
    P = pd.DataFrame(ps, columns=["p", "won"])
    if P.empty:
        return {"n": 0, "bins": []}
    bins = pd.cut(P["p"], [0.5, 0.55, 0.65, 0.8, 1.0], include_lowest=True)
    return {"n": int(len(P)), "pred": round(float(P["p"].mean()), 3), "actual": round(float(P["won"].mean()), 3), "ci": wilson(P["won"].sum(), len(P)),
            "bins": [{"lo": float(b.left), "hi": float(b.right), "n": int(len(g)), "pred": round(float(g["p"].mean()), 3),
                      "actual": round(float(g["won"].mean()), 3), "ci": wilson(g["won"].sum(), len(g))} for b, g in P.groupby(bins, observed=True) if len(g)]}


def _back(inp: d.Inputs, done: int) -> dict:
    """First game back after missing his team's previous game(s), every player projected 5+ (the backtest's set)."""
    pw = inp.players_weekly[inp.players_weekly["kind"] == "actual"].drop_duplicates(["espn_id", "week"])
    pj = inp.espn_proj.drop_duplicates(["week", "espn_id"]).set_index(["espn_id", "week"])["espn_proj"]
    bye = dict(zip(inp.pro_teams["pro_team_id"], inp.pro_teams["bye_week"])) if "pro_team_id" in inp.pro_teams else {}
    played = {(int(i), int(w)) for i, w, p in zip(pw["espn_id"], pw["week"], pw["played"]) if bool(p)}
    rows = []
    for r in pw[pw["played"].astype(bool) & pw["position"].isin(POS) & (pw["week"] <= done)].itertuples():
        w, i = int(r.week), int(r.espn_id)
        b = bye.get(r.pro_team_id)
        prev = [x for x in range(w - 1, 0, -1) if x != b]
        if not prev or (i, prev[0]) in played:
            continue
        if not any((i, x) in played for x in prev):  # never played before: a debut, not a return
            continue
        proj = pj.get((i, w))
        if proj is None or proj < 5:
            continue
        rows.append({"pos": r.position, "miss": float(r.points) - float(proj)})
    B = pd.DataFrame(rows, columns=["pos", "miss"])
    return {p: {**_mean_ci(B[B["pos"] == p]["miss"]), "expect": BACK_EXPECT[p]} for p in POS}


def _lineups(season: int, team: int) -> dict:
    lg = analytics.league(season)
    C = pd.DataFrame(lg.get("cells") or [])
    if C.empty:
        return {"weeks": []}
    me = C[C["team_id"] == team].sort_values("week")
    weeks = [{"week": int(r.week), "pts": r.pts, "espn": r.espn, "opt": r.opt} for r in me.itertuples()]
    league_gap = (C["pts"] - C["espn"])
    return {"weeks": weeks, "vs_espn": _mean_ci(me["pts"] - me["espn"]), "league_vs_espn": _mean_ci(league_gap),
            "left": round(float((me["opt"] - me["pts"]).sum()), 1), "expect_vs_espn": -1.16}


def _winp(inp: d.Inputs, done: int, rq: dict) -> dict:
    from .live import simulate
    from .views import _opponent
    pw = inp.players_weekly[inp.players_weekly["kind"] == "actual"].drop_duplicates(["espn_id", "week"])
    pts_of = {(int(i), int(w)): float(p) for i, w, p in zip(pw["espn_id"], pw["week"], pw["points"])}
    pj = inp.espn_proj.drop_duplicates(["week", "espn_id"]).set_index(["espn_id", "week"])["espn_proj"]
    games, seen = [], set()
    for w in range(1, done + 1):
        R = inp.rosters[(inp.rosters["week"] == w) & ~inp.rosters["lineup_slot"].isin(["BE", "IR"])]
        for t in sorted(R["team_id"].unique()):
            o = _opponent(inp, int(t), w)
            if o is None or (w, min(t, o), max(t, o)) in seen:
                continue
            seen.add((w, min(t, o), max(t, o)))
            side = lambda tid: [{"pos": r.position if r.position in rq else "WR", "proj": float(pj.get((int(r.espn_id), w), 0) or 0), "state": "pre", "pts": 0.0}
                                for r in R[R["team_id"] == tid].itertuples()]
            a, b = side(int(t)), side(int(o))
            pa = sum(pts_of.get((int(i), w), 0.0) for i in R[R["team_id"] == t]["espn_id"])
            pb = sum(pts_of.get((int(i), w), 0.0) for i in R[R["team_id"] == o]["espn_id"])
            if pa == pb:
                continue
            p = simulate(a, b, rq, seed=w * 100 + int(t), n=4000)["p_win"]
            fav_p, fav_won = (p, pa > pb) if p >= 0.5 else (1 - p, pb > pa)
            games.append({"week": w, "p": fav_p, "won": bool(fav_won)})
    G = pd.DataFrame(games, columns=["week", "p", "won"])
    if G.empty:
        return {"n": 0}
    brier = float(((G["p"] - G["won"].astype(float)) ** 2).mean())
    return {"n": int(len(G)), "pred": round(float(G["p"].mean()), 3), "actual": round(float(G["won"].mean()), 3), "ci": wilson(G["won"].sum(), len(G)),
            "brier": round(brier, 3), "brier_coin": 0.25}


def _sources(season: int) -> list:
    p = d.ROOT / "history" / "projection_scores" / f"{season}_summary.csv"
    if not p.exists():
        return []
    s = pd.read_csv(p)
    s = s[s["position"] == "ALL"]
    return [{"source": r.source, "weeks": int(r.weeks), "n": int(r.n), "mae": float(r.mae), "bias": float(r.bias), "rank_corr": float(r.rank_corr)} for r in s.itertuples()]


def build(season: int = 2026, team: int = 3) -> dict:
    inp = d.load(season)
    done = analytics.complete_week(season, inp.week - 1)
    key = (season, team, done, analytics._inputs_key(season))
    if _C["key"] == key:
        return _C["v"]
    rq = {k: list(v) for k, v in views.ratio_quantiles(inp).items()}
    D = _rows(inp, done)
    out = {"season": season, "through_week": done, "n_players": int(len(D))}
    if len(D):
        out["range"], out["boom"] = _range_boom(D, rq)
        out["h2h"] = _h2h(D, rq)
        out["proj_err"] = {"mae": round(float((D["pts"] - D["proj"]).abs().mean()), 2), "bias": round(float((D["pts"] - D["proj"]).mean()), 2), "n": int(len(D))}
    out["back"] = _back(inp, done)
    out["lineup"] = _lineups(season, team)
    out["winp"] = _winp(inp, done, rq)
    out["sources"] = _sources(season)
    _C.update(key=key, v=out)
    return out


if __name__ == "__main__":
    import json, sys
    print(json.dumps(build(int(sys.argv[1]) if len(sys.argv) > 1 else 2026), indent=1, default=float)[:6000])
