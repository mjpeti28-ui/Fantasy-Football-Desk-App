"""Player roles: who a player is (style + volume tier), what stands out (traits), and what changed
(trends). Descriptive, built from nflverse usage; never fed into valuation.

Model v2 (fit offline by roles_fit.py on 2016-25, shipped as roles_model.json):
- Every usage metric carries a stabilization constant M (games at which half of the observed
  spread is real, from split-half reliability). In-season numbers are shrunk toward a prior
  (last season, regressed by its year-to-year correlation, else the position average) by
  R(n) = n / (n + M). Percentiles and traits use the shrunk value.
- Style is a 3-group Gaussian mixture per position on reliable shape features. Classification
  inflates each group's covariance by the sampling noise of an n-game sample, so a 3-game label
  says "70%" when it is right about 70% of the time (backtested).
Scoring here is numpy only.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
COUNTS = ["tgt", "ay", "ay_sum", "deep", "behind", "screen", "charted", "contested", "rz_tgt", "ez_tgt", "ld_tgt", "tm_tgt",
          "rec", "yac", "car", "scr", "rz_car", "gl_car", "ryd", "db", "att", "sacks", "p_ay", "p_deep", "offense_snaps",
          "t_tgt", "t_ay", "t_rz_tgt", "t_ld_tgt", "t_2m_tgt", "t_car", "t_rz_car", "t_gl_car", "t_db", "t_plays"]
BOX = ["cmp", "pass_att", "pass_yd", "pass_td", "ints", "rush_yd", "rush_td", "recs", "rec_yd", "rec_td", "fl", "ppr"]

STYLE = {  # features and clip ranges; chosen for split-half reliability (see roles_fit.py)
    "RB": {"rec_mix": (0, 1), "pd_ratio": (0, 3)},
    "WR": {"adot": (-5, 30), "behind_r": (0, 1), "deep_r": (0, 1)},
    "TE": {"tgt_per_snap": (0, .4), "adot": (-5, 25), "behind_r": (0, 1)},
    "QB": {"des_rush_g": (0, 15), "scr_rate": (0, .3), "p_adot": (0, 15)},
}
AXES = {  # two interpretable style axes per position for the role map
    "RB": ("rec_mix", "pd_ratio"), "WR": ("adot", "behind_r"), "TE": ("tgt_per_snap", "adot"), "QB": ("des_rush_g", "p_adot"),
}
# fingerprint metrics: key -> (label, format, group)
P, N1, N2 = "pct", "num1", "num2"
METRICS = {
    "RB": {"snap": ("Snap share", "pct100", "Role"), "opp_sh": ("Opportunity share", P, "Role"), "car_sh": ("Carry share", P, "Role"),
           "tgt_sh": ("Target share", P, "Role"), "rec_mix": ("Touches that are catches", P, "Style"),
           "pd_sh": ("3rd-down & 2-minute target share", P, "Style"), "rz_car_sh": ("Red-zone carry share", P, "Scoring"),
           "gl_car_sh": ("Goal-line carry share", P, "Scoring"), "td_g": ("TDs per game", N2, "Scoring"),
           "ypc": ("Yards per carry", N1, "Efficiency"), "ppr_g": ("PPR points per game", N1, "Scoring")},
    "WR": {"snap": ("Snap share", "pct100", "Role"), "tgt_sh": ("Target share", P, "Role"), "ay_sh": ("Air-yards share", P, "Role"),
           "ld_tgt_sh": ("3rd/4th-down target share", P, "Role"), "adot": ("Average depth of target", N1, "Style"),
           "deep_r": ("Targets 20+ yards downfield", P, "Style"), "behind_r": ("Targets at or behind the line", P, "Style"),
           "rz_tgt_sh": ("Red-zone target share", P, "Scoring"), "td_g": ("TDs per game", N2, "Scoring"),
           "yac_rec": ("Yards after catch per catch", N1, "Efficiency"), "catch_r": ("Catch rate", P, "Efficiency"),
           "ypt": ("Yards per target", N1, "Efficiency"), "cont_r": ("Contested targets", P, "Efficiency"),
           "ppr_g": ("PPR points per game", N1, "Scoring")},
    "TE": {"snap": ("Snap share", "pct100", "Role"), "tgt_sh": ("Target share", P, "Role"), "tgt_per_snap": ("Targets per snap", N2, "Role"),
           "ld_tgt_sh": ("3rd/4th-down target share", P, "Role"), "adot": ("Average depth of target", N1, "Style"),
           "behind_r": ("Targets at or behind the line", P, "Style"), "rz_tgt_sh": ("Red-zone target share", P, "Scoring"),
           "td_g": ("TDs per game", N2, "Scoring"), "yac_rec": ("Yards after catch per catch", N1, "Efficiency"),
           "catch_r": ("Catch rate", P, "Efficiency"), "ypt": ("Yards per target", N1, "Efficiency"), "ppr_g": ("PPR points per game", N1, "Scoring")},
    "QB": {"des_rush_g": ("Designed runs per game", N1, "Style"), "scr_rate": ("Scramble rate", P, "Style"),
           "p_adot": ("Average throw depth", N1, "Style"), "p_deep": ("Throws 20+ yards", P, "Style"),
           "sack_r": ("Sack rate", P, "Efficiency"), "ypa": ("Yards per attempt", N1, "Efficiency"), "ppr_g": ("PPR points per game", N1, "Scoring")},
}
LOWER_IS_BETTER = {"sack_r"}


def _r(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def rates(g: pd.DataFrame) -> pd.DataFrame:
    """Derived rates from summed counts. Needs COUNTS (+ BOX if present), games and snap (mean %)."""
    g = g.copy()
    g["car_sh"] = _r(g.car, g.t_car); g["tgt_sh"] = _r(g.tgt, g.t_tgt); g["ay_sh"] = _r(g.ay, g.t_ay)
    g["opp_sh"] = _r(g.car + g.tgt, g.t_car + g.t_tgt)
    g["rz_car_sh"] = _r(g.rz_car, g.t_rz_car); g["gl_car_sh"] = _r(g.gl_car, g.t_gl_car)
    g["rz_tgt_sh"] = _r(g.rz_tgt, g.t_rz_tgt); g["ld_tgt_sh"] = _r(g.ld_tgt, g.t_ld_tgt)
    g["pd_sh"] = _r(g.ld_tgt + g.tm_tgt, g.t_ld_tgt + g.t_2m_tgt)
    g["rz_sh"] = _r(g.rz_car + g.rz_tgt, g.t_rz_car + g.t_rz_tgt)
    g["adot"] = _r(g.ay_sum, g.tgt); g["deep_r"] = _r(g.deep, g.tgt); g["behind_r"] = _r(g.behind, g.tgt)
    g["scr_r"] = _r(g.screen, g.charted); g["cont_r"] = _r(g.contested, g.charted)
    g["yac_rec"] = _r(g.yac, g.rec); g["ez_r"] = _r(g.ez_tgt, g.tgt)
    g["rec_mix"] = _r(g.tgt, g.tgt + g.car); g["tgt_per_snap"] = _r(g.tgt, g.offense_snaps)
    g["pd_ratio"] = _r(g.pd_sh, g.tgt_sh)
    g["des_rush_g"] = _r(g.car, g.games); g["scr_rate"] = _r(g.scr, g.db); g["sack_r"] = _r(g.sacks, g.db)
    g["p_adot"] = _r(g.p_ay, g.att); g["p_deep"] = _r(g.p_deep, g.att); g["tgt_g"] = _r(g.tgt, g.games)
    g["car_g"] = _r(g.car, g.games); g["db_g"] = _r(g.db, g.games); g["db_sh"] = _r(g.db, g.t_db)
    if "ppr" in g:
        g["ppr_g"] = _r(g.ppr, g.games); g["td_g"] = _r(g.rec_td + g.rush_td, g.games)
        g["ypc"] = _r(g.rush_yd, g.car); g["ypt"] = _r(g.rec_yd, g.tgt); g["catch_r"] = _r(g.recs, g.tgt)
        g["ypa"] = _r(g.pass_yd, g.pass_att)
    return g


def aggregate(w: pd.DataFrame, by=("season", "gsis", "pos")) -> pd.DataFrame:
    by = list(by)
    cols = COUNTS + [c for c in BOX if c in w.columns]
    g = w.groupby(by)[cols].sum().reset_index()
    g["games"] = w.groupby(by).size().values
    g["snap"] = w.groupby(by)["snap"].mean().values
    return rates(g)


@lru_cache(maxsize=1)
def load_model() -> dict:
    return json.loads((HERE / "roles_model.json").read_text())


def reliability(model: dict, pos: str, metric: str, games: float) -> float | None:
    s = model["stab"].get(pos, {}).get(metric)
    if not s or games is None or games <= 0:
        return None
    return float(games / (games + s["M"]))


def shrink(model: dict, pos: str, metric: str, raw, games: float, prev=None, prev_games: float = 0) -> float | None:
    """Regress an n-game value toward last season (if any) or the position average."""
    s = model["stab"].get(pos, {}).get(metric)
    if not s or raw is None or (isinstance(raw, float) and np.isnan(raw)):
        if s and prev is not None and not np.isnan(prev) and prev_games >= 6:
            return float(s["mean"] + s["yoy"] * (prev - s["mean"]))
        return None
    prior = s["mean"]
    if prev is not None and not np.isnan(prev) and prev_games >= 6:
        prior = s["mean"] + max(0.0, s["yoy"]) * (prev - s["mean"])
    R = games / (games + s["M"])
    return float(prior + R * (raw - prior))


def pct_of(model: dict, pos: str, metric: str, v) -> float | None:
    s = model["stab"].get(pos, {}).get(metric)
    if s is None or v is None or np.isnan(v):
        return None
    return float(np.interp(v, s["q"], np.linspace(0, 100, len(s["q"]))))


def classify(model: dict, pos: str, row, games: float) -> np.ndarray | None:
    """Style probabilities with each group's covariance widened by n-game sampling noise."""
    m = model["style"].get(pos)
    if not m:
        return None
    f = m["features"]
    x = np.array([np.clip(row.get(c, np.nan), *STYLE[pos][c]) for c in f], float)
    if np.isnan(x).any():
        return None
    z = (x - np.array(m["mean"])) / np.array(m["scale"])
    M = np.array([model["stab"][pos][c]["M"] for c in f])
    rseas = 15.0 / (15.0 + M)
    extra = np.clip((1.0 / max(games, 1.0) - 1.0 / 15.0) * rseas * M, 0, None)
    lp = []
    for w, mu, cov in zip(m["weights"], m["means"], m["covs"]):
        c = np.array(cov) + np.diag(extra); d = z - np.array(mu)
        _, logdet = np.linalg.slogdet(c)
        lp.append(np.log(w) - 0.5 * (logdet + d @ np.linalg.solve(c, d)))
    lp = np.array(lp); p = np.exp(lp - lp.max())
    return p / p.sum()


def tier(pos: str, r) -> str:
    g = lambda k: float(r.get(k) if r.get(k) is not None and not np.isnan(r.get(k)) else 0)
    if pos == "RB":
        c = g("car_sh")
        if c >= .60:
            return "Workhorse" if g("snap") >= 70 else "Lead back"
        return "Split backfield" if c >= .40 else "Change of pace" if c >= .20 else "Depth"
    if pos == "WR":
        t = g("tgt_sh")
        return "Alpha target share" if t >= .25 else "Starter" if t >= .17 else "Full-time, low volume" if g("snap") >= 70 else "Rotational"
    if pos == "TE":
        return "Featured" if g("tgt_sh") >= .17 else "Starter" if g("snap") >= 60 else "Rotational"
    return "Starter" if g("db_sh") >= .8 else "Part-time"


TRAIT_CUT = 85
TRAITS = {  # metric -> (hi label, lo label or None, minimum reliability)
    "RB": {"snap": ("Every-down snaps", None), "gl_car_sh": ("Goal-line back", None), "rz_car_sh": ("Red-zone carries", None),
           "pd_sh": ("Third-down & 2-minute back", None), "tgt_sh": ("Real receiving role", None), "rec_mix": ("Receiver out of the backfield", "Pure runner")},
    "WR": {"tgt_sh": ("Target hog", None), "ay_sh": ("Air-yards leader", None), "adot": ("Deep aDOT", "Short-area"),
           "behind_r": ("Screen game", None), "rz_tgt_sh": ("Red-zone target", None), "ld_tgt_sh": ("Third-down go-to", None),
           "yac_rec": ("YAC after the catch", None), "cont_r": ("Contested catches", None), "snap": (None, "Part-time snaps")},
    "TE": {"snap": ("Every-down TE", None), "tgt_per_snap": ("Targeted when he's out there", "Mostly blocks"),
           "rz_tgt_sh": ("Red-zone target", None), "adot": ("Seam threat", None), "ld_tgt_sh": ("Third-down go-to", None)},
    "QB": {"des_rush_g": ("Designed runs", None), "scr_rate": ("Scrambles", None), "p_deep": ("Throws deep", None),
           "p_adot": (None, "Checks down"), "sack_r": ("Takes sacks", None)},
}
MIN_R = 0.45  # don't call a trait on a number that's mostly noise yet


def fingerprint(model: dict, pos: str, cur, games: float, prev=None) -> list[dict]:
    """Every fingerprint metric: raw, shrunk estimate, percentile, reliability."""
    out = []
    for k, (label, fmt, group) in METRICS.get(pos, {}).items():
        raw = None if cur is None else cur.get(k)
        raw = None if raw is None or (isinstance(raw, float) and np.isnan(raw)) else float(raw)
        pv = None if prev is None else prev.get(k); pg = 0 if prev is None else float(prev.get("games", 0))
        pv = None if pv is None or (isinstance(pv, float) and np.isnan(pv)) else float(pv)
        est = shrink(model, pos, k, raw, games, pv, pg)
        if est is None:
            continue
        out.append({"key": k, "label": label, "fmt": fmt, "group": group, "raw": raw, "est": est, "prev": pv,
                    "pct": pct_of(model, pos, k, est), "raw_pct": pct_of(model, pos, k, raw) if raw is not None else None, "rel": reliability(model, pos, k, games) if raw is not None else 0.0,
                    "stab_games": model["stab"][pos][k]["M"], "lower_better": k in LOWER_IS_BETTER})
    return out


def traits_from(pos: str, fp: list[dict]) -> list[dict]:
    out = []
    spec = TRAITS.get(pos, {})
    for m in fp:
        if m["key"] not in spec or m["pct"] is None or (m["rel"] or 0) < MIN_R:
            continue
        hi, lo = spec[m["key"]]
        if hi and m["pct"] >= TRAIT_CUT:
            out.append({"label": hi, "key": m["key"], "pct": round(m["pct"]), "side": "hi", "rel": round(m["rel"], 2)})
        elif lo and m["pct"] <= 100 - TRAIT_CUT:
            out.append({"label": lo, "key": m["key"], "pct": round(m["pct"]), "side": "lo", "rel": round(m["rel"], 2)})
    return out


def fmt_val(fmt: str, v) -> str:
    if v is None:
        return "–"
    return f"{v:.0f}%" if fmt == "pct100" else f"{v * 100:.0f}%" if fmt == "pct" else f"{v:.1f}" if fmt == "num1" else f"{v:.2f}"


# ---------- trends ----------
SNAP_T, SHARE_T = 15.0, 5.0


def _share(pos: str, d: pd.DataFrame) -> np.ndarray:
    return (_r(d.tgt + d.car, d.t_tgt + d.t_car) if pos == "RB" else _r(d.tgt, d.t_tgt)) * 100


def trends(model, pos, d, team, tw, last_week, snaps, mates, names, g, c, p):
    """Usage trends for one player. d: his weekly rows this season (sorted)."""
    out = []
    if pos == "QB" or team is None:
        return out
    cal, pts, shift = model["trend"][pos], model.get("trend_pts", {}).get(pos, {}), model["season_shift"].get(pos, {})
    wk = d["week"].astype(int).tolist(); sn = d["snap"].astype(float).values; sh = _share(pos, d)
    lab = "opportunity share" if pos == "RB" else "target share"

    def add(kind, label, text, key, extra=None):
        h = cal.get(key, {}); e = pts.get(key, {})
        t = {"kind": kind, "label": label, "text": text, "hold": h.get("hold"), "hold_n": h.get("n"),
             "next_vs_espn": e.get("rel"), "next_lo": e.get("lo"), "next_hi": e.get("hi"), "next_n": e.get("n")}
        if extra:
            t.update(extra)
        out.append(t)

    team_weeks = [x for x in tw if x <= last_week]
    missed = [x for x in team_weeks if x not in wk]
    if missed and team_weeks and missed[-1] == team_weeks[-1]:
        out.append({"kind": "absent", "label": "Didn't play last game", "text": f"No snaps in week {missed[-1]}.", "hold": None})
    if len(sn) >= 4:
        base, rec, bsh, rsh = sn[:-2].mean(), sn[-2:].mean(), np.nanmean(sh[:-2]), np.nanmean(sh[-2:])
        src = f"weeks {wk[0]}–{wk[-3]}"
        ds, dh = rec - base, rsh - bsh
        if ds >= SNAP_T:
            out_m = []
            for m in mates - {g}:
                bw = [x for x in wk[:-2]]
                b_ = np.mean([snaps.get((team, pos, x, m), 0) for x in bw]) if bw else 0
                if b_ >= 30 and all(snaps.get((team, pos, x, m), 0) == 0 for x in wk[-2:]):
                    out_m.append(names.get(m, "a teammate"))
            if out_m:
                add("snap_up", "Snaps up", f"{base:.0f}% → {rec:.0f}% of snaps the last two games (vs {src}) with {', '.join(out_m)} out.",
                    "snap_up_out", {"cause": out_m, "hold_if_still_out": cal.get("snap_up_still_out", {}).get("hold")})
            else:
                add("snap_up", "Snaps up", f"{base:.0f}% → {rec:.0f}% of snaps the last two games (vs {src}).", "snap_up")
        elif ds <= -SNAP_T:
            add("snap_down", "Snaps down", f"{base:.0f}% → {rec:.0f}% of snaps the last two games (vs {src}).", "snap_down")
        if not np.isnan(dh):
            if dh >= SHARE_T and ds < 10:
                add("share_spike", f"{lab.capitalize()} spike", f"{bsh:.0f}% → {rsh:.0f}% {lab} without more snaps.", "share_spike")
            elif dh >= SHARE_T:
                add("share_up", f"{lab.capitalize()} up", f"{bsh:.0f}% → {rsh:.0f}% {lab}, with more snaps.", "share_up_with_snaps")
            elif dh <= -SHARE_T and ds > -10:
                add("share_down", f"{lab.capitalize()} down", f"{bsh:.0f}% → {rsh:.0f}% {lab} on the same snaps.", "share_drop")
    if p is not None and c is not None and len(sn) >= 3 and p["games"] >= 6:
        key = "opp_sh" if pos == "RB" else "tgt_sh"
        d1 = float(c["snap"]) - float(p["snap"]); d2 = (float(c[key]) - float(p[key])) * 100
        txt = f"{p['snap']:.0f}% → {c['snap']:.0f}% of snaps; {lab} {float(p[key]) * 100:.0f}% → {float(c[key]) * 100:.0f}%."
        if d1 >= SNAP_T or d2 >= SHARE_T:
            k = "snap_up" if d1 >= SNAP_T else "share_up"
            h = shift.get(k, {})
            out.append({"kind": "bigger_role", "label": "Bigger role than last year", "text": txt, "hold": h.get("hold"), "hold_n": h.get("n"), "season": True})
        elif d1 <= -SNAP_T or d2 <= -SHARE_T:
            k = "snap_down" if d1 <= -SNAP_T else "share_down"
            h = shift.get(k, {})
            out.append({"kind": "smaller_role", "label": "Smaller role than last year", "text": txt, "hold": h.get("hold"), "hold_n": h.get("n"), "season": True})
    return out


# ---------- profiles for every player (tags) ----------
def _context(cur_w: pd.DataFrame):
    team_now = cur_w.sort_values("week").groupby("gsis")["posteam"].last()
    snaps = {(r.posteam, r.pos, r.week, r.gsis): r.snap for r in cur_w[["posteam", "pos", "week", "gsis", "snap"]].itertuples()}
    mates = cur_w.groupby(["posteam", "pos"])["gsis"].apply(set).to_dict()
    team_weeks = cur_w.groupby("posteam")["week"].apply(lambda s: sorted(set(int(x) for x in s))).to_dict()
    return team_now, snaps, mates, team_weeks


def style_of(model, pos, c, p, games):
    """Style probabilities from this season (noise-aware), or last season if he hasn't played 2 games."""
    src, n = c, games
    if (c is None or games < 2) and p is not None:
        src, n = p, float(p["games"])
    if src is None:
        return None
    pr = classify(model, pos, src, n)
    if pr is None:
        return None
    nm = model["style"][pos]["names"]; o = np.argsort(-pr)
    return {"name": nm[o[0]], "p": round(float(pr[o[0]]), 2), "alt": nm[o[1]], "alt_p": round(float(pr[o[1]]), 2),
            "probs": {nm[i]: round(float(pr[i]), 3) for i in range(len(nm))}, "from_last_season": src is p}


def profile_one(model, g, d, c, p, ctx, names, last_week):
    team_now, snaps, mates, team_weeks = ctx
    pos = d["pos"].iloc[0]; team = team_now.get(g)
    games = float(c["games"]) if c is not None else 0.0
    fp = fingerprint(model, pos, c, games, p)
    est = {m["key"]: m["est"] for m in fp}
    for k in ("car_sh", "tgt_sh", "snap", "db_sh"):
        if k not in est and c is not None:
            est[k] = c.get(k)
    res = {"pos": pos, "team": team, "games": int(games), "tier": tier(pos, est), "style": style_of(model, pos, c, p, games),
           "traits": traits_from(pos, fp), "trends": trends(model, pos, d, team, team_weeks.get(team, []), last_week, snaps,
                                                             mates.get((team, pos), set()), names, g, c, p)}
    sh = _share(pos, d); rz = _r(d.rz_tgt + d.rz_car, d.t_rz_tgt + d.t_rz_car) * 100
    res["series"] = {"weeks": d["week"].astype(int).tolist(), "snap": [round(float(x), 1) for x in d["snap"]],
                     "share": [None if np.isnan(x) else round(float(x), 1) for x in sh],
                     "rz": [None if np.isnan(x) else round(float(x), 1) for x in rz],
                     "share_label": "Opportunity share" if pos == "RB" else "Target share"}
    for t in res["traits"]:
        m = next(x for x in fp if x["key"] == t["key"])
        t["text"] = f"{m['label']}: {fmt_val(m['fmt'], m['est'])} (top {max(1, 100 - t['pct'])}%)" if t["side"] == "hi" else \
            f"{m['label']}: {fmt_val(m['fmt'], m['est'])} (bottom {max(1, t['pct'])}%)"
    return res, fp


def profiles(cur_w: pd.DataFrame, prev_w: pd.DataFrame | None, model: dict, names: dict) -> dict:
    cur = aggregate(cur_w, ("gsis", "pos")).set_index("gsis")
    prev = aggregate(prev_w, ("gsis", "pos")).set_index("gsis") if prev_w is not None and len(prev_w) else pd.DataFrame()
    last_week = int(cur_w["week"].max()); ctx = _context(cur_w)
    out = {}
    for g, d in cur_w.sort_values("week").groupby("gsis"):
        c = cur.loc[g] if g in cur.index else None
        p = prev.loc[g] if (len(prev) and g in prev.index and prev.loc[g]["games"] >= 3) else None
        out[g], _ = profile_one(model, g, d, c, p, ctx, names, last_week)
    return out


# ---------- engine entry ----------
_CACHE: dict = {}


def tables(refresh: bool = False):
    """(season, current weekly table, last season's table, ids) with caching by file time."""
    from . import data as d
    from . import roles_data as rd
    season = d.load().season
    cache = rd.cache_dir()
    f_cur = cache / str(season) / "roles_weekly.csv"
    key = (season, f_cur.stat().st_mtime if f_cur.exists() else None)
    if not refresh and _CACHE.get("tkey") == key:
        return _CACHE["tables"]
    ids = pd.read_csv(d.NFLVERSE / "ids.csv")
    cur = rd.load(season, cache, ids, refresh=refresh)
    try:
        prev = rd.load(season - 1, cache, ids)
        if "ppr" not in prev.columns:  # cached before box scores were added
            prev = rd.load(season - 1, cache, ids, refresh=True)
    except Exception:
        prev = None
    if "ppr" not in cur.columns:
        cur = rd.load(season, cache, ids, refresh=True)
    key = (season, f_cur.stat().st_mtime if f_cur.exists() else None)
    _CACHE.update(tkey=key, tables=(season, cur, prev, ids))
    return _CACHE["tables"]


def board(refresh: bool = False) -> dict:
    """Profiles for every QB/RB/WR/TE who has played this season, keyed by ESPN id."""
    season, cur, prev, ids = tables(refresh)
    key = ("board", _CACHE.get("tkey"))
    if not refresh and _CACHE.get("bkey") == key:
        return _CACHE["board"]
    names = dict(zip(ids["gsis_id"], ids["name"]))
    if "pname" in cur.columns:
        for g, n in cur.dropna(subset=["pname"]).groupby("gsis")["pname"].last().items():
            names.setdefault(g, n)
    model = load_model()
    P = profiles(cur, prev, model, names)
    espn = ids.dropna(subset=["gsis_id", "espn_id"]).drop_duplicates("gsis_id").set_index("gsis_id")["espn_id"].astype(int)
    out = {"season": season, "through_week": int(cur["week"].max()),
           "styles": {p: {"names": m["names"], "yoy_same": m["yoy_same"], "early_acc": m.get("early_acc")} for p, m in model["style"].items()},
           "fit_seasons": model["seasons"], "players": {}}
    for g, prof in P.items():
        if g in espn.index:
            out["players"][str(int(espn[g]))] = {**prof, "name": names.get(g)}
    _CACHE.update(bkey=key, board=out)
    return out
