"""Which of Desk's analytics carry signal? One backtest behind every number the analytics views show.

Every candidate metric is measured on 2021-25 (league history + nflverse) at checkpoints W = 4..13,
using only games before W, and compared with what happened from W through week 17:

  repeat  - rank correlation between the metric over weeks < W and the same metric over weeks W..17.
            Does the trait persist at all?
  predict - rank correlation between the metric (weeks < W) and points per game over weeks W..17.
  beyond  - points per game per +1 SD of the metric on (future ppg - the model's estimate at W), where the
            model is v2's est_ppg (ours + ESPN + ECR). Does it know something the model doesn't?
  weekly  - points per +1 SD on (week W points - ESPN's projection for week W). Does it beat ESPN's weekly
            number for start/sit?

Correlations are computed on ranks within (season, W, position) and pooled; CIs come from a
cluster bootstrap over player-seasons (team-seasons, defense-seasons for the other tables).
Labels follow from fixed thresholds set before looking at the results (see label()).

  python3 backtest/analytics_metrics.py run      # all tables -> history/backtest/multi/analytics_*.csv
                                                 #            -> app/engine/analytics_registry.json
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.engine import analytics as an  # noqa: E402

ACT = ROOT / "history" / "actuals"
NFL = ROOT / "history" / "nflverse"
OUT = ROOT / "history" / "backtest" / "multi"
REG = ROOT / "app" / "engine" / "analytics_registry.json"
SEASONS = [2021, 2022, 2023, 2024, 2025]
SKILL = ["QB", "RB", "WR", "TE"]
CHECKS = range(4, 14)
LAST = 17
B = 300


# The weekly panel, tiers and summaries are shared with the engine so the screens compute exactly
# what was tested.
tiers, n_teams, weekly, summarize = an.tiers, an.n_teams, an.weekly, an.summarize


def panel() -> pd.DataFrame:
    ml = pd.read_csv(OUT / "ml_rows.csv", usecols=["season", "W", "espn_id", "position", "est_ppg"])
    rows = []
    for s in SEASONS:
        P = weekly(s)
        m = ml[ml["season"] == s]
        est = m.pivot_table(index="espn_id", columns="W", values="est_ppg")
        for W in CHECKS:
            past, fut = P[P["week"] < W], P[P["week"] >= W]
            a, b = summarize(past), summarize(fut)
            a = a.join(an.trends(past, a))
            nxt = P[P["week"] == W].drop_duplicates("espn_id").set_index("espn_id")
            d = a.add_prefix("p_").join(b.add_prefix("f_"), how="inner")
            d = d[(d["p_n"] >= 3) & (d["f_n"] >= 3)]
            mm = m[m["W"] == W].set_index("espn_id")
            d = d.join(mm[["position", "est_ppg"]], how="inner")
            if W - 3 in est.columns and W in est.columns:
                d["p_est_tr"] = (est[W] - est[W - 3]).reindex(d.index)
            d["next_pts"] = nxt["points"].reindex(d.index)
            d["next_proj"] = nxt["espn_proj"].reindex(d.index)
            d["season"], d["W"] = s, W
            rows.append(d.reset_index())
    D = pd.concat(rows, ignore_index=True)
    D["y_beyond"] = D["f_ppg"] - D["est_ppg"]
    return D


# ------------------------------------------------------------------------------------- statistics
def boot_w(cl: np.ndarray, seed: int = 0) -> np.ndarray:
    u, inv = np.unique(cl, return_inverse=True)
    cnt = np.random.default_rng(seed).poisson(1.0, size=(B, len(u))).astype(float)
    return cnt[:, inv]


def _stats(Wm, x, y, kind):
    S = Wm.sum(1)
    mx, my = Wm @ x / S, Wm @ y / S
    vx = Wm @ (x * x) / S - mx ** 2
    vy = Wm @ (y * y) / S - my ** 2
    cxy = Wm @ (x * y) / S - mx * my
    return cxy / np.sqrt(vx * vy) if kind == "corr" else cxy / vx


def _pslope(Wm, x, c, y):
    """Slope on x holding c fixed (two-regressor OLS), per bootstrap row."""
    S = Wm.sum(1)
    m = lambda v: Wm @ v / S  # noqa: E731
    mx, mc, my = m(x), m(c), m(y)
    cxx, ccc, cxc = m(x * x) - mx ** 2, m(c * c) - mc ** 2, m(x * c) - mx * mc
    cxy, ccy = m(x * y) - mx * my, m(c * y) - mc * my
    return (cxy * ccc - ccy * cxc) / (cxx * ccc - cxc ** 2)


def est(x, y, cl, kind, c=None, seasons=None):
    ok = np.isfinite(x) & np.isfinite(y) & (np.isfinite(c) if c is not None else True)
    x, y, cl = x[ok], y[ok], cl[ok]
    c = c[ok] if c is not None else None
    if len(x) < 60:
        return None
    f = (lambda Wm: _pslope(Wm, x, c, y)) if kind == "pslope" else (lambda Wm: _stats(Wm, x, y, kind))
    pt = float(f(np.ones((1, len(x))))[0])
    bs = f(boot_w(cl))
    lo, hi = np.nanpercentile(bs, [5, 95])
    out = {"v": round(pt, 3), "lo": round(float(lo), 3), "hi": round(float(hi), 3), "n": int(len(x))}
    if seasons is not None:  # same sign in how many seasons?
        ss = seasons[ok]
        per = []
        for s in np.unique(ss):
            k = ss == s
            if k.sum() >= 40:
                per.append(float(f(k[None, :].astype(float))[0]))
        out["seasons_same_sign"] = int(sum(np.sign(v) == np.sign(pt) for v in per))
        out["seasons"] = len(per)
    return out


def grp_rank(D, col, by):
    return D.groupby(by)[col].rank(pct=True)


def grp_z(D, col, by):
    g = D.groupby(by)[col]
    return (D[col] - g.transform("mean")) / g.transform("std")


# --------------------------------------------------------------------------------------- labels
def strength(r):
    if r is None:
        return "untested"
    a = abs(r)
    return "strong" if a >= 0.6 else "moderate" if a >= 0.3 else "weak" if a >= 0.15 else "none"


def label(t: dict, holds: Optional[bool] = None) -> str:
    """Fixed before results: 'priced' = predict >= 0.3, 'trait' = repeat >= 0.3, else 'noise'.
    Added after the first run (both only demote): a 'beyond' CI clear of 0 is an 'edge' only if adding
    the stat to the model alone also lowered held-out error (leave-one-season-out, CI above 0);
    otherwise it is an in-sample 'tilt'. 'adds' would need an edge of 0.25+ ppg per SD in 4 of 5
    seasons as well; nothing met it."""
    b = t.get("beyond")
    if b is not None and (b["lo"] > 0 or b["hi"] < 0):
        if holds:
            big = abs(b["v"]) >= 0.25 and b.get("seasons_same_sign", 0) >= 4
            return "adds" if big else "edge"
        return "tilt"
    p, r = t.get("predict"), t.get("repeat")
    if p is not None and abs(p["v"]) >= 0.3:
        return "priced"        # predicts later points, but the model already has it
    if r is not None and abs(r["v"]) >= 0.3:
        return "trait"         # repeats, but isn't tied to later points beyond the model
    return "noise"


LABEL_TEXT = {
    "adds": "Adds to the model",
    "edge": "Small edge beyond the model",
    "tilt": "Tilt in past seasons that didn't hold out of sample",
    "priced": "Predicts, already in the model",
    "trait": "Repeats, but doesn't predict points",
    "noise": "Mostly noise",
    "weak": "Weak signal",
    "descriptive": "Descriptive only",
}

PLAYER = {  # key: (name, past col, future col or None)
    "ppg": ("Points per game", "p_ppg", "f_ppg"),
    "xfp": ("Expected points per game (usage)", "p_xfp", "f_xfp"),
    "fpoe": ("Points over expected (efficiency/luck)", "p_fpoe", "f_fpoe"),
    "opp": ("Opportunities per game", "p_opp", "f_opp"),
    "snap": ("Snap share", "p_snap", "f_snap"),
    "tsh": ("Target share", "p_tsh", "f_tsh"),
    "ays": ("Air-yards share", "p_ays", "f_ays"),
    "wopr": ("WOPR", "p_wopr", "f_wopr"),
    "csh": ("Team carry share", "p_csh", "f_csh"),
    "cv": ("Week-to-week volatility (CV)", "p_cv", "f_cv"),
    "spread": ("Floor-to-ceiling spread", "p_spread", "f_spread"),
    "smash": ("Smash-week rate", "p_smash", "f_smash"),
    "start": ("Startable-week rate", "p_start", "f_start"),
    "dud": ("Dud-week rate", "p_dud", "f_dud"),
    "resid": ("Average vs ESPN's projection", "p_resid", "f_resid"),
    "beat": ("Beat-ESPN rate", "p_beat", "f_beat"),
    "xfp_tr": ("Usage trend (last 3 vs season, xFP)", "p_xfp_tr", None),
    "snap_tr": ("Snap trend (last 3 vs season)", "p_snap_tr", None),
    "tsh_tr": ("Target-share trend (last 3 vs season)", "p_tsh_tr", None),
    "opp_tr": ("Opportunity trend (last 3 vs season)", "p_opp_tr", None),
    "est_tr": ("Model value trend (last 3 weeks)", "p_est_tr", None),
}
TREND_BASE = {"xfp_tr": ("p_xfp", "f_xfp"), "snap_tr": ("p_snap", "f_snap"),
              "tsh_tr": ("p_tsh", "f_tsh"), "opp_tr": ("p_opp", "f_opp")}
STICK_RAW = {"fpoe": ("p_fpoe", "f_fpoe"), "resid": ("p_resid", "f_resid")}  # share of the gap that carries on


def players(D: pd.DataFrame, holds: Optional[dict] = None) -> dict:
    holds = holds or {}
    by = ["season", "W", "position"]
    cl = (D["season"] * 10_000_000 + D["espn_id"]).to_numpy()
    res = {}
    for key, (name, pc, fc) in PLAYER.items():
        if pc not in D:
            continue
        out = {"name": name, "by_pos": {}}
        for pos in ["ALL"] + SKILL:
            m = np.ones(len(D), bool) if pos == "ALL" else (D["position"] == pos).to_numpy()
            if D.loc[m, pc].notna().sum() < 60:
                continue
            Dm = D[m]
            rx = grp_rank(Dm, pc, by).to_numpy()
            zx = grp_z(Dm, pc, by).to_numpy()
            t = {}
            if fc is not None:
                t["repeat"] = est(rx, grp_rank(Dm, fc, by).to_numpy(), cl[m], "corr")
            elif key in TREND_BASE:  # share of a recent change that sticks: (future - season) on (last3 - season)
                b0, b1 = TREND_BASE[key]
                t["sticks"] = est(Dm[pc].to_numpy(), (Dm[b1] - Dm[b0]).to_numpy(), cl[m], "slope")
            if key in STICK_RAW:
                b0, b1 = STICK_RAW[key]
                t["sticks"] = est(Dm[b0].to_numpy(), Dm[b1].to_numpy(), cl[m], "slope")
            t["predict"] = est(rx, grp_rank(Dm, "f_ppg", by).to_numpy(), cl[m], "corr")
            sea = Dm["season"].to_numpy()
            t["beyond"] = est(zx, Dm["f_ppg"].to_numpy(), cl[m], "pslope", c=Dm["est_ppg"].to_numpy(), seasons=sea)
            t["weekly"] = est(zx, Dm["next_pts"].to_numpy(), cl[m], "pslope", c=Dm["next_proj"].to_numpy(), seasons=sea)
            t["label"] = label(t, holds.get(key))
            out["by_pos"][pos] = t
        if out["by_pos"]:
            top = out["by_pos"].get("ALL", next(iter(out["by_pos"].values())))["label"]
            for t in out["by_pos"].values():  # a position can't claim more than its pooled stat
                if t["label"] in ("adds", "edge") and top not in ("adds", "edge"):
                    t["label"] = "tilt"
                elif t["label"] == "adds" and top == "edge":
                    t["label"] = "edge"
            out["label"] = top
            res[key] = out
    return res


# ------------------------------------------------------------------------------------ joint edge
# Fixed after the first run: every stat whose 'beyond' CI cleared 0 there (beat-ESPN dropped as a
# duplicate of the average vs ESPN). Judged leave-one-season-out against the model alone.
EDGE = ["p_snap_tr", "p_dud", "p_est_tr", "p_xfp_tr", "p_opp_tr", "p_resid"]


def edge(D: pd.DataFrame, feats=None) -> dict:
    feats = feats or EDGE
    by = ["season", "W", "position"]
    X = np.column_stack([grp_z(D, c, by).fillna(0).to_numpy() for c in feats])
    e, y, sea = D["est_ppg"].to_numpy(), D["f_ppg"].to_numpy(), D["season"].to_numpy()
    ok = np.isfinite(y) & np.isfinite(e)
    base = np.full(len(D), np.nan)
    full = np.full(len(D), np.nan)
    for s_ in SEASONS:
        tr, te = ok & (sea != s_), ok & (sea == s_)
        A0 = np.column_stack([np.ones(tr.sum()), e[tr]])
        A1 = np.column_stack([A0, X[tr]])
        b0 = np.linalg.lstsq(A0, y[tr], rcond=None)[0]
        b1 = np.linalg.lstsq(A1, y[tr], rcond=None)[0]
        base[te] = np.column_stack([np.ones(te.sum()), e[te]]) @ b0
        full[te] = np.column_stack([np.ones(te.sum()), e[te], X[te]]) @ b1
    d2 = (y - base) ** 2 - (y - full) ** 2          # > 0 means the edge helped
    cl = (D["season"] * 10_000_000 + D["espn_id"]).to_numpy()
    k = ok & np.isfinite(d2)
    Wm = boot_w(cl[k])
    bs = (Wm @ d2[k]) / Wm.sum(1)
    rmse0, rmse1 = float(np.sqrt(np.nanmean((y - base)[k] ** 2))), float(np.sqrt(np.nanmean((y - full)[k] ** 2)))
    D2 = D[k].assign(base=base[k], full=full[k])
    rk = lambda c: grp_rank(D2, c, by)  # noqa: E731
    r0 = float(np.corrcoef(rk("base"), rk("f_ppg"))[0, 1])
    r1 = float(np.corrcoef(rk("full"), rk("f_ppg"))[0, 1])
    A = np.column_stack([np.ones(ok.sum()), e[ok], X[ok]])
    beta = np.linalg.lstsq(A, y[ok], rcond=None)[0]
    ed = X[ok] @ beta[2:]
    per = {}
    for s_ in SEASONS:
        m_ = k & (sea == s_)
        per[str(s_)] = round(float(np.nanmean(d2[m_])), 3)
    return {"features": [c[2:] for c in feats], "coef": [round(float(b), 4) for b in beta[2:]],
            "intercept": round(float(beta[0]), 4), "est_coef": round(float(beta[1]), 4),
            "rmse_model": round(rmse0, 3), "rmse_with_edge": round(rmse1, 3),
            "mse_gain": {"v": round(float(np.nanmean(d2[k])), 3), "lo": round(float(np.percentile(bs, 5)), 3),
                         "hi": round(float(np.percentile(bs, 95)), 3)},
            "mse_gain_by_season": per, "rank_model": round(r0, 4), "rank_with_edge": round(r1, 4),
            "edge_sd": round(float(np.std(ed)), 3), "edge_p95": round(float(np.percentile(np.abs(ed), 95)), 3),
            "n": int(k.sum())}


def metric_sd(D: pd.DataFrame) -> dict:
    """Typical within-checkpoint SD of each stat by position, to turn a 'per SD' result into a size."""
    out = {}
    for key, (_n, pc, _f) in PLAYER.items():
        if pc not in D:
            continue
        g = D.groupby(["season", "W", "position"])[pc].std().groupby(level="position").median()
        out[key] = {p: round(float(v), 4) for p, v in g.items() if np.isfinite(v)}
    return out


# ---------------------------------------------------------------------------------------- teams
def team_panel() -> pd.DataFrame:
    rows = []
    for s in SEASONS:
        ss = pd.read_csv(OUT / f"start_sit_{s}.csv")
        R = pd.read_csv(ACT / str(s) / "rosters_by_week.csv")
        R = R[~R["lineup_slot"].isin(["BE", "IR"])]
        P = pd.read_csv(ACT / str(s) / "players_weekly.csv")
        P = P[P["kind"] == "actual"][["espn_id", "week", "points"]]
        E = pd.read_csv(ACT / str(s) / "espn_weekly_proj.csv").drop_duplicates(["week", "espn_id"])
        R = R.merge(P, on=["espn_id", "week"], how="left").merge(E, on=["espn_id", "week"], how="left")
        lk = R.groupby(["team_id", "week"]).apply(lambda g: (g["points"].fillna(0) - g["espn_proj"].fillna(0)).sum())
        ss = ss[ss["week"] <= 14].copy()
        ss["eff"] = ss["manager"] / ss["optimal"]
        ss["luck"] = ss.set_index(["team_id", "week"]).index.map(lk.to_dict())
        ss["sd_proxy"] = ss["manager"]
        ss["season"] = s
        rows.append(ss)
    T = pd.concat(rows, ignore_index=True)
    out = []
    for (s), g in T.groupby("season"):
        for W in range(4, 13):
            a = g[g["week"] < W].groupby("team_id").agg(ppg=("manager", "mean"), eff=("eff", "mean"),
                                                         luck=("luck", "mean"), sd=("manager", "std"))
            b = g[g["week"] >= W].groupby("team_id").agg(ppg=("manager", "mean"), eff=("eff", "mean"),
                                                         luck=("luck", "mean"), sd=("manager", "std"))
            d = a.add_prefix("p_").join(b.add_prefix("f_"), how="inner").reset_index()
            d["season"], d["W"] = s, W
            out.append(d)
    return pd.concat(out, ignore_index=True)


TEAM = {"ppg": "Points per week", "eff": "Lineup efficiency (actual / best possible)",
        "luck": "Starters vs ESPN's projection (luck)", "sd": "Week-to-week swing"}


def teams(T: pd.DataFrame) -> dict:
    by = ["season", "W"]
    cl = (T["season"] * 100 + T["team_id"]).to_numpy()
    res = {}
    for k, name in TEAM.items():
        rx = grp_rank(T, f"p_{k}", by).to_numpy()
        t = {"repeat": est(rx, grp_rank(T, f"f_{k}", by).to_numpy(), cl, "corr"),
             "predict": est(rx, grp_rank(T, "f_ppg", by).to_numpy(), cl, "corr")}
        r = t["repeat"]["v"] if t["repeat"] else 0
        lo = t["repeat"]["lo"] if t["repeat"] else 0
        t["label"] = "trait" if r >= 0.3 else "weak" if lo > 0 else "noise"
        res[k] = {"name": name, "label": t["label"], "all": t}
    return res


# ------------------------------------------------------------------------------------- schedule
def schedule_panel() -> pd.DataFrame:
    sch = pd.read_csv(NFL / "schedules.csv", low_memory=False)
    sch = sch[sch["game_type"] == "REG"]
    out = []
    for s in SEASONS:
        st = pd.read_csv(NFL / str(s) / "player_stats.csv")
        st = st[st["position"].isin(SKILL) & st["week"].between(1, LAST)]
        sc = sch[sch["season"] == s]
        opp = pd.concat([sc[["week", "home_team", "away_team"]].rename(columns={"home_team": "team", "away_team": "def"}),
                         sc[["week", "away_team", "home_team"]].rename(columns={"away_team": "team", "home_team": "def"})])
        st = st.merge(opp, on=["week", "team"], how="inner")
        F = st.groupby(["def", "week", "position"], as_index=False)["fantasy_points_ppr"].sum()
        for W in range(6, 12):
            a = F[F["week"] < W].groupby(["def", "position"])["fantasy_points_ppr"].mean()
            nx = F[F["week"].between(W, W + 3)].groupby(["def", "position"])["fantasy_points_ppr"].mean()
            po = F[F["week"].between(15, 17)].groupby(["def", "position"])["fantasy_points_ppr"].mean()
            d = pd.DataFrame({"p": a, "next4": nx, "playoffs": po}).dropna(subset=["p"]).reset_index()
            d["season"], d["W"] = s, W
            out.append(d)
    return pd.concat(out, ignore_index=True)


def schedule(S: pd.DataFrame) -> dict:
    res = {}
    for pos in ["ALL"] + SKILL:
        m = np.ones(len(S), bool) if pos == "ALL" else (S["position"] == pos).to_numpy()
        Sm = S[m]
        by = ["season", "W", "position"]
        cl = (Sm["season"].astype(str) + Sm["def"]).to_numpy()
        rx = grp_rank(Sm, "p", by).to_numpy()
        res[pos] = {"next4": est(rx, grp_rank(Sm, "next4", by).to_numpy(), cl, "corr"),
                    "playoffs": est(rx, grp_rank(Sm, "playoffs", by).to_numpy(), cl, "corr")}
    po = res["ALL"]["playoffs"] or {"v": 0, "lo": 0}
    return {"fpa": {"name": "Fantasy points allowed by position (through week W)", "by_pos": res,
                    "label": "trait" if po["v"] >= 0.3 else "weak" if po["lo"] > 0 else "noise"}}


def schedule_epa() -> dict:
    """Matchups' own defense rating (EPA allowed, prior + this season) before week W vs what the defense allowed
    in weeks W..W+3 and in the fantasy playoffs (15-17). 2023-25 only (play-by-play cache)."""
    from backtest import matchup_mechanisms as mm
    P = mm.plays()
    R = mm.weekly_ratings(P)
    real = []
    for (s, d), g in P.groupby(["season", "defteam"]):
        for lab, lo, hi in (("next4", None, None), ("playoffs", 15, 17)):
            pass
    out = {}
    for side, flag in (("pass", "db"), ("run", "run")):
        A = P[P[flag] == 1].groupby(["season", "defteam", "week"])["epa"].agg(["sum", "count"]).reset_index()
        rows = []
        for W in range(6, 12):
            r = R[R["week"] == W][["season", "team", f"{side}_r"]].rename(columns={"team": "defteam", f"{side}_r": "p"})
            n4 = A[A["week"].between(W, W + 3)].groupby(["season", "defteam"])[["sum", "count"]].sum()
            po = A[A["week"].between(15, 17)].groupby(["season", "defteam"])[["sum", "count"]].sum()
            r = r.set_index(["season", "defteam"])
            r["next4"] = n4["sum"] / n4["count"]
            r["playoffs"] = po["sum"] / po["count"]
            r = r.reset_index()
            r["W"] = W
            rows.append(r)
        S = pd.concat(rows, ignore_index=True).dropna()
        by = ["season", "W"]
        cl = (S["season"].astype(str) + S["defteam"]).to_numpy()
        rx = grp_rank(S, "p", by).to_numpy()
        out[side] = {"next4": est(rx, grp_rank(S, "next4", by).to_numpy(), cl, "corr"),
                     "playoffs": est(rx, grp_rank(S, "playoffs", by).to_numpy(), cl, "corr")}
    po = out["pass"]["playoffs"] or {"v": 0, "lo": 0}
    return {"name": "Matchups defense rating (EPA allowed) before week W", "by_side": out, "seasons": [2023, 2024, 2025],
            "label": "trait" if po["v"] >= 0.3 else "weak" if po["lo"] > 0 else "noise"}


# ------------------------------------------------------------------------------------------ run
def run():
    D = panel()
    D.to_csv(OUT / "analytics_panel.csv", index=False)
    E = edge(D)
    E["single"] = {c[2:]: {k: edge(D, [c])[k] for k in ("mse_gain", "mse_gain_by_season", "rank_model", "rank_with_edge")}
                   for c in EDGE}
    holds = {k: v["mse_gain"]["lo"] > 0 for k, v in E["single"].items()}
    P = players(D, holds)
    T = teams(team_panel())
    S = schedule(schedule_panel())
    S["epa"] = schedule_epa()
    reg = {"built": date.today().isoformat(), "edge": E, "sd": metric_sd(D), "seasons": SEASONS, "checks": [min(CHECKS), max(CHECKS)],
           "labels": LABEL_TEXT, "players": P, "teams": T, "schedule": S,
           "descriptive": {"swing_plays": "Plays that moved win chance most (what happened)",
                           "draft_vs_now": "Draft slot vs current value (what happened)",
                           "percentiles": "League percentiles inherit the label of the stat they rank"},
           "tiers_note": "start = top N_teams at QB/TE, top 2.5*N_teams at RB/WR; smash = top 3 QB/TE, top "
                         "quarter of the startable RB/WR pool; dud = outside twice the startable pool"}
    REG.write_text(json.dumps(reg, indent=1))
    flat = []
    for k, v in P.items():
        for pos, t in v["by_pos"].items():
            flat.append({"metric": k, "pos": pos, "label": t["label"],
                         **{f"{s}_{f}": (t.get(s) or {}).get(f) for s in ("repeat", "sticks", "predict", "beyond", "weekly")
                            for f in ("v", "lo", "hi", "n")}})
    pd.DataFrame(flat).to_csv(OUT / "analytics_players.csv", index=False)
    print(f"panel rows {len(D)}; players {D['espn_id'].nunique()}")
    print("edge", json.dumps(E))
    for k, v in P.items():
        a = v["by_pos"].get("ALL", {})
        f = lambda e: "   -  " if not e else f"{e['v']:+.2f}[{e['lo']:+.2f},{e['hi']:+.2f}]"
        ss = lambda e: "" if not e or "seasons" not in e else f"({e['seasons_same_sign']}/{e['seasons']})"  # noqa: E731
        print(f"{k:8s} {a.get('label','?'):7s} rep {f(a.get('repeat') or a.get('sticks'))} pred {f(a.get('predict'))} "
              f"beyond {f(a.get('beyond'))}{ss(a.get('beyond'))} wk {f(a.get('weekly'))}{ss(a.get('weekly'))}")
    for k, v in T.items():
        a = v["all"]
        print(f"team {k:5s} {v['label']:6s} rep {a['repeat']} pred {a['predict']}")
    print("schedule", json.dumps(S["fpa"]["by_pos"]))
    print("epa", json.dumps(S["epa"]["by_side"]))
    for k in STICK_RAW:
        print(k, "sticks", {p: P[k]["by_pos"][p].get("sticks") for p in P[k]["by_pos"]})


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        run()
    else:
        print(__doc__)
