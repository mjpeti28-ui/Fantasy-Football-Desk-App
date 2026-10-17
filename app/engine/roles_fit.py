"""Fit the role model offline and write roles_model.json + roles_library.csv.gz (needs scikit-learn).

python -m app.engine.roles_fit <cache_dir> [<actuals_dir>]
python -m app.engine.roles_fit --pff <cache_dir> <pff_cache>   (WR/TE PFF styles on top of the shipped model)
  cache_dir    holds <season>/roles_weekly.csv for 2016-2025 (python -m app.engine.roles_data 2016 ... 2025)
  actuals_dir  history/actuals (ESPN weekly projections, 2021+) for the "next week vs ESPN" trend test

What it measures (all printed, all stored):
- stabilization: split-half (odd vs even games) reliability per metric -> M = games at which half
  the spread between players is real; year-over-year correlation for the prior.
- style: 3-group Gaussian mixture per position on reliable shape features; early-season accuracy
  of the noise-aware classifier vs the full-season label; year-to-year transitions.
- trends: how often a 2-game change keeps half its size over the next 2 games, and how players
  with each tag scored next week relative to ESPN's projection (vs untagged players).
- comps library: every player-season at every week 2-14 with his usage so far and what happened
  after; leave-one-season-out backtest of comp ranges.
"""
from __future__ import annotations

import io, json, sys, urllib.request, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import cohen_kappa_score

warnings.filterwarnings("ignore")
try:
    from . import roles as R
    from .comps import COMP_K, COMP_W, comp_matrix
except ImportError:
    import roles as R
    from comps import COMP_K, COMP_W, comp_matrix

HERE = Path(__file__).resolve().parent
SEASONS = range(2016, 2026)
QUAL = {"RB": ("car", 40), "WR": ("tgt", 25), "TE": ("tgt", 15), "QB": ("db", 150)}
CHECKPOINTS = range(2, 15)


def name_clusters(pos, cent):
    names = {}
    if pos == "RB":
        names[cent.pd_ratio.idxmax()] = "Passing-down back"; names[cent.pd_ratio.idxmin()] = "Early-down runner"
        for i in cent.index: names.setdefault(i, "All-purpose back")
    elif pos == "WR":
        names[cent.adot.idxmax()] = "Field stretcher"; names[cent.behind_r.idxmax()] = "Underneath / screen"
        for i in cent.index: names.setdefault(i, "Intermediate / possession")
    elif pos == "TE":
        names[cent.tgt_per_snap.idxmax()] = "Receiving TE"; names[cent.tgt_per_snap.idxmin()] = "Blocking / dump-off TE"
        for i in cent.index: names.setdefault(i, "Inline / traditional TE")
    elif pos == "QB":
        names[cent.des_rush_g.idxmax()] = "Dual-threat"; names[cent.des_rush_g.idxmin()] = "Pocket passer"
        for i in cent.index: names.setdefault(i, "Mobile passer")
    return names


def stabilization(w: pd.DataFrame, full: pd.DataFrame) -> dict:
    w = w.sort_values(["gsis", "season", "week"]).copy()
    w["half"] = w.groupby(["gsis", "season"]).cumcount() % 2
    h = R.aggregate(w, ("season", "gsis", "pos", "half"))
    out = {}
    for pos, (col, mn) in QUAL.items():
        keys = list(dict.fromkeys(list(R.METRICS[pos]) + list(R.STYLE[pos])))
        qd = full[(full.pos == pos) & (full.games >= 8) & (full[col] >= mn)]
        q = qd[["season", "gsis"]]
        a = h[(h.pos == pos) & (h.half == 0)].merge(q); b = h[(h.pos == pos) & (h.half == 1)].merge(q)
        m = a.merge(b, on=["season", "gsis"], suffixes=("_a", "_b"))
        y = qd.merge(qd.assign(season=qd.season - 1), on=["gsis", "season"], suffixes=("", "_n"))
        nh = m[["games_a", "games_b"]].mean().mean()
        out[pos] = {}
        for c in keys:
            x = m[[c + "_a", c + "_b"]].replace([np.inf, -np.inf], np.nan).dropna()
            r = float(x.corr().iloc[0, 1]) if len(x) > 30 else np.nan
            M = float(nh * (1 - r) / r) if r and r > 0.02 else 99.0
            yy = y[[c, c + "_n"]].dropna(); ryoy = float(yy.corr().iloc[0, 1]) if len(yy) > 30 else 0.0
            v = qd[c].replace([np.inf, -np.inf], np.nan).dropna()
            out[pos][c] = {"M": round(M, 2), "r_half": round(r, 3), "yoy": round(max(ryoy, 0), 3), "mean": round(float(v.mean()), 5),
                           "sd": round(float(v.std()), 5), "q": [round(float(t), 5) for t in np.percentile(v, np.linspace(0, 100, 21))], "n": int(len(v))}
            print(f"  stab {pos} {c:14s} M={M:5.1f} games  r_half={r:.2f} yoy={ryoy:.2f}")
    return out


def fit_styles(full: pd.DataFrame, w: pd.DataFrame, stab: dict) -> dict:
    model = {"stab": stab}
    w = w.sort_values(["gsis", "season", "week"]).copy(); w["gn"] = w.groupby(["gsis", "season"]).cumcount()
    out = {}
    for pos, (col, mn) in QUAL.items():
        f = list(R.STYLE[pos])
        d = full[(full.pos == pos) & (full.games >= 10) & (full[col] >= mn)].dropna(subset=f).copy()
        for c in f:
            d[c] = d[c].clip(*R.STYLE[pos][c])
        mu, sc = d[f].mean().values, d[f].std(ddof=0).values
        X = (d[f].values - mu) / sc
        gm = GaussianMixture(3, n_init=10, random_state=0, covariance_type="full").fit(X)
        d["cl"] = gm.predict(X)
        cent = d.groupby("cl")[f].mean(); names = name_clusters(pos, cent)
        nm = [names[i] for i in range(3)]
        m = d[["gsis", "season", "cl"]].merge(d[["gsis", "season", "cl"]].assign(season=d.season - 1), on=["gsis", "season"], suffixes=("", "_n"))
        trans = {nm[a]: {nm[b]: round(float(((m.cl == a) & (m.cl_n == b)).sum() / max(1, (m.cl == a).sum())), 3) for b in range(3)} for a in range(3)}
        out[pos] = {"features": f, "mean": mu.round(5).tolist(), "scale": sc.round(5).tolist(), "weights": gm.weights_.round(5).tolist(),
                    "means": gm.means_.round(5).tolist(), "covs": gm.covariances_.round(6).tolist(), "names": nm, "n": int(len(d)),
                    "centroids": {nm[i]: {c: round(float(cent.loc[i, c]), 4) for c in f} for i in range(3)},
                    "yoy_same": round(float((m.cl == m.cl_n).mean()), 3), "kappa": round(float(cohen_kappa_score(m.cl, m.cl_n)), 3),
                    "transitions": trans, "meaning": {nm[i]: {"ppg": round(float(d[d.cl == i].ppr_g.mean()), 1), "n": int((d.cl == i).sum()),
                                                                 "snap": round(float(d[d.cl == i].snap.mean()), 0)} for i in range(3)}}
        # early-season check of the noise-aware classifier
        model["style"] = {pos: out[pos]}
        acc = {}
        for G in (3, 6):
            e = R.aggregate(w[w.gn < G]).merge(d[["gsis", "season", "cl"]], on=["gsis", "season"])
            hits, confs = [], []
            for r in e.itertuples():
                pr = R.classify(model, pos, r._asdict(), G)
                if pr is None:
                    continue
                hits.append(int(np.argmax(pr) == r.cl)); confs.append(float(pr.max()))
            acc[f"G{G}"] = {"acc": round(float(np.mean(hits)), 3), "conf": round(float(np.mean(confs)), 3), "n": len(hits)}
        out[pos]["early_acc"] = acc
        print(f"  style {pos}: {nm} yoy={out[pos]['yoy_same']} kappa={out[pos]['kappa']} early={acc}")
    return out


def trend_holds(w: pd.DataFrame):
    w = w[w.pos.isin(["RB", "WR", "TE"])].sort_values(["gsis", "season", "week"]).copy()
    w["share"] = np.where(w.pos == "RB", R._r(w.tgt + w.car, w.t_tgt + w.t_car), R._r(w.tgt, w.t_tgt)) * 100
    snaps = {(r.season, r.posteam, r.pos, r.week, r.gsis): r.snap for r in w[["season", "posteam", "pos", "week", "gsis", "snap"]].itertuples()}
    grp = w.groupby(["season", "posteam", "pos"])["gsis"].apply(set).to_dict()
    rows = []
    for (g, season), d in w.groupby(["gsis", "season"]):
        if len(d) < 5 or d.posteam.nunique() > 1:
            continue
        pos, team = d.pos.iloc[0], d.posteam.iloc[0]
        wk, sn, sh = d.week.values, d.snap.values, d.share.values
        mates = grp.get((season, team, pos), set()) - {g}
        for i in range(4, len(d)):
            bw, rw = wk[:i - 2], wk[i - 2:i]
            out = [m for m in mates if np.mean([snaps.get((season, team, pos, x, m), 0) for x in bw]) >= 30
                   and all(snaps.get((season, team, pos, x, m), 0) == 0 for x in rw)]
            fw = wk[i:i + 2]
            still = bool(out) and all(snaps.get((season, team, pos, x, m), 0) == 0 for m in out for x in fw)
            rows.append(dict(pos=pos, gsis=g, season=season, next_week=int(wk[i]), d=sn[i - 2:i].mean() - sn[:i - 2].mean(),
                             f=(sn[i:i + 2].mean() - sn[:i - 2].mean()) if i + 2 <= len(d) else np.nan,
                             sd=np.nanmean(sh[i - 2:i]) - np.nanmean(sh[:i - 2]),
                             sf=(np.nanmean(sh[i:i + 2]) - np.nanmean(sh[:i - 2])) if i + 2 <= len(d) else np.nan, out=bool(out), still=still))
    t = pd.DataFrame(rows)
    t["hold"] = t.f >= t.d / 2; t["shold"] = t.sf >= t.sd / 2
    return t


def masks(x):
    return {"snap_up": x.d >= 15, "snap_up_out": (x.d >= 15) & x.out, "snap_up_still_out": (x.d >= 15) & x.still,
            "snap_up_nocause": (x.d >= 15) & ~x.out, "snap_down": x.d <= -15, "share_spike": (x.sd >= 5) & (x.d < 10),
            "share_up_with_snaps": (x.sd >= 5) & (x.d >= 10), "share_drop": (x.sd <= -5) & (x.d > -10),
            "none": (x.d.abs() < 15) & (x.sd.abs() < 5)}


def trend_cal(t):
    res = {}
    for pos, x in t.groupby("pos"):
        x2 = x[x.f.notna()]
        res[pos] = {}
        for k, m in masks(x2).items():
            col = "shold" if k.startswith("share") else "hold"
            y = x2[m]
            res[pos][k] = {"hold": round(float(y[col].mean()), 3) if len(y) else None, "n": int(len(y))}
    return res


def trend_vs_espn(t, actuals: Path, ids: pd.DataFrame):
    """Next game's points minus ESPN's projection, tagged players vs untagged (2021+)."""
    g2e = ids.dropna(subset=["gsis_id", "espn_id"]).drop_duplicates("gsis_id").set_index("gsis_id")["espn_id"].astype(int)
    E = []
    for y in range(2021, 2026):
        try:
            p = pd.read_csv(actuals / str(y) / "espn_weekly_proj.csv"); a = pd.read_csv(actuals / str(y) / "players_weekly.csv")
        except Exception:
            continue
        if "kind" in a:
            a = a[a.kind == "actual"]
        m = a.merge(p, on=["week", "espn_id"]); m["season"] = y; E.append(m[["season", "week", "espn_id", "points", "espn_proj"]])
    if not E:
        return {}
    E = pd.concat(E)
    x = t[t.season >= 2021].copy(); x["espn_id"] = x.gsis.map(g2e)
    x = x.merge(E, left_on=["season", "next_week", "espn_id"], right_on=["season", "week", "espn_id"])
    x = x[x.espn_proj > 0]; x["res"] = x.points - x.espn_proj
    rng = np.random.default_rng(0); res = {}
    for pos, y in x.groupby("pos"):
        mk = masks(y); none = y[mk["none"]]
        res[pos] = {}
        for k, m in mk.items():
            if k == "none":
                continue
            z = y[m]
            if len(z) < 40:
                continue
            gs = z.gsis.unique(); gn = none.gsis.unique()
            iz = {a: b.res.values for a, b in z.groupby("gsis")}; inn = {a: b.res.values for a, b in none.groupby("gsis")}
            diffs = []
            for _ in range(300):
                s1 = np.concatenate([iz[a] for a in rng.choice(gs, len(gs))]); s2 = np.concatenate([inn[a] for a in rng.choice(gn, len(gn))])
                diffs.append(s1.mean() - s2.mean())
            res[pos][k] = {"rel": round(float(z.res.mean() - none.res.mean()), 2), "lo": round(float(np.percentile(diffs, 5)), 2),
                           "hi": round(float(np.percentile(diffs, 95)), 2), "n": int(len(z)), "beat": round(float((z.res > 0).mean()), 3)}
            print(f"  vs ESPN {pos} {k:20s} n={len(z):5d} rel {res[pos][k]['rel']:+.2f} [{res[pos][k]['lo']:+.2f},{res[pos][k]['hi']:+.2f}]")
    return res


def season_shift(w):
    w = w[w.pos.isin(["RB", "WR", "TE"])].copy()
    w["share"] = np.where(w.pos == "RB", R._r(w.tgt + w.car, w.t_tgt + w.t_car), R._r(w.tgt, w.t_tgt)) * 100
    prev = w.groupby(["gsis", "season"]).agg(snap=("snap", "mean"), share=("share", "mean"), n=("week", "size")).reset_index()
    prev["season"] += 1
    rows = []
    for (g, season), d in w.sort_values("week").groupby(["gsis", "season"]):
        if len(d) < 8:
            continue
        p = prev[(prev.gsis == g) & (prev.season == season)]
        if p.empty or p.n.iloc[0] < 6:
            continue
        e, r = d.iloc[:4], d.iloc[4:]
        rows.append(dict(pos=d.pos.iloc[0], ds=e.snap.mean() - p.snap.iloc[0], fs=r.snap.mean() - p.snap.iloc[0],
                         dh=e.share.mean() - p.share.iloc[0], fh=r.share.mean() - p.share.iloc[0]))
    t = pd.DataFrame(rows); out = {}
    for pos, x in t.groupby("pos"):
        f = lambda y, a, b: {"hold": round(float((y[b] >= y[a] / 2).mean()), 3) if len(y) else None, "n": int(len(y))}
        out[pos] = {"snap_up": f(x[x.ds >= 15], "ds", "fs"), "snap_down": f(x[x.ds <= -15], "ds", "fs"),
                    "share_up": f(x[x.dh >= 5], "dh", "fh"), "share_down": f(x[x.dh <= -5], "dh", "fh")}
    return out


def library(w: pd.DataFrame, full: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    w = w[w.week <= 17].sort_values(["gsis", "season", "week"]).copy()
    w["snap_sum"] = w.snap; w["one"] = 1
    cs = R.COUNTS + R.BOX + ["snap_sum", "one"]
    cum = w.groupby(["season", "gsis"])[cs].cumsum()
    cum[["season", "gsis", "week", "pos", "posteam"]] = w[["season", "gsis", "week", "pos", "posteam"]]
    cum["snap_l2"] = w.groupby(["season", "gsis"]).snap.transform(lambda s: s.rolling(2, min_periods=2).mean())
    prev = full[["season", "gsis", "games", "ppr_g", "snap"]].copy(); prev["season"] += 1
    prev = prev.rename(columns={"games": "prev_games", "ppr_g": "prev_ppg", "snap": "prev_snap"})
    teamweeks = w.groupby(["season", "posteam"]).week.apply(lambda s: np.array(sorted(set(s)))).to_dict()
    pw_by = {k: dict(zip(d.week, d.ppr)) for k, d in w.groupby(["season", "gsis"])}
    sw_by = {k: dict(zip(d.week, d.snap)) for k, d in w.groupby(["season", "gsis"])}
    rows = []
    for W in CHECKPOINTS:
        c = cum[cum.week <= W].groupby(["season", "gsis"]).tail(1).copy()
        c = c[c.one >= 2]; c["games"] = c.one; c["snap"] = c.snap_sum / c.one
        c = R.rates(c); c["W"] = W
        a, b, cc, dd, ee = [], [], [], [], []
        for r in c.itertuples():
            pw, sw = pw_by[(r.season, r.gsis)], sw_by[(r.season, r.gsis)]
            fut = [x for x in teamweeks.get((r.season, r.posteam), []) if W < x <= 17]
            pl = [x for x in fut if x in pw]
            a.append(np.mean([pw[x] for x in pl]) if pl else np.nan); b.append(len(pl)); cc.append(len(fut))
            dd.append(np.mean([sw[x] for x in pl]) if pl else np.nan)
            ee.append("|".join("" if x not in pw else f"{pw[x]:.1f}" for x in fut))
        c["ros_ppg"], c["ros_games"], c["ros_team_games"], c["ros_snap"], c["traj"] = a, b, cc, dd, ee
        rows.append(c)
    L = pd.concat(rows, ignore_index=True).merge(prev, on=["season", "gsis"], how="left").merge(meta, on="gsis", how="left")
    L["age"] = (pd.to_datetime(L.season.astype(str) + "-09-01") - pd.to_datetime(L.birth_date, errors="coerce")).dt.days / 365.25
    L["exp"] = L.season - L.rookie_season
    L["share"] = np.where(L.pos == "RB", L.opp_sh, np.where(L.pos == "QB", L.db_sh, L.tgt_sh))
    L["trend"] = L.snap_l2 - L.snap
    keep = ["season", "W", "gsis", "name", "pos", "posteam", "games", "snap", "share", "car_sh", "tgt_sh", "opp_sh", "ay_sh", "db_sh",
            "rec_mix", "pd_ratio", "adot", "behind_r", "deep_r", "tgt_per_snap", "des_rush_g", "scr_rate", "p_adot", "ppr_g", "prev_ppg",
            "prev_games", "age", "exp", "trend", "ros_ppg", "ros_games", "ros_team_games", "ros_snap", "traj"]
    L = L[keep]
    return L[L.snap >= 15].reset_index(drop=True)


def comps_backtest(L: pd.DataFrame) -> dict:
    from scipy.stats import spearmanr
    Q = np.array([.1, .25, .5, .75, .9]); out = {}
    for pos in ["RB", "WR", "TE", "QB"]:
        agg = {"mae": [], "mae_sofar": [], "mae_lin": [], "cov80": [], "cov50": [], "rho": [], "n": 0}
        for W in (3, 5, 8, 11):
            d = L[(L.pos == pos) & (L.W == W) & (L.ros_team_games >= 3)].reset_index(drop=True)
            d, cols, wts = comp_matrix(d, pos)
            X = d[cols].values; mu, sd = X.mean(0), X.std(0).clip(1e-6); X = (X - mu) / sd * wts
            ok = (d.ros_games >= 3).values; y = d.ros_ppg.values
            P = np.full((len(d), 5), np.nan); PL = np.full(len(d), np.nan)
            Xl = np.c_[np.ones(len(d)), d.ppr_g, d.prev_ppg_i, d.has_prev, d.games]
            for s in d.season.unique():
                te = np.where(d.season == s)[0]; tr = np.where((d.season != s) & ok)[0]
                D = ((X[te, None, :] - X[None, tr, :]) ** 2).sum(-1); nn = np.argsort(D, 1)[:, :COMP_K]
                P[te] = np.quantile(y[tr][nn], Q, axis=1).T
                bl = np.linalg.lstsq(Xl[tr], y[tr], rcond=None)[0]; PL[te] = Xl[te] @ bl
            yy, PP = y[ok], P[ok]
            agg["mae"].append(np.mean(np.abs(yy - PP[:, 2]))); agg["mae_sofar"].append(np.mean(np.abs(yy - d.ppr_g.values[ok])))
            agg["mae_lin"].append(np.mean(np.abs(yy - PL[ok])))
            agg["cov80"].append(np.mean((yy >= PP[:, 0]) & (yy <= PP[:, 4]))); agg["cov50"].append(np.mean((yy >= PP[:, 1]) & (yy <= PP[:, 3])))
            agg["rho"].append(np.nanmean([spearmanr(yy[d.season.values[ok] == s], PP[d.season.values[ok] == s, 2])[0] for s in d.season.unique()]))
            agg["n"] += int(ok.sum())
        out[pos] = {k: (round(float(np.mean(v)), 3) if isinstance(v, list) else v) for k, v in agg.items()}
        print(f"  comps {pos}: {out[pos]}")
    return out


PFF_SEASONS = range(2021, 2026)
# WR: k=3 (BIC's pick) labelled earlier but was no steadier year to year than the free styles (kappa .41 vs .41);
# slot vs outside at k=2 is (kappa .54). TE: k=2 by BIC. Backtest: backtest/pff_roles_cluster.py on the PFF branch.
PFF_K = {"WR": 2, "TE": 2}
PFF_AXES = {"WR": ("slot_rate", "adot"), "TE": ("inline_rate", "tgt_per_snap")}


def name_pff(pos, cent):
    if pos == "WR":
        names = {cent.slot_rate.idxmax(): "Slot receiver"}
        for i in cent.index: names.setdefault(i, "Outside receiver")
    else:
        names = {cent.inline_rate.idxmax(): "Inline TE"}
        for i in cent.index: names.setdefault(i, "Move TE")
    return names


def fit_pff_styles(roles_cache: Path, pff_cache: Path) -> None:
    """Refit the WR/TE styles on free features + PFF alignment (2021-25, weeks 1-12) and write them into
    roles_model.json; the free-data WR/TE styles move to model["style_free"] as the no-PFF fallback.
    Only model parameters and aggregates are stored (no per-player PFF values)."""
    from . import roles_pff as rp
    model = json.loads((HERE / "roles_model.json").read_text())
    model.setdefault("style_free", {p: model["style"][p] for p in PFF_K})
    ids = pd.read_csv(HERE.parents[1] / "history" / "nflverse" / "ids.csv")
    w = pd.concat([pd.read_csv(roles_cache / str(y) / "roles_weekly.csv") for y in PFF_SEASONS], ignore_index=True)
    w = w[w.pos.isin(list(PFF_K))]

    def pff(y, lo, hi):
        c = rp.load_counts(rp.window_path(y, lo, hi, pff_cache))
        c["gsis"] = c.player_id.map(rp.match(c, w[w.season == y], ids))
        return c.dropna(subset=["gsis"]).drop_duplicates("gsis").set_index("gsis")

    CNT = ["player_game_count", "slot_snaps", "inline_snaps", "routes", "pass_plays"]

    def window(lo, hi, wins=None):
        """nflverse weeks lo..hi + PFF rates summed over `wins` (default the same weeks), one row per season-player."""
        out = []
        for y in PFF_SEASONS:
            a = R.aggregate(w[(w.season == y) & w.week.between(lo, hi)])
            c = sum(pff(y, *x)[CNT] for x in (wins or [(lo, hi)])) if wins and len(wins) > 1 else pff(y, *(wins or [(lo, hi)])[0])[CNT]
            c = rp.rates(c.dropna())
            out.append(a.join(c[rp.FEATS + ["pass_plays", "player_game_count"]], on="gsis"))
        return pd.concat(out, ignore_index=True)

    full = window(1, 12)
    halves = [window(1, 12, [(1, 3), (7, 9)]), window(1, 12, [(4, 6), (10, 12)])]
    for pos, k in PFF_K.items():
        col, mn = QUAL[pos]
        f = list(R.STYLE[pos]) + list(R.PFF_STYLE[pos])
        clip = {**R.STYLE[pos], **R.PFF_STYLE[pos]}
        q = full[(full.pos == pos) & (full.games >= 8) & (full[col] >= mn) & (full.pass_plays >= 100)].dropna(subset=f).copy()
        for c in f:
            q[c] = q[c].clip(*clip[c])
        # stabilization of the PFF features: weeks 1-3 + 7-9 vs 4-6 + 10-12, summed from counts
        a, b = (h.merge(q[["season", "gsis"]]) for h in halves)
        m2 = a.merge(b, on=["season", "gsis"], suffixes=("_a", "_b"))
        nh = (m2.player_game_count_a.mean() + m2.player_game_count_b.mean()) / 2
        yy = q.merge(q.assign(season=q.season - 1), on=["gsis", "season"], suffixes=("", "_n"))
        for c in R.PFF_STYLE[pos]:
            r = float(m2[[c + "_a", c + "_b"]].dropna().corr().iloc[0, 1]); ry = float(yy[[c, c + "_n"]].corr().iloc[0, 1])
            v = q[c]
            model["stab"][pos][c] = {"M": round(float(nh * (1 - r) / r), 2), "r_half": round(r, 3), "yoy": round(max(ry, 0), 3),
                                     "mean": round(float(v.mean()), 5), "sd": round(float(v.std()), 5),
                                     "q": [round(float(t), 5) for t in np.percentile(v, np.linspace(0, 100, 21))], "n": int(len(v)), "source": "PFF"}
            print(f"  stab {pos} {c:12s} M={model['stab'][pos][c]['M']:5.1f} games  r_half={r:.2f} yoy={ry:.2f}")
        mu, sc = q[f].mean().values, q[f].std(ddof=0).values
        X = (q[f].values - mu) / sc
        gm = GaussianMixture(k, n_init=10, random_state=0, covariance_type="full").fit(X)
        q["cl"] = gm.predict(X)
        cent = q.groupby("cl")[f].mean(); names = name_pff(pos, cent); nm = [names[i] for i in range(k)]
        m = q[["gsis", "season", "cl"]].merge(q[["gsis", "season", "cl"]].assign(season=q.season - 1), on=["gsis", "season"], suffixes=("", "_n"))
        st = {"features": f, "mean": mu.round(5).tolist(), "scale": sc.round(5).tolist(), "weights": gm.weights_.round(5).tolist(),
              "means": gm.means_.round(5).tolist(), "covs": gm.covariances_.round(6).tolist(), "names": nm, "n": int(len(q)),
              "n_fit": round(float(q.games.mean()), 2), "seasons": [min(PFF_SEASONS), max(PFF_SEASONS)], "source": "PFF alignment + nflverse",
              "centroids": {nm[i]: {c: round(float(cent.loc[i, c]), 4) for c in f} for i in range(k)},
              "yoy_same": round(float((m.cl == m.cl_n).mean()), 3), "kappa": round(float(cohen_kappa_score(m.cl, m.cl_n)), 3),
              "transitions": {nm[a]: {nm[b]: round(float(((m.cl == a) & (m.cl_n == b)).sum() / max(1, (m.cl == a).sum())), 3) for b in range(k)} for a in range(k)},
              "meaning": {nm[i]: {"ppg": round(float(q[q.cl == i].ppr.sum() / q[q.cl == i].games.sum()), 1), "n": int((q.cl == i).sum()),
                                  "snap": round(float(q[q.cl == i].snap.mean()), 0)} for i in range(k)}}
        # early check through the shipped classifier (noise-aware), weeks 1..G, vs the weeks 1-12 label;
        # the free-data fallback is scored on the same rows against its own weeks 1-12 label
        free = model["style_free"][pos]
        lab_free = q.apply(lambda r: int(np.argmax(R.classify(model, pos, r, 99, free))), axis=1)
        fq = q.assign(cl_free=lab_free.values)
        mf = fq[["gsis", "season", "cl_free"]].merge(fq[["gsis", "season", "cl_free"]].assign(season=fq.season - 1), on=["gsis", "season"], suffixes=("", "_n"))
        acc, acc_free = {}, {}
        for G in (3, 6):
            e = window(1, G).merge(fq[["season", "gsis", "cl", "cl_free"]], on=["season", "gsis"])
            h, cf, hf = [], [], []
            for r in e.to_dict("records"):
                pr = R.classify(model, pos, r, r["games"], st)
                prf = R.classify(model, pos, r, r["games"], free)
                if pr is None or prf is None:
                    continue
                h.append(int(np.argmax(pr) == r["cl"])); cf.append(float(pr.max())); hf.append(int(np.argmax(prf) == r["cl_free"]))
            acc[f"G{G}"] = {"acc": round(float(np.mean(h)), 3), "conf": round(float(np.mean(cf)), 3), "n": len(h)}
            acc_free[f"G{G}"] = round(float(np.mean(hf)), 3)
        st["early_acc"] = acc
        st["check_free_same_rows"] = {"early_acc": acc_free, "yoy_same": round(float((mf.cl_free == mf.cl_free_n).mean()), 3),
                                      "kappa": round(float(cohen_kappa_score(mf.cl_free, mf.cl_free_n)), 3)}
        model["style"][pos] = st
        print(f"  style {pos}: {nm} n={len(q)} yoy={st['yoy_same']} kappa={st['kappa']} early={acc}")
        print(f"    free-data styles on the same rows: {st['check_free_same_rows']}")
        for i in range(k):
            print(f"    {nm[i]:18s} share={np.mean(q.cl == i):.2f} " + " ".join(f"{c}={cent.loc[i, c]:.3g}" for c in f))
    model["axes_pff"] = {p: list(a) for p, a in PFF_AXES.items()}
    model["version"] = 3
    (HERE / "roles_model.json").write_text(json.dumps(model, separators=(",", ":")))
    print("wrote", HERE / "roles_model.json")


def main(cache: Path, actuals: Path | None):
    w = pd.concat([pd.read_csv(cache / str(y) / "roles_weekly.csv") for y in SEASONS], ignore_index=True)
    w = w[w.pos.isin(["QB", "RB", "WR", "TE"])]
    full = R.aggregate(w)
    print("stabilization"); stab = stabilization(w, full)
    print("styles"); styles = fit_styles(full, w, stab)
    print("trends"); t = trend_holds(w)
    ids = pd.read_csv(cache / "ids.csv") if (cache / "ids.csv").exists() else pd.DataFrame(columns=["gsis_id", "espn_id"])
    model = {"version": 2, "seasons": [min(SEASONS), max(SEASONS)], "stab": stab, "style": styles,
             "axes": {p: list(a) for p, a in R.AXES.items()}, "trend": trend_cal(t), "season_shift": season_shift(w),
             "trend_pts": trend_vs_espn(t, actuals, ids) if actuals else {}}
    # what "starter level" means per position (12-team, 1QB, 2RB/3WR/1TE + flex), median over seasons
    q = full[full.games >= 8]
    rank = {"QB": 12, "RB": 24, "WR": 30, "TE": 12}
    model["starter_ppg"] = {p: round(float(np.median([x.sort_values("ppr_g", ascending=False).ppr_g.iloc[min(rank[p], len(x)) - 1]
                                                       for _, x in q[q.pos == p].groupby("season")])), 1) for p in rank}
    print("  starter-level ppg", model["starter_ppg"])
    print("comps library")
    raw = urllib.request.urlopen("https://github.com/nflverse/nflverse-data/releases/download/players/players.csv", timeout=120).read()
    pl = pd.read_csv(io.BytesIO(raw), low_memory=False)
    meta = pl[["gsis_id", "display_name", "birth_date", "rookie_season"]].rename(columns={"gsis_id": "gsis", "display_name": "name"})
    L = library(w, full, meta)
    L.to_csv(HERE / "roles_library.csv.gz", index=False, float_format="%.4g")
    model["comps"] = {"weights": COMP_W, "K": COMP_K, "bt": comps_backtest(L), "checkpoints": [min(CHECKPOINTS), max(CHECKPOINTS)]}
    (HERE / "roles_model.json").write_text(json.dumps(model, separators=(",", ":")))
    print("wrote", HERE / "roles_model.json", "and roles_library.csv.gz", len(L), "rows")


if __name__ == "__main__":
    if sys.argv[1] == "--pff":  # python -m app.engine.roles_fit --pff <roles_cache> <pff_cache>
        fit_pff_styles(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else None)
