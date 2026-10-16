"""Does "good against X through week N" carry into week N+1, and does it pay off against a defense that uses / is weak in X?  (2018-25)

X = a defensive mode, labelled per play from nflverse participation + FTN charting (free data only; no PFF, no per-defender stats):
  mz     man vs zone coverage              shell  Cover 0/1, 2, 3, 4/6          rush  4 vs 5+ pass rushers
  dbs    base / nickel / dime+ (DB count)  box    defenders in the box (runs only: <=6, 7, 8+)
Player strength in a mode = his EPA per opportunity in that mode, minus the league average for that mode and season, minus his own overall
(a "differential"), shrunk with a constant K estimated from odd-week vs even-week agreement (method of moments).  Only plays before the
game count ("through week x").  Opportunities: QB dropbacks, WR/TE/RB targets, RB designed runs.

Step 1 `tau`:    how big is the true, stable mode-specific skill (sd of the differential) and does the shrunk to-date strength predict
                 later plays in that mode (calibration slope)?  Same for defenses (is a defense's mode-specific weakness stable?).
Step 2 `stage2`: week-x+1 points minus ESPN's projection, regressed on (a) strength x opponent's expected use of the mode and
                 (b) strength x opponent's mode-specific weakness, with a placebo (strengths dealt to other players), era split and FDR.

  python3 backtest/mode_strength.py lean     # builds the per-play table (about 1 min)
  python3 backtest/mode_strength.py tau
  python3 backtest/mode_strength.py stage2
Findings: project doc claude/mode-strength-2026-10.md.
"""
from __future__ import annotations

import json
import math
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
from app.engine import matchups_data as md, roles_data as rd, store  # noqa: E402

SEAS = list(range(2018, 2026))
S2 = [2021, 2022, 2023, 2024, 2025]          # seasons with ESPN projections
OUT = store.local_dir("mode_strength")
NW = 19                                       # week index 0..18 (0 unused)
KMAX = 3000.0                                 # cap on the shrinkage constant (K = KMAX means "no stable skill found")
MODES = {"mz": ["zone", "man"], "shell": ["Cover 0/1", "Cover 2", "Cover 3", "Cover 4/6"], "rush": ["4 rushers", "5+ rushers"],
         "dbs": ["base (<=4 DB)", "nickel (5 DB)", "dime+ (6+ DB)"], "box": ["<=6 in box", "7 in box", "8+ in box"]}
PASSF = ["mz", "shell", "rush", "dbs"]
ROLES = {"QB": ("QB", "pass", "passer", PASSF), "WR": ("WR", "pass", "receiver", PASSF), "TE": ("TE", "pass", "receiver", PASSF),
         "RBrec": ("RB", "pass", "receiver", PASSF), "RBrun": ("RB", "run", "rusher", ["box", "dbs"])}
DEFS = {"DEFpass": ("pass", PASSF), "DEFrun": ("run", ["box", "dbs"])}


# ───────────────────────────── per-play table ─────────────────────────────
def _dbs(s):
    if not isinstance(s, str):
        return np.nan
    n = sum(int(m.group(1)) for m in re.finditer(r"(\d+)\s+([A-Z]+)", s) if m.group(2) in ("CB", "FS", "SS", "S", "DB"))
    return float(n) if n >= 2 else np.nan


def lean(rebuild: bool = False) -> pd.DataFrame:
    p = OUT / "plays_lean.csv.gz"
    if p.exists() and not rebuild:
        return pd.read_csv(p)
    out = []
    for y in SEAS:
        d = md.load(y, max_age_h=1e9)
        d = d[((d["db"] == 1) | (d["run"] == 1)) & d["epa"].notna()].copy()
        o = d[["season", "week", "game_id", "posteam", "defteam", "db", "run", "epa", "passer", "receiver", "rusher"]].copy()
        o["mz"] = d["defense_man_zone_type"].map({"ZONE_COVERAGE": 0.0, "MAN_COVERAGE": 1.0})
        cov = d["defense_coverage_type"]
        sh = pd.Series(np.nan, index=d.index)
        sh[cov.isin(["COVER_0", "COVER_1"])] = 0
        sh[cov.isin(["COVER_2", "2_MAN"])] = 1
        sh[cov == "COVER_3"] = 2
        sh[cov.isin(["COVER_4", "COVER_6"])] = 3
        o["shell"] = sh
        n = d["n_rushers_p"].where((d["db"] == 1) & (d["n_rushers_p"] >= 2))
        o["rush"] = (n >= 5).astype(float).where(n.notna())
        u = {s: _dbs(s) for s in d["defense_personnel"].dropna().unique()}
        nd = d["defense_personnel"].map(u)
        o["dbs"] = pd.Series(np.select([nd <= 4, nd == 5, nd >= 6], [0.0, 1.0, 2.0], default=np.nan), index=d.index).where(nd.notna())
        b = d["defenders_in_box"].where(d["defenders_in_box"].between(3, 11))
        o["box"] = pd.Series(np.select([b <= 6, b == 7, b >= 8], [0.0, 1.0, 2.0], default=np.nan), index=d.index).where(b.notna() & (d["run"] == 1))
        out.append(o)
    P = pd.concat(out, ignore_index=True)
    P.to_csv(p, index=False, compression="gzip")
    return P


def pos_map() -> pd.DataFrame:
    fr = []
    for y in SEAS:
        w = pd.read_csv(rd.cache_dir() / str(y) / "roles_weekly.csv", usecols=["gsis", "pos"])
        w["season"] = y
        fr.append(w.dropna().drop_duplicates(["season", "gsis"]))
    return pd.concat(fr, ignore_index=True).rename(columns={"gsis": "pid"})


def _excess(d: pd.DataFrame, fam: str) -> pd.Series:
    return d["epa"] - d.groupby(["season", fam])["epa"].transform("mean")


def role_frame(P, pm, role, fam):
    pos, side, col, _ = ROLES[role]
    d = P[(P["db" if side == "pass" else "run"] == 1) & P[col].notna() & P[fam].notna()].copy()
    d = d.rename(columns={col: "pid"}).merge(pm, on=["season", "pid"])
    d = d[d["pos"] == pos]
    d["e"] = _excess(d, fam)
    d["mode"] = d[fam].astype(int)
    return d[["season", "week", "pid", "mode", "e"]].reset_index(drop=True)


def def_frame(P, side, fam):
    d = P[(P["db" if side == "pass" else "run"] == 1) & P[fam].notna()].copy()
    d["e"] = _excess(d, fam)
    d["mode"] = d[fam].astype(int)
    return d.rename(columns={"defteam": "pid"})[["season", "week", "pid", "mode", "e"]].reset_index(drop=True)


# ───────────────────────────── strengths ─────────────────────────────
def arrays(A, M):
    keys = A[["season", "pid"]].drop_duplicates().reset_index(drop=True)
    keys["ix"] = np.arange(len(keys))
    A = A.merge(keys, on=["season", "pid"])
    wk, ix, m = A["week"].clip(1, 18).to_numpy(), A["ix"].to_numpy(), A["mode"].to_numpy()
    n = np.zeros((len(keys), NW, M))
    s = np.zeros_like(n)
    np.add.at(n, (ix, wk, m), 1.0)
    np.add.at(s, (ix, wk, m), A["e"].to_numpy())
    return keys, n, s


def split_stats(n, s):
    odd = np.arange(NW) % 2 == 1
    return n[:, odd].sum(1), s[:, odd].sum(1), n[:, ~odd].sum(1), s[:, ~odd].sum(1)


def tau_from(st, idx, nmin=5):
    no, so, ne, se = (a[idx] for a in st)
    No, Ne = no.sum(1), ne.sum(1)
    ok0 = (No >= 15) & (Ne >= 15)
    ov_o, ov_e = so.sum(1) / np.maximum(No, 1), se.sum(1) / np.maximum(Ne, 1)
    c_o = np.cov(ov_o[ok0], ov_e[ok0])[0, 1]
    res = []
    for m in range(no.shape[1]):
        ok = ok0 & (no[:, m] >= nmin) & (ne[:, m] >= nmin)
        if ok.sum() < 30:
            res.append((np.nan, np.nan, int(ok.sum())))
            continue
        xo, xe = so[ok, m] / no[ok, m] - ov_o[ok], se[ok, m] / ne[ok, m] - ov_e[ok]
        res.append((np.cov(xo, xe)[0, 1], np.corrcoef(xo, xe)[0, 1], int(ok.sum())))
    return c_o, res


def estimate(A, M, boot=300, rng=None):
    """Mode-specific true variance (tau2), its bootstrap CI over player-seasons, and the shrinkage constants K."""
    rng = rng or np.random.default_rng(7)
    keys, n, s = arrays(A, M)
    st = split_stats(n, s)
    P = len(keys)
    c_o, res = tau_from(st, np.arange(P))
    bs = [tau_from(st, rng.integers(0, P, P)) for _ in range(boot)]
    sig2 = A.groupby("mode")["e"].var().reindex(range(M)).to_numpy()
    pi = A.groupby("mode").size().reindex(range(M)).to_numpy() / len(A)
    tau2 = np.array([r[0] for r in res])
    ci = np.array([[np.nanpercentile([b[1][m][0] for b in bs], q) for q in (2.5, 97.5)] for m in range(M)])
    K = np.array([(1 - pi[m]) * sig2[m] / max(tau2[m], sig2[m] / KMAX) if np.isfinite(tau2[m]) else KMAX for m in range(M)])
    K = np.clip(K, 5.0, KMAX)
    co_ci = np.nanpercentile([b[0] for b in bs], [2.5, 97.5])
    Ko = float(np.clip(A["e"].var() / max(c_o, A["e"].var() / KMAX), 5.0, KMAX))
    return dict(keys=keys, n=n, s=s, tau2=tau2, ci=ci, r=np.array([r[1] for r in res]), npairs=[r[2] for r in res], K=K, Ko=Ko,
                tau2_o=c_o, tau2_o_ci=co_ci, sig2=sig2, pi=pi)


def strength(n, s, K, Ko, incl=False):
    """Through week w-1 (strictly before the game): differential d[p, w, m] and overall o[p, w].  incl=True also counts week w itself
    (look-ahead on purpose: the positive control for the tests below)."""
    cn, cs = (np.cumsum(n, 1), np.cumsum(s, 1)) if incl else (np.cumsum(n, 1) - n, np.cumsum(s, 1) - s)
    o = cs.sum(2) / (cn.sum(2) + Ko)
    d = (cs - cn * o[:, :, None]) / (cn + K)
    return d, o, cn, cs


def ols_cl(y, X, starts):
    """OLS with clustered (sandwich) errors; rows must be sorted so each cluster is contiguous (starts = first row of each)."""
    n, p = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ (X.T @ y)
    r = y - X @ b
    Sg = np.add.reduceat(X * r[:, None], starts, axis=0)
    G = len(starts)
    V = XtXi @ (Sg.T @ Sg) @ XtXi * (G / max(G - 1, 1)) * ((n - 1) / max(n - p, 1))
    return b, np.sqrt(np.diag(V))


def pnorm2(t):
    return math.erfc(abs(t) / math.sqrt(2))


def calib(A, E, M):
    """Do the to-date shrunk strengths predict later plays in that mode?  e_i ~ d[p, w, mode_i] + o[p, w] (weeks >= 3)."""
    d, o, _, _ = strength(E["n"], E["s"], E["K"], E["Ko"])
    A = A.merge(E["keys"], on=["season", "pid"]).sort_values("ix")
    wk, ix, m = A["week"].clip(1, 18).to_numpy(), A["ix"].to_numpy(), A["mode"].to_numpy()
    di, oi = d[ix, wk, m], o[ix, wk]
    keep = wk >= 3
    y = A["e"].to_numpy()[keep]
    X = np.column_stack([np.ones(keep.sum()), di[keep], oi[keep]])
    ixk = ix[keep]
    starts = np.r_[0, np.flatnonzero(np.diff(ixk)) + 1]
    b, se = ols_cl(y, X, starts)
    sd = float(di[keep].std())
    return dict(n=int(keep.sum()), b_d=b[1], se_d=se[1], b_o=b[2], se_o=se[2], sd_d=sd, pred=b[1] * sd, pred_se=se[1] * sd)


def units(P, pm):
    for r in ROLES:
        for f in ROLES[r][3]:
            yield r, f, role_frame(P, pm, r, f)
    for r, (side, fams) in DEFS.items():
        for f in fams:
            yield r, f, def_frame(P, side, f)


def tau_report(boot=300):
    P, pm = lean(), pos_map()
    rows, krows = [], {}
    for role, fam, A in units(P, pm):
        M = len(MODES[fam])
        E = estimate(A, M, boot)
        c = calib(A, E, M)
        krows[f"{role}|{fam}"] = dict(K=[float(k) for k in E["K"]], Ko=E["Ko"], tau2=[float(x) for x in E["tau2"]], sig2=[float(x) for x in E["sig2"]])
        for m in range(M):
            lo, hi = E["ci"][m]
            rows.append(dict(unit=role, fam=fam, mode=MODES[fam][m], n_units=E["npairs"][m], share=round(float(E["pi"][m]), 3),
                             tau_sd=math.sqrt(max(E["tau2"][m], 0)) if np.isfinite(E["tau2"][m]) else np.nan,
                             tau2=E["tau2"][m], tau2_lo=lo, tau2_hi=hi, r_half=E["r"][m], K=E["K"][m], sd_play=math.sqrt(E["sig2"][m])))
        rows.append(dict(unit=role, fam=fam, mode="[calibration]", n_units=c["n"], tau_sd=c["sd_d"], tau2=c["b_d"], tau2_lo=c["b_d"] - 1.96 * c["se_d"],
                         tau2_hi=c["b_d"] + 1.96 * c["se_d"], r_half=c["pred"], K=c["pred_se"], sd_play=c["b_o"]))
        print(role, fam, "tau2_overall=%.5f [%.5f, %.5f]" % (E["tau2_o"], *E["tau2_o_ci"]), "Ko=%.0f" % E["Ko"],
              "| calib slope %.2f (se %.2f), per-sd %.4f EPA/play (se %.4f), overall slope %.2f" % (c["b_d"], c["se_d"], c["pred"], c["pred_se"], c["b_o"]), flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "results_tau.csv", index=False)
    (OUT / "K.json").write_text(json.dumps(krows))
    return R


# ───────────────────────────── step 2: week x+1 points vs ESPN ─────────────────────────────
POS = ["QB", "RB", "WR", "TE"]


def games_frame() -> pd.DataFrame:
    from backtest import vegas_signal as vs
    cols = ["season", "week", "gsis", "pos", "opp", "ppr", "espn_proj", "pass_r", "run_r", "tgt", "car", "att"]
    rd_dir = store.local_dir("role_defense")
    a = pd.read_csv(rd_dir / "games.csv.gz", usecols=cols)
    h = pd.read_csv(rd_dir / "games_holdout.csv.gz", usecols=[c for c in cols if c != "espn_proj"] + ["espn_id"])
    h = h[h["season"].isin([2021, 2022])].merge(vs.rows()[["season", "week", "espn_id", "espn_proj"]], on=["season", "week", "espn_id"], how="left")
    g = pd.concat([a, h[cols]], ignore_index=True)
    g["resid"] = g["ppr"] - g["espn_proj"]
    g = g[g["resid"].notna() & g["week"].between(3, 17) & g["pos"].isin(POS)].reset_index(drop=True)
    return g


def def_objects(P, side, fam, K, Ko):
    """Per defense-week: expected use of each mode (u), actual use that game (act), league use (lg) and mode-specific softness (q)."""
    M = len(MODES[fam])
    keys, n, s = arrays(def_frame(P, side, fam), M)
    q, _, cn, _ = strength(n, s, K, Ko)
    nw = n.sum(2)
    cN = cn.sum(2)
    td = cn / np.maximum(cN[:, :, None], 1)
    g = np.cumsum(nw > 0, 1) - (nw > 0)
    seas = keys["season"].to_numpy()
    LG = np.zeros_like(cn)
    for sn in np.unique(seas):
        mk = seas == sn
        tot = cn[mk].sum(0)
        LG[mk] = tot / np.maximum(tot.sum(1, keepdims=True), 1)
    act = n / np.maximum(nw[:, :, None], 1)
    act[nw == 0] = np.nan
    full = n.sum(1)
    share = full / np.maximum(full.sum(1, keepdims=True), 1)
    lgs = {sn: full[seas == sn].sum(0) / full[seas == sn].sum() for sn in np.unique(seas)}
    where = {(a, b): i for i, (a, b) in enumerate(zip(keys["season"], keys["pid"]))}
    dev = np.zeros((len(keys), M))
    for i, (sn, t) in enumerate(zip(keys["season"], keys["pid"])):
        j = where.get((sn - 1, t))
        if j is not None and (sn - 1) in lgs:
            dev[i] = share[j] - lgs[sn - 1]
    use = (seas >= 2019)[:, None] & (g >= 1) & (nw > 0) & (np.arange(NW)[None, :] >= 2) & (np.arange(NW)[None, :] <= 17)
    best = None
    for k in (0.5, 1, 2, 3, 5, 8, 12, 20):
        sse, coefs = 0.0, []
        for m in range(M):
            y = (act - LG)[:, :, m][use]
            X = np.column_stack([np.broadcast_to(dev[:, None, m], g.shape)[use], ((g / (g + k)) * (td - LG)[:, :, m])[use]])
            b = np.linalg.lstsq(X, y, rcond=None)[0]
            sse += float(((y - X @ b) ** 2).sum())
            coefs.append(b)
        if best is None or sse < best[0]:
            best = (sse, k, coefs)
    _, k, coefs = best
    u = LG.copy()
    r2 = []
    for m in range(M):
        pred = coefs[m][0] * dev[:, None, m] + coefs[m][1] * (g / (g + k)) * (td - LG)[:, :, m]
        u[:, :, m] += pred
        y = (act - LG)[:, :, m][use]
        r2.append(1 - float(((y - pred[use]) ** 2).sum()) / float((y ** 2).sum()))
    return dict(keys=keys, u=u, q=q, act=act, lg=LG, k=k, r2=r2, sd_u=[float((u - LG)[:, :, m][use].std()) for m in range(M)])


def player_d(P, pm, role, fam, K, Ko, incl=False):
    M = len(MODES[fam])
    keys, n, s = arrays(role_frame(P, pm, role, fam), M)
    d, _, _, _ = strength(n, s, K, Ko, incl)
    return keys, d


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = np.empty(len(p))
    q[o] = np.minimum.accumulate((p[o] * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def _z(x):
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else x * 0


def fit2(y, ctrl, cols, starts):
    X = np.column_stack([np.ones(len(y)), ctrl] + [_z(c) for c in cols])
    return ols_cl(y, X, starts)


def stage2(naive: bool = False, B: int = 300, leak: bool = False):
    rng = np.random.default_rng(11)
    P, pm = lean(), pos_map()
    Kd = json.loads((OUT / "K.json").read_text())
    G = games_frame()
    print("games", len(G), G.groupby("season").size().to_dict(), "naive K=100" if naive else "estimated K", flush=True)
    dcache, rows, extra = {}, [], []
    for role, (pos, side, col, fams) in ROLES.items():
        Gp = G[G["pos"] == pos].sort_values(["gsis", "season", "week"]).reset_index(drop=True)
        dside = "DEFpass" if side == "pass" else "DEFrun"
        ctrl = Gp["pass_r" if side == "pass" else "run_r"].to_numpy(float)
        ctrl = np.nan_to_num(ctrl - np.nanmean(ctrl))
        y = Gp["resid"].to_numpy(float)
        starts = np.r_[0, np.flatnonzero(Gp["gsis"].to_numpy()[1:] != Gp["gsis"].to_numpy()[:-1]) + 1]
        wk = Gp["week"].to_numpy()
        pj = Gp.groupby(["season", "gsis"]).ngroup().to_numpy()
        pj_season = Gp.groupby(["season", "gsis"])["season"].first().to_numpy()
        comps = []
        for fam in fams:
            M = len(MODES[fam])
            kp, kd = Kd[f"{role}|{fam}"], Kd[f"{dside}|{fam}"]
            Kp, Kop = (np.full(M, 100.0), 100.0) if naive else (np.array(kp["K"]), kp["Ko"])
            Kq, Koq = (np.full(M, 100.0), 100.0) if naive else (np.array(kd["K"]), kd["Ko"])
            if (dside, fam, naive) not in dcache:
                dcache[(dside, fam, naive)] = def_objects(P, side, fam, Kq, Koq)
            Dd = dcache[(dside, fam, naive)]
            keys, D = player_d(P, pm, role, fam, Kp, Kop, leak)
            # opponent row index
            oi = Gp.merge(Dd["keys"].rename(columns={"pid": "opp"}), on=["season", "opp"], how="left")["ix"].to_numpy()
            ok = ~np.isnan(oi)
            oi = np.where(ok, oi, 0).astype(int)
            lg = Dd["lg"][oi, wk, :]
            du = (Dd["u"][oi, wk, :] - lg) * ok[:, None]
            dq = lg * Dd["q"][oi, wk, :] * ok[:, None]
            da = np.nan_to_num(Dd["act"][oi, wk, :] - lg) * ok[:, None]
            pix = Gp[["season", "gsis"]].merge(keys.rename(columns={"pid": "gsis"}), on=["season", "gsis"], how="left")["ix"].to_numpy()
            first = np.unique(pj, return_index=True)[1]
            ixj = np.where(np.isnan(pix[first]), -1, pix[first]).astype(int)    # per sample player -> key index
            comps.append(dict(fam=fam, D=D, ixj=ixj, du=du, dq=dq, da=da, ok=float(ok.mean()), k=Dd["k"], r2=Dd["r2"], sd_u=Dd["sd_u"]))
            print(role, fam, "opp matched %.3f" % ok.mean(), "usage predictor k=%g r2=%s" % (Dd["k"], np.round(Dd["r2"], 3)), "sd(u-lg)=%s" % np.round(Dd["sd_u"], 3), flush=True)

        def indices(perm):
            out = []
            for c in comps:
                ix = c["ixj"][perm[pj]]
                Dg = c["D"][np.maximum(ix, 0), wk, :] * (ix >= 0)[:, None]
                out.append(((c["du"] * Dg).sum(1), (c["dq"] * Dg).sum(1), (c["da"] * Dg).sum(1)))
            return out

        def stats(perm, mask=None):
            idx = indices(perm)
            res = []
            sets = [(U, Q, A) for (U, Q, A) in idx]
            sets.append((sum(_z(u) for u, _, _ in idx), sum(_z(q) for _, q, _ in idx), sum(_z(a) for _, _, a in idx)))   # all families together
            for U, Q, A in sets:
                b, se = fit2(y, ctrl, [U, Q], starts)
                b2, se2 = fit2(y, ctrl, [A], starts)
                res.append((b[2], se[2], b[3], se[3], b2[2], se2[2]))
            return res
        ident = np.arange(len(pj_season))
        obs = stats(ident)
        null = []
        for _ in range(B):
            perm = ident.copy()
            for sn in np.unique(pj_season):
                ids = np.flatnonzero(pj_season == sn)
                perm[ids] = rng.permutation(ids)
            null.append([(r[0] / r[1], r[2] / r[3], r[4] / r[5]) for r in stats(perm)])
        null = np.array(null)       # [B, nfam+1, 3]
        # era split (observed only)
        eras = {}
        for lab, mk in (("e21_23", Gp["season"].to_numpy() <= 2023), ("e24_25", Gp["season"].to_numpy() >= 2024)):
            sub = np.flatnonzero(mk)
            ys, cs_ = y[sub], ctrl[sub]
            idx = indices(ident)
            sg = Gp["gsis"].to_numpy()[sub]
            st = np.r_[0, np.flatnonzero(sg[1:] != sg[:-1]) + 1]
            for fi, (U, Q, A) in enumerate(idx):
                b, se = fit2(ys, cs_, [U[sub], Q[sub]], st)
                eras[(fi, lab)] = (b[2], b[3])
        names = [c["fam"] for c in comps] + ["ALL"]
        for fi, nm in enumerate(names):
            o = obs[fi]
            pe = [(1 + np.sum(np.abs(null[:, fi, j]) >= abs(o[2 * j] / o[2 * j + 1]))) / (1 + B) for j in range(3)]
            rows.append(dict(role=role, fam=nm, N=len(y), mean_proj=float(Gp["espn_proj"].mean()),
                             bU=o[0], seU=o[1], pU=pnorm2(o[0] / o[1]), pU_emp=pe[0], bQ=o[2], seQ=o[3], pQ=pnorm2(o[2] / o[3]), pQ_emp=pe[1],
                             bA=o[4], seA=o[5], pA_emp=pe[2], U_e1=eras.get((fi, "e21_23"), (np.nan, np.nan))[0], U_e2=eras.get((fi, "e24_25"), (np.nan, np.nan))[0],
                             Q_e1=eras.get((fi, "e21_23"), (np.nan, np.nan))[1], Q_e2=eras.get((fi, "e24_25"), (np.nan, np.nan))[1],
                             null_sdU=float(null[:, fi, 0].std()), null_sdQ=float(null[:, fi, 1].std())))
        print(role, "done", flush=True)
    R = pd.DataFrame(rows)
    fam_only = R["fam"] != "ALL"
    for c in ("pU_emp", "pQ_emp"):
        R.loc[fam_only, c.replace("p", "q", 1)] = bh(R.loc[fam_only, c].to_numpy())
    R.to_csv(OUT / ("results_stage2_leak.csv" if leak else "results_stage2_naive.csv" if naive else "results_stage2.csv"), index=False)
    return R


def pts_per_epa():
    """Fantasy points (PPR, standard passing/rushing scoring) per EPA on the same play, 2023-25: the exchange rate for the bound below."""
    d = pd.concat([md.load(y, max_age_h=1e9) for y in (2023, 2024, 2025)], ignore_index=True)
    f = lambda pts, e: float(np.cov(pts, e)[0, 1] / np.var(e, ddof=1))
    t = d[(d["db"] == 1) & d["receiver"].notna() & d["epa"].notna()]
    q = d[(d["db"] == 1) & d["epa"].notna()]
    r = d[(d["run"] == 1) & d["rusher"].notna() & d["epa"].notna()]
    rec = t["complete_pass"].fillna(0) * (1 + t["yards_gained"].fillna(0) / 10) + 6 * t["pass_touchdown"].fillna(0)
    qb = 0.04 * q["complete_pass"].fillna(0) * q["yards_gained"].fillna(0) + 4 * q["pass_touchdown"].fillna(0) - 2 * q["interception"].fillna(0)
    ru = r["yards_gained"].fillna(0) / 10 + 6 * r["rush_touchdown"].fillna(0)
    return {"QB": f(qb, q["epa"]), "WR": f(rec, t["epa"]), "TE": f(rec, t["epa"]), "RBrec": f(rec, t["epa"]), "RBrun": f(ru, r["epa"])}


def bound():
    """How many points could a REAL mode-specific skill be worth per game?  spread = opportunities x points-per-EPA x
    sqrt(sum_m sd_u[m]^2 * tau2[m]), with sd_u the predictable spread of the opponent's use of each mode and tau2 either the point
    estimate or the top of its 95% interval (negative values clipped to 0)."""
    P = lean()
    G = games_frame()
    Kd = json.loads((OUT / "K.json").read_text())
    R = pd.read_csv(OUT / "results_tau.csv")
    slope = pts_per_epa()
    opp = {"QB": G[G.pos == "QB"]["att"].mean(), "WR": G[G.pos == "WR"]["tgt"].mean(), "TE": G[G.pos == "TE"]["tgt"].mean(),
           "RBrec": G[G.pos == "RB"]["tgt"].mean(), "RBrun": G[G.pos == "RB"]["car"].mean()}
    rows = []
    for role, (pos, side, col, fams) in ROLES.items():
        dside = "DEFpass" if side == "pass" else "DEFrun"
        for fam in fams:
            kd = Kd[f"{dside}|{fam}"]
            Dd = def_objects(P, side, fam, np.array(kd["K"]), kd["Ko"])
            t = R[(R["unit"] == role) & (R["fam"] == fam) & (R["mode"] != "[calibration]")]
            pt = np.clip(t["tau2"].to_numpy(float), 0, None)
            hi = np.clip(t["tau2_hi"].to_numpy(float), 0, None)
            sd_u = np.array(Dd["sd_u"])
            f = lambda tau2: opp[role] * slope[role] * math.sqrt(float(np.nansum(sd_u ** 2 * tau2)))
            rows.append(dict(role=role, fam=fam, opps=opp[role], pts_per_epa=slope[role], pts_point=f(pt), pts_top_of_ci=f(hi)))
    B = pd.DataFrame(rows)
    B.to_csv(OUT / "bound.csv", index=False)
    return B


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "tau"
    if cmd == "lean":
        lean(True)
    elif cmd == "tau":
        tau_report()
    elif cmd == "bound":
        print(bound().round(3).to_string(index=False))
    elif cmd == "stage2":
        stage2(naive="naive" in sys.argv[2:], leak="leak" in sys.argv[2:])
