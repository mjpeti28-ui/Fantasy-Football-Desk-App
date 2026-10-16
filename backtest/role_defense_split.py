"""Do the role-style groups (Roles screen) respond differently to different kinds of defenses?  (2023-25)

Step 1 (this file, `build`): one row per player-game with
  - the player's style probabilities AS OF THE PRIOR WEEK (roles_library checkpoint W = week-1, classified with the shipped
    roles_model.json, so nothing from the game itself or later is used),
  - the defense he faced: ratings known before kickoff (pass/run EPA-allowed rating, prior-season-carryover man/pressure,
    to-date blitz) and what the defense actually did in that game (man share, blitz share, shell mix),
  - outcomes: PPR points, ESPN's pre-kickoff projection, and usage (targets / carries / their shares) with the player's own
    as-of-prior-week baseline.
Step 2 (`analyze`) tests style x defense interactions with player-clustered inference, a fake-style-label placebo and a 2023-24 vs 2025 split;
`tiers` does the same for the volume tiers (Workhorse, Alpha target share ...).  backtest/role_defense_holdout.py repeats the pass/run-defense
questions on 2016-22 and pools all ten seasons.  Findings: project doc claude/role-defense-split-2026-10.md.

  python3 backtest/role_defense_split.py build      # writes app/cache.nosync/role_defense/games.csv.gz
  python3 backtest/role_defense_split.py analyze    # ~60 s, writes results.csv + placebo_chi2.npy
  python3 backtest/role_defense_split.py tiers      # ~70 s, writes results_tier.csv
Free data only: nflverse play-by-play, FTN charting and participation via nflverse; ESPN projections.  No PFF, no per-defender stats;
offensive snap counts arrive through the existing roles pipeline (roles_data.py), as they do for the Roles screen.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
from app.engine import matchups_data as md, roles, roles_data as rd, store  # noqa: E402
from backtest import matchup_mechanisms as mm  # noqa: E402
from backtest import vegas_signal as vs  # noqa: E402

SEAS = [2023, 2024, 2025]
POS = ["QB", "RB", "WR", "TE"]
OUT = store.local_dir("role_defense")
PERSIST = {"man": 0.43, "blitz": 0.54, "pressure": 0.50}  # year-to-year carry-over measured for the Matchups screen
PRIOR_G = 5.0  # games of prior for the to-date blitz rate


def defense_games() -> pd.DataFrame:
    """One row per defense-game: what the defense actually did (dropbacks only), plus pre-game expectations."""
    rows = []
    for s in SEAS:
        d = md.load(s)
        d = d[d["db"] == 1].copy()
        d["man"] = d["defense_man_zone_type"].map({"MAN_COVERAGE": 1.0, "ZONE_COVERAGE": 0.0})
        cov = d["defense_coverage_type"]
        d["c3"] = np.where(cov.isna(), np.nan, (cov == "COVER_3").astype(float))
        d["c2"] = np.where(cov.isna(), np.nan, cov.isin(["COVER_2", "2_MAN"]).astype(float))
        d["c46"] = np.where(cov.isna(), np.nan, cov.isin(["COVER_4", "COVER_6"]).astype(float))
        d["c01"] = np.where(cov.isna(), np.nan, cov.isin(["COVER_0", "COVER_1"]).astype(float))
        d["blitz"] = (d["n_blitzers"].fillna(0) > 0).astype(float).where(d["n_blitzers"].notna())
        g = d.groupby(["season", "week", "defteam"]).agg(db=("db", "size"), man_g=("man", "mean"), blitz_g=("blitz", "mean"), press_g=("was_pressure", "mean"),
                                                          c3_g=("c3", "mean"), c2_g=("c2", "mean"), c46_g=("c46", "mean"), c01_g=("c01", "mean")).reset_index()
        rows.append(g)
    G = pd.concat(rows, ignore_index=True).rename(columns={"defteam": "opp"})
    G = G[G["db"] >= 15]
    # pre-game expectations
    out = []
    for s in SEAS:
        cur = G[G["season"] == s]
        prev = G[G["season"] == s - 1]
        for k, col in (("man", "man_g"), ("pressure", "press_g"), ("blitz", "blitz_g")):
            pass
        lg_prev = {c: (prev[c].mean() if len(prev) else np.nan) for c in ("man_g", "press_g", "blitz_g")}
        tp = prev.groupby("opp")[["man_g", "press_g", "blitz_g"]].mean() if len(prev) else pd.DataFrame(columns=["man_g", "press_g", "blitz_g"])
        for w in range(2, 19):
            past = cur[cur["week"] < w]
            lg_cur = past["blitz_g"].mean() if len(past) else np.nan
            for t in cur["opp"].unique():
                r = {"season": s, "week": w, "opp": t}
                for k, c in (("man", "man_g"), ("pressure", "press_g"), ("blitz", "blitz_g")):
                    if t in tp.index and not np.isnan(lg_prev[c]):
                        r[f"{k}_prior"] = lg_prev[c] + PERSIST[k] * (tp.loc[t, c] - lg_prev[c])
                    else:
                        r[f"{k}_prior"] = np.nan
                pt = past[past["opp"] == t]
                base = r["blitz_prior"] if not np.isnan(r["blitz_prior"]) else lg_cur
                n = len(pt)
                r["blitz_td"] = (n * pt["blitz_g"].mean() + PRIOR_G * base) / (n + PRIOR_G) if n and not np.isnan(base) else base
                out.append(r)
    E = pd.DataFrame(out)
    return G.merge(E, on=["season", "week", "opp"], how="right")


def style_table() -> pd.DataFrame:
    """Style probabilities from the library checkpoint at W = game week - 1 (cumulative through that week)."""
    model = roles.load_model()
    lib = pd.read_csv(ROOT / "app" / "engine" / "roles_library.csv.gz")
    lib = lib[lib["season"].isin(SEAS) & lib["pos"].isin(POS)].copy()
    names = {p: model["style"][p]["names"] for p in POS}
    P = []
    for r in lib.to_dict("records"):
        pr = roles.classify(model, r["pos"], r, float(r["games"]))
        P.append(pr if pr is not None else np.full(3, np.nan))
    P = np.array(P)
    for i in range(3):
        lib[f"p{i}"] = P[:, i]
    lib["style"] = [names[p][int(np.nanargmax(row))] if not np.isnan(row).any() else None for p, row in zip(lib["pos"], P)]
    lib["pmax"] = np.nanmax(P, axis=1)
    lib["game_week"] = lib["W"] + 1
    keep = ["season", "game_week", "gsis", "pos", "games", "p0", "p1", "p2", "style", "pmax", "tgt_sh", "car_sh", "ppr_g", "snap", "share"]
    return lib[keep].rename(columns={"games": "g_prior", "snap": "snap_prior", "tgt_sh": "tgt_sh_prior", "car_sh": "car_sh_prior", "ppr_g": "ppr_g_prior"})


def build() -> pd.DataFrame:
    r = vs.rows()
    w = pd.concat([pd.read_csv(rd.cache_dir() / str(y) / "roles_weekly.csv").assign(season=y) if "season" not in pd.read_csv(rd.cache_dir() / str(y) / "roles_weekly.csv", nrows=1)
                   else pd.read_csv(rd.cache_dir() / str(y) / "roles_weekly.csv") for y in SEAS], ignore_index=True)
    w = w[w["pos"].isin(POS)].copy()
    w = w.merge(r[["season", "week", "espn_id", "espn_proj"]], on=["season", "week", "espn_id"], how="left")
    S = style_table()
    w = w.merge(S, left_on=["season", "week", "gsis", "pos"], right_on=["season", "game_week", "gsis", "pos"], how="inner")
    R = mm.weekly_ratings(mm.plays(), "def").rename(columns={"team": "opp"})
    w = w.merge(R[["season", "week", "opp", "pass_r", "run_r"]], on=["season", "week", "opp"], how="left")
    D = defense_games()
    w = w.merge(D, on=["season", "week", "opp"], how="left")
    w["tgt_share"] = np.where(w["t_tgt"] > 0, w["tgt"] / w["t_tgt"], np.nan)
    w["car_share"] = np.where(w["t_car"] > 0, w["car"] / w["t_car"], np.nan)
    w["resid"] = w["ppr"] - w["espn_proj"]
    w["d_tgt_sh"] = w["tgt_share"] - w["tgt_sh_prior"]
    w["d_car_sh"] = w["car_share"] - w["car_sh_prior"]
    w["d_ppr"] = w["ppr"] - w["ppr_g_prior"]
    return w


# ───────────────────────────── step 2: analyze ─────────────────────────────
import math

PRE = ["pass_r", "run_r", "man_prior", "blitz_td", "pressure_prior"]   # known before kickoff
ACT = ["man_g", "blitz_g", "c3_g", "c2_g", "c46_g"]                      # what the defense did in the game (after the fact)
OUTS = {"QB": ["resid", "fe_ppr"], "RB": ["resid", "fe_ppr", "d_tgt_sh", "d_car_sh"], "WR": ["resid", "fe_ppr", "d_tgt_sh"], "TE": ["resid", "fe_ppr", "d_tgt_sh"]}
NICE = {"pass_r": "Pass-defense rating", "run_r": "Run-defense rating", "man_prior": "Man-coverage rate (last year)", "blitz_td": "Blitz rate (to date)",
        "pressure_prior": "Pressure rate (last year)", "man_g": "Man coverage that game", "blitz_g": "Blitz rate that game", "c3_g": "Cover-3 share that game",
        "c2_g": "Cover-2 share that game", "c46_g": "Cover-4/6 share that game"}
B_BOOT, B_PLACEBO = 400, 300


def chi2_sf(x: float, df: int) -> float:
    if df == 1:
        return math.erfc(math.sqrt(max(x, 0.0) / 2.0))
    if df == 2:
        return math.exp(-max(x, 0.0) / 2.0)
    raise ValueError(df)


def add_z(w: pd.DataFrame) -> pd.DataFrame:
    """Standardise defense features: pre-game ones centred within season-week (so season drift is out), actuals within season;
    divide by the pooled sd. Coverage coding levels move a lot between seasons (league man share 42% -> 49% -> 30%), so only
    within-season relative values are used."""
    D = w.drop_duplicates(["season", "week", "opp"])[["season", "week", "opp"] + PRE + ACT].copy()
    Z = D[["season", "week", "opp"]].copy()
    for f in PRE:
        grp = ["season"] if f in ("man_prior", "pressure_prior") else ["season", "week"]
        c = D[f] - D.groupby(grp)[f].transform("mean")
        Z["z_" + f] = c / c.std()
    for f in ACT:
        c = D[f] - D.groupby("season")[f].transform("mean")
        Z["z_" + f] = c / c.std()
    return w.merge(Z, on=["season", "week", "opp"], how="left")


def _starts(cl: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.r_[True, cl[1:] != cl[:-1]])


def _dm(A: np.ndarray, fs: np.ndarray, cnt: np.ndarray) -> np.ndarray:
    mu = np.add.reduceat(A, fs, axis=0) / (cnt.reshape(-1, 1) if A.ndim == 2 else cnt)
    return A - np.repeat(mu, cnt, axis=0)


def fit(P: np.ndarray, x: np.ndarray, y: np.ndarray, cl: np.ndarray, want_v: bool = False, fe: np.ndarray | None = None):
    """y = sum_s p_s*(a_s + b_s*x); player-clustered sandwich.  With `fe` (contiguous player-season codes) everything is demeaned within
    player-season first, so each player is compared with himself.  Returns slopes, omnibus chi2 (equal slopes) and the common-slope fit."""
    n = len(y)
    X = P * x[:, None] if fe is not None else np.column_stack([P, P * x[:, None]])
    X0 = x[:, None] if fe is not None else np.column_stack([P, x])
    if fe is not None:
        fs = _starts(fe)
        cnt = np.diff(np.r_[fs, n])
        X, X0, y = _dm(X, fs, cnt), _dm(X0, fs, cnt), _dm(y, fs, cnt)
    k = X.shape[1]
    ns = k - 3
    XtX = X.T @ X
    try:
        inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return None
    beta = inv @ (X.T @ y)
    e = y - X @ beta
    st = _starts(cl)
    S = np.add.reduceat(X * e[:, None], st, axis=0)
    G = len(st)
    if G < 8:
        return None
    V = inv @ (S.T @ S) @ inv * (G / (G - 1)) * ((n - 1) / (n - k))
    R = np.zeros((2, k))
    R[0, ns], R[0, ns + 1], R[1, ns + 1], R[1, ns + 2] = 1, -1, 1, -1
    d = R @ beta
    try:
        chi2 = float(d @ np.linalg.solve(R @ V @ R.T, d))
    except np.linalg.LinAlgError:
        return None
    out = {"chi2": chi2, "n": n, "G": G}
    if want_v:
        out.update(b=beta[ns:], se=np.sqrt(np.diag(V)[ns:]), V=V)
        k0 = X0.shape[1]
        i0 = np.linalg.inv(X0.T @ X0)
        b0 = i0 @ (X0.T @ y)
        e0 = y - X0 @ b0
        S0 = np.add.reduceat(X0 * e0[:, None], st, axis=0)
        V0 = i0 @ (S0.T @ S0) @ i0 * (G / (G - 1)) * ((n - 1) / (n - k0))
        out.update(b0=float(b0[-1]), se0=float(math.sqrt(V0[-1, -1])))
        se_pairs = [math.sqrt(max(V[ns + i, ns + i] + V[ns + j, ns + j] - 2 * V[ns + i, ns + j], 0)) for i, j in ((0, 1), (0, 2), (1, 2))]
        out["mde"] = 2.8 * float(np.median(se_pairs))
    return out


def bh(p: np.ndarray) -> np.ndarray:
    m = len(p)
    o = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for rank, i in zip(range(m, 0, -1), o[::-1]):
        prev = min(prev, p[i] * m / rank)
        q[i] = prev
    return q


def tests_for(pos: str):
    feats = [f for f in PRE if not (f == "run_r" and pos != "RB")]
    for o in OUTS[pos]:
        for f in feats:
            yield o, f, "A"
        for f in ACT:
            yield o, f, "B"


def terc_stats(P, x, y, cl, rng, B=B_BOOT):
    """Soft-weighted mean outcome per style in the softest vs toughest third, with a player-cluster bootstrap on the gap."""
    lo, hi = np.quantile(x, [1 / 3, 2 / 3])
    T = [(x <= lo), (x >= hi)]                      # toughest (low rating / low feature) , softest (high)
    st = _starts(cl)
    cols = []
    for t in T:
        for s in range(3):
            w_ = P[:, s] * t
            cols += [w_ * y, w_]
    M = np.add.reduceat(np.column_stack(cols), st, axis=0)  # per-cluster sums: [lo s0 num, den, s1..., hi ...]
    G = len(st)

    def gap(Msum):
        r = []
        for s in range(3):
            lo_n, lo_d = Msum[2 * s], Msum[2 * s + 1]
            hi_n, hi_d = Msum[6 + 2 * s], Msum[6 + 2 * s + 1]
            r.append((hi_n / hi_d - lo_n / lo_d) if lo_d > 0 and hi_d > 0 else np.nan)
        return np.array(r)

    tot = M.sum(0)
    g0 = gap(tot)
    lo_m = np.array([tot[2 * s] / tot[2 * s + 1] if tot[2 * s + 1] > 0 else np.nan for s in range(3)])
    hi_m = np.array([tot[6 + 2 * s] / tot[6 + 2 * s + 1] if tot[6 + 2 * s + 1] > 0 else np.nan for s in range(3)])
    den_lo = np.array([tot[2 * s + 1] for s in range(3)])
    den_hi = np.array([tot[6 + 2 * s + 1] for s in range(3)])
    bs = np.empty((B, 3))
    for b in range(B):
        idx = rng.integers(0, G, G)
        bs[b] = gap(M[idx].sum(0))
    ci = np.nanquantile(bs, [0.025, 0.975], axis=0)
    return {"lo_mean": lo_m, "hi_mean": hi_m, "gap": g0, "gap_lo": ci[0], "gap_hi": ci[1], "n_lo": den_lo, "n_hi": den_hi}


def perm_P(P: np.ndarray, st: np.ndarray, n_cl: np.ndarray, rank: np.ndarray, cl_of_row: np.ndarray, rng) -> np.ndarray:
    """Placebo: every player gets another player's style trajectory (rank-matched), so style carries no information about him."""
    G = len(st)
    d = rng.integers(0, G - 1, G)
    d = d + (d >= np.arange(G))                                  # donor != self
    src = st[d[cl_of_row]] + rank % n_cl[d[cl_of_row]]
    return P[src]


def analyze() -> None:
    rng = np.random.default_rng(7)
    w = add_z(pd.read_csv(OUT / "games.csv.gz"))
    rows = []
    store_data = {}
    for pos in POS:
        d = w[w["pos"] == pos].dropna(subset=["p0", "p1", "p2"]).sort_values(["gsis", "season", "week"]).reset_index(drop=True)
        names = roles.load_model()["style"][pos]["names"]
        P = d[["p0", "p1", "p2"]].to_numpy(float)
        cl_all = d["gsis"].astype("category").cat.codes.to_numpy()
        d["fe_ppr"] = d["ppr"]
        d["ps_code"] = (d["gsis"] + d["season"].astype(str)).astype("category").cat.codes
        d["n_ps"] = d.groupby("ps_code")["ppr"].transform("size")
        st_all = _starts(cl_all)
        n_cl = np.diff(np.r_[st_all, len(d)])
        cl_of_row = np.repeat(np.arange(len(st_all)), n_cl)
        rank = np.arange(len(d)) - st_all[cl_of_row]
        store_data[pos] = (d, P, cl_all, st_all, n_cl, cl_of_row, rank, names)
        for o, f, tier in tests_for(pos):
            m = d[o].notna() & d["z_" + f].notna()
            if o == "fe_ppr":
                m &= d["n_ps"] >= 4
            sub = d[m]
            if len(sub) < 200:
                continue
            Pm, x, y, cl = P[m.to_numpy()], sub["z_" + f].to_numpy(float), sub[o].to_numpy(float), cl_all[m.to_numpy()]
            fe = sub["ps_code"].to_numpy() if o == "fe_ppr" else None
            r = fit(Pm, x, y, cl, want_v=True, fe=fe)
            if r is None:
                continue
            yd = (sub["ppr"] - sub.groupby("ps_code")["ppr"].transform("mean")).to_numpy(float) if o == "fe_ppr" else y
            ts = terc_stats(Pm, x, y if fe is None else yd, cl, rng)
            # out-of-time halves
            halves = {}
            for lab, sel in (("A", sub["season"].isin([2023, 2024]).to_numpy()), ("B", (sub["season"] == 2025).to_numpy())):
                if sel.sum() < 150:
                    halves[lab] = None
                    continue
                halves[lab] = fit(Pm[sel], x[sel], y[sel], cl[sel], want_v=True, fe=None if fe is None else fe[sel])
            row = dict(pos=pos, outcome=o, feature=f, tier=tier, n=r["n"], G=r["G"], chi2=r["chi2"], p_wald=chi2_sf(r["chi2"], 2), pooled=r["b0"], pooled_se=r["se0"], mde=r["mde"])
            for s in range(3):
                row.update({f"b{s}": r["b"][s], f"se{s}": r["se"][s], f"lo{s}": ts["lo_mean"][s], f"hi{s}": ts["hi_mean"][s], f"gap{s}": ts["gap"][s], f"gap_lo{s}": ts["gap_lo"][s], f"gap_hi{s}": ts["gap_hi"][s],
                            f"n_lo{s}": ts["n_lo"][s], f"n_hi{s}": ts["n_hi"][s], f"style{s}": names[s]})
            for lab in ("A", "B"):
                h = halves[lab]
                for s in range(3):
                    row[f"{lab}_b{s}"] = h["b"][s] if h else np.nan
                    row[f"{lab}_se{s}"] = h["se"][s] if h else np.nan
                row[f"{lab}_pooled"] = h["b0"] if h else np.nan
            rows.append(row)
    R = pd.DataFrame(rows)
    for t in ("A", "B"):
        m = R["tier"] == t
        R.loc[m, "q_wald"] = bh(R.loc[m, "p_wald"].to_numpy())
    print(f"tests: {len(R)} (tier A {int((R.tier == 'A').sum())}, tier B {int((R.tier == 'B').sum())})", flush=True)

    # placebo: fake style trajectories, same tests
    chi_pl = np.full((B_PLACEBO, len(R)), np.nan)
    for b in range(B_PLACEBO):
        permP = {}
        for pos in POS:
            d, P, cl_all, st_all, n_cl, cl_of_row, rank, names = store_data[pos]
            permP[pos] = perm_P(P, st_all, n_cl, rank, cl_of_row, rng)
        for j, r in enumerate(R.itertuples(index=False)):
            d, P, cl_all, *_ = store_data[r.pos]
            mm_ = d[r.outcome].notna() & d["z_" + r.feature].notna()
            if r.outcome == "fe_ppr":
                mm_ &= d["n_ps"] >= 4
            m = mm_.to_numpy()
            out = fit(permP[r.pos][m], d.loc[m, "z_" + r.feature].to_numpy(float), d.loc[m, r.outcome].to_numpy(float), cl_all[m], fe=d.loc[m, "ps_code"].to_numpy() if r.outcome == "fe_ppr" else None)
            if out:
                chi_pl[b, j] = out["chi2"]
        if b % 50 == 0:
            print("placebo", b, flush=True)
    R["p_placebo"] = [(np.nansum(chi_pl[:, j] >= R["chi2"].iloc[j]) + 1) / (np.isfinite(chi_pl[:, j]).sum() + 1) for j in range(len(R))]
    R["fp_rate_wald05"] = [np.nanmean(np.array([chi2_sf(c, 2) for c in chi_pl[:, j] if np.isfinite(c)]) < 0.05) for j in range(len(R))]
    # family-wise: how often does the best (smallest p) test in a family of this size look at least this good by chance?
    for t in ("A", "B"):
        idx = np.flatnonzero((R["tier"] == t).to_numpy())
        minp_pl = np.array([min(chi2_sf(c, 2) for c in chi_pl[b, idx] if np.isfinite(c)) for b in range(B_PLACEBO)])
        R.attrs[f"minp_{t}"] = float(R.loc[R.index[idx], "p_wald"].min())
        R.attrs[f"fw_{t}"] = float((np.sum(minp_pl <= R.attrs[f"minp_{t}"]) + 1) / (B_PLACEBO + 1))
        R.attrs[f"fpr_{t}"] = float(np.nanmean(R.loc[R.index[idx], "fp_rate_wald05"]))
    R.to_csv(OUT / "results.csv", index=False)
    np.save(OUT / "placebo_chi2.npy", chi_pl)
    summarize(R)


def summarize(R: pd.DataFrame | None = None) -> None:
    if R is None:
        R = pd.read_csv(OUT / "results.csv")
    pd.set_option("display.width", 250, "display.max_columns", 50, "display.max_rows", 500)
    print("\n== calibration ==")
    for t in ("A", "B"):
        m = R["tier"] == t
        print(f"tier {t}: {int(m.sum())} tests; omnibus p<0.05: {int((R.loc[m,'p_wald']<0.05).sum())} (chance ~{0.05*m.sum():.1f}); q<0.10: {int((R.loc[m,'q_wald']<0.10).sum())}; "
              f"placebo p<0.05: {int((R.loc[m,'p_placebo']<0.05).sum())}; mean placebo false-positive rate of the Wald test {R.loc[m,'fp_rate_wald05'].mean():.3f}")
    print("\n== smallest omnibus p per tier ==")
    print(R.sort_values("p_wald").groupby("tier").head(6)[["pos", "outcome", "feature", "tier", "n", "G", "chi2", "p_wald", "q_wald", "p_placebo"]].round(4).to_string(index=False))
    print("\n== detectable style gap (80% power), median over tests ==")
    print(R.groupby(["pos", "outcome"])["mde"].median().round(3).to_string())
    print("\n== pooled (all-style) slope per +1 sd, tier A ==")
    print(R[R.tier == "A"][["pos", "outcome", "feature", "pooled", "pooled_se"]].round(3).to_string(index=False))
    # out-of-time
    dev = []
    for r in R.itertuples(index=False):
        for s in range(3):
            a, b = getattr(r, f"A_b{s}"), getattr(r, f"B_b{s}")
            sa = getattr(r, f"A_se{s}")
            if np.isfinite(a) and np.isfinite(b) and np.isfinite(r.A_pooled) and np.isfinite(r.B_pooled):
                dev.append((r.tier, a - r.A_pooled, b - r.B_pooled, sa))
    dv = pd.DataFrame(dev, columns=["tier", "dA", "dB", "seA"])
    print("\n== out-of-time (2023-24 vs 2025): style-slope deviations from the position-wide slope ==")
    for t, g in dv.groupby("tier"):
        big = g[np.abs(g.dA) > g.seA]
        print(f"tier {t}: n={len(g)} corr(dA,dB)={g.dA.corr(g.dB):.2f}; |dA|>1se: n={len(big)}, sign repeats in 2025: {(np.sign(big.dA)==np.sign(big.dB)).mean():.2f}")


def tier_groups(d: pd.DataFrame, pos: str) -> tuple[np.ndarray, list[str]]:
    """Volume tier (as of the prior week) collapsed to three groups, one-hot."""
    if pos == "RB":
        c = d["car_sh_prior"].fillna(0)
        g = np.where(c >= .60, 0, np.where(c >= .40, 1, 2))
        names = ["Workhorse / Lead back", "Split backfield", "Change of pace / Depth"]
    elif pos == "WR":
        t = d["tgt_sh_prior"].fillna(0)
        g = np.where(t >= .25, 0, np.where(t >= .17, 1, 2))
        names = ["Alpha target share", "Starter", "Low volume"]
    else:
        t, s = d["tgt_sh_prior"].fillna(0), d["snap_prior"].fillna(0)
        g = np.where(t >= .17, 0, np.where(s >= 60, 1, 2))
        names = ["Featured", "Starter", "Rotational"]
    return np.eye(3)[g], names


def analyze_tiers() -> None:
    """Same tests, but grouping players by the volume tier (Workhorse, Alpha target share ...) instead of the style cluster."""
    rng = np.random.default_rng(13)
    w = add_z(pd.read_csv(OUT / "games.csv.gz"))
    rows, data = [], {}
    for pos in ("RB", "WR", "TE"):
        d = w[w["pos"] == pos].sort_values(["gsis", "season", "week"]).reset_index(drop=True)
        P, names = tier_groups(d, pos)
        d["fe_ppr"] = d["ppr"]
        d["ps_code"] = (d["gsis"] + d["season"].astype(str)).astype("category").cat.codes
        d["n_ps"] = d.groupby("ps_code")["ppr"].transform("size")
        cl_all = d["gsis"].astype("category").cat.codes.to_numpy()
        st_all = _starts(cl_all)
        n_cl = np.diff(np.r_[st_all, len(d)])
        cor = np.repeat(np.arange(len(st_all)), n_cl)
        data[pos] = (d, P, cl_all, st_all, n_cl, cor, np.arange(len(d)) - st_all[cor], names)
        for o, f, tier in tests_for(pos):
            if tier == "B":
                continue
            m = d[o].notna() & d["z_" + f].notna()
            if o == "fe_ppr":
                m &= d["n_ps"] >= 4
            m = m.to_numpy()
            fe = d.loc[m, "ps_code"].to_numpy() if o == "fe_ppr" else None
            x, y = d.loc[m, "z_" + f].to_numpy(float), d.loc[m, o].to_numpy(float)
            r = fit(P[m], x, y, cl_all[m], want_v=True, fe=fe)
            if r is None:
                continue
            row = dict(pos=pos, outcome=o, feature=f, n=r["n"], G=r["G"], chi2=r["chi2"], p_wald=chi2_sf(r["chi2"], 2), pooled=r["b0"], pooled_se=r["se0"], mde=r["mde"])
            for s in range(3):
                row.update({f"tier{s}": names[s], f"b{s}": r["b"][s], f"se{s}": r["se"][s], f"n{s}": float(P[m][:, s].sum())})
            rows.append(row)
    R = pd.DataFrame(rows)
    R["q_wald"] = bh(R["p_wald"].to_numpy())
    chi = np.full((B_PLACEBO, len(R)), np.nan)
    for b in range(B_PLACEBO):
        perm = {pos: perm_P(data[pos][1], data[pos][3], data[pos][4], data[pos][6], data[pos][5], rng) for pos in data}
        for j, r in enumerate(R.itertuples(index=False)):
            d, P, cl_all = data[r.pos][:3]
            mm_ = d[r.outcome].notna() & d["z_" + r.feature].notna()
            if r.outcome == "fe_ppr":
                mm_ &= d["n_ps"] >= 4
            m = mm_.to_numpy()
            out = fit(perm[r.pos][m], d.loc[m, "z_" + r.feature].to_numpy(float), d.loc[m, r.outcome].to_numpy(float), cl_all[m],
                      fe=d.loc[m, "ps_code"].to_numpy() if r.outcome == "fe_ppr" else None)
            if out:
                chi[b, j] = out["chi2"]
    R["p_placebo"] = [(np.nansum(chi[:, j] >= R["chi2"].iloc[j]) + 1) / (np.isfinite(chi[:, j]).sum() + 1) for j in range(len(R))]
    minp = np.array([min(chi2_sf(c, 2) for c in chi[b] if np.isfinite(c)) for b in range(B_PLACEBO)])
    np.save(OUT / "placebo_chi2_tier.npy", chi)
    R.to_csv(OUT / "results_tier.csv", index=False)
    print(f"volume-tier tests: {len(R)}; omnibus p<0.05: {(R.p_wald < 0.05).sum()} (chance ~{0.05 * len(R):.1f}); placebo p<0.05: {(R.p_placebo < 0.05).sum()}; q<0.10: {(R.q_wald < 0.10).sum()}")
    print(f"smallest p {R.p_wald.min():.4f}; matched or beaten by chance in {np.mean(minp <= R.p_wald.min()):.0%} of fake-label runs")
    print(R.sort_values("p_wald").head(8)[["pos", "outcome", "feature", "n", "G", "chi2", "p_wald", "q_wald", "p_placebo"]].round(4).to_string(index=False))
    print(R.groupby(["pos", "outcome"])["mde"].median().round(3).to_string())


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        w = build()
        w.to_csv(OUT / "games.csv.gz", index=False, compression="gzip")
        print("rows", len(w), "by pos", w["pos"].value_counts().to_dict())
        print("with espn", w["espn_proj"].notna().sum(), "with pass_r", w["pass_r"].notna().sum(), "with man_g", w["man_g"].notna().sum(), "with man_prior", w["man_prior"].notna().sum())
        print(w.groupby(["pos", "style"]).size())
    elif cmd == "analyze":
        analyze()
    elif cmd == "summary":
        summarize()
    elif cmd == "tiers":
        analyze_tiers()
