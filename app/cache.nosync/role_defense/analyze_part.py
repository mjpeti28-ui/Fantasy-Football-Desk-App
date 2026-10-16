

# ───────────────────────────── step 2: analyze ─────────────────────────────
import math

PRE = ["pass_r", "run_r", "man_prior", "blitz_td", "pressure_prior"]   # known before kickoff
ACT = ["man_g", "blitz_g", "c3_g", "c2_g", "c46_g"]                      # what the defense did in the game (after the fact)
OUTS = {"QB": ["resid"], "RB": ["resid", "d_tgt_sh", "d_car_sh"], "WR": ["resid", "d_tgt_sh"], "TE": ["resid", "d_tgt_sh"]}
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


def fit(P: np.ndarray, x: np.ndarray, y: np.ndarray, cl: np.ndarray, want_v: bool = False):
    """y = sum_s p_s*(a_s + b_s*x); player-clustered sandwich.  Returns slopes, omnibus chi2 (equal slopes), pooled-slope fit."""
    n = len(y)
    X = np.column_stack([P, P * x[:, None]])
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
    k = X.shape[1]
    V = inv @ (S.T @ S) @ inv * (G / (G - 1)) * ((n - 1) / (n - k))
    R = np.array([[0, 0, 0, 1, -1, 0], [0, 0, 0, 0, 1, -1]], float)
    d = R @ beta
    try:
        chi2 = float(d @ np.linalg.solve(R @ V @ R.T, d))
    except np.linalg.LinAlgError:
        return None
    out = {"chi2": chi2, "n": n, "G": G}
    if want_v:
        out.update(b=beta[3:], se=np.sqrt(np.diag(V)[3:]), a=beta[:3], V=V)
        # common slope (style intercepts + one slope)
        X0 = np.column_stack([P, x])
        i0 = np.linalg.inv(X0.T @ X0)
        b0 = i0 @ (X0.T @ y)
        e0 = y - X0 @ b0
        S0 = np.add.reduceat(X0 * e0[:, None], st, axis=0)
        V0 = i0 @ (S0.T @ S0) @ i0 * (G / (G - 1)) * ((n - 1) / (n - 4))
        out.update(b0=float(b0[3]), se0=float(math.sqrt(V0[3, 3])))
        # smallest detectable gap between two style slopes at 80% power (2.8 se), median over pairs
        se_pairs = [math.sqrt(max(V[3 + i, 3 + i] + V[3 + j, 3 + j] - 2 * V[3 + i, 3 + j], 0)) for i, j in ((0, 1), (0, 2), (1, 2))]
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
        st_all = _starts(cl_all)
        n_cl = np.diff(np.r_[st_all, len(d)])
        cl_of_row = np.repeat(np.arange(len(st_all)), n_cl)
        rank = np.arange(len(d)) - st_all[cl_of_row]
        store_data[pos] = (d, P, cl_all, st_all, n_cl, cl_of_row, rank, names)
        for o, f, tier in tests_for(pos):
            m = d[o].notna() & d["z_" + f].notna()
            sub = d[m]
            if len(sub) < 200:
                continue
            Pm, x, y, cl = P[m.to_numpy()], sub["z_" + f].to_numpy(float), sub[o].to_numpy(float), cl_all[m.to_numpy()]
            r = fit(Pm, x, y, cl, want_v=True)
            if r is None:
                continue
            ts = terc_stats(Pm, x, y, cl, rng)
            # out-of-time halves
            halves = {}
            for lab, sel in (("A", sub["season"].isin([2023, 2024]).to_numpy()), ("B", (sub["season"] == 2025).to_numpy())):
                if sel.sum() < 150:
                    halves[lab] = None
                    continue
                halves[lab] = fit(Pm[sel], x[sel], y[sel], cl[sel], want_v=True)
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
            m = (d[r.outcome].notna() & d["z_" + r.feature].notna()).to_numpy()
            out = fit(permP[r.pos][m], d.loc[m, "z_" + r.feature].to_numpy(float), d.loc[m, r.outcome].to_numpy(float), cl_all[m])
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
