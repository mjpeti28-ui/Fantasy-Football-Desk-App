import re, sys
p = sys.argv[1]
src = open(p).read()

# 1. new fit()
i = src.index("def fit(")
j = src.index("def bh(")
new_fit = '''def _dm(A: np.ndarray, fs: np.ndarray, cnt: np.ndarray) -> np.ndarray:
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


'''
src = src[:i] + new_fit + src[j:]

# 2. outcomes
src = src.replace('OUTS = {"QB": ["resid"], "RB": ["resid", "d_tgt_sh", "d_car_sh"], "WR": ["resid", "d_tgt_sh"], "TE": ["resid", "d_tgt_sh"]}',
                  'OUTS = {"QB": ["resid", "fe_ppr"], "RB": ["resid", "fe_ppr", "d_tgt_sh", "d_car_sh"], "WR": ["resid", "fe_ppr", "d_tgt_sh"], "TE": ["resid", "fe_ppr", "d_tgt_sh"]}')

# 3. analyze(): per-position setup
old = '        cl_all = d["gsis"].astype("category").cat.codes.to_numpy()\n'
new = ('        cl_all = d["gsis"].astype("category").cat.codes.to_numpy()\n'
       '        d["fe_ppr"] = d["ppr"]\n'
       '        d["ps_code"] = (d["gsis"] + d["season"].astype(str)).astype("category").cat.codes\n'
       '        d["n_ps"] = d.groupby("ps_code")["ppr"].transform("size")\n')
assert old in src; src = src.replace(old, new, 1)

old = '            m = d[o].notna() & d["z_" + f].notna()\n            sub = d[m]\n'
new = ('            m = d[o].notna() & d["z_" + f].notna()\n            if o == "fe_ppr":\n                m &= d["n_ps"] >= 4\n            sub = d[m]\n')
assert old in src; src = src.replace(old, new, 1)

old = '            r = fit(Pm, x, y, cl, want_v=True)\n            if r is None:\n                continue\n            ts = terc_stats(Pm, x, y, cl, rng)\n'
new = ('            fe = sub["ps_code"].to_numpy() if o == "fe_ppr" else None\n'
       '            r = fit(Pm, x, y, cl, want_v=True, fe=fe)\n            if r is None:\n                continue\n'
       '            yd = (sub["ppr"] - sub.groupby("ps_code")["ppr"].transform("mean")).to_numpy(float) if o == "fe_ppr" else y\n'
       '            ts = terc_stats(Pm, x, y if fe is None else yd, cl, rng)\n')
assert old in src; src = src.replace(old, new, 1)

old = '                halves[lab] = fit(Pm[sel], x[sel], y[sel], cl[sel], want_v=True)\n'
new = '                halves[lab] = fit(Pm[sel], x[sel], y[sel], cl[sel], want_v=True, fe=None if fe is None else fe[sel])\n'
assert old in src; src = src.replace(old, new, 1)

old = ('            m = (d[r.outcome].notna() & d["z_" + r.feature].notna()).to_numpy()\n'
       '            out = fit(permP[r.pos][m], d.loc[m, "z_" + r.feature].to_numpy(float), d.loc[m, r.outcome].to_numpy(float), cl_all[m])\n')
new = ('            mm_ = d[r.outcome].notna() & d["z_" + r.feature].notna()\n'
       '            if r.outcome == "fe_ppr":\n                mm_ &= d["n_ps"] >= 4\n            m = mm_.to_numpy()\n'
       '            out = fit(permP[r.pos][m], d.loc[m, "z_" + r.feature].to_numpy(float), d.loc[m, r.outcome].to_numpy(float), cl_all[m], fe=d.loc[m, "ps_code"].to_numpy() if r.outcome == "fe_ppr" else None)\n')
assert old in src; src = src.replace(old, new, 1)
open(p, "w").write(src)
print("patched")
