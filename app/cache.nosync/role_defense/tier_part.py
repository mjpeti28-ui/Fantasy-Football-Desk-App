

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
    R.to_csv(OUT / "results_tier.csv", index=False)
    print(f"volume-tier tests: {len(R)}; omnibus p<0.05: {(R.p_wald < 0.05).sum()} (chance ~{0.05 * len(R):.1f}); placebo p<0.05: {(R.p_placebo < 0.05).sum()}; q<0.10: {(R.q_wald < 0.10).sum()}")
    print(f"smallest p {R.p_wald.min():.4f}; matched or beaten by chance in {np.mean(minp <= R.p_wald.min()):.0%} of fake-label runs")
    print(R.sort_values("p_wald").head(8)[["pos", "outcome", "feature", "n", "G", "chi2", "p_wald", "q_wald", "p_placebo"]].round(4).to_string(index=False))
    print(R.groupby(["pos", "outcome"])["mde"].median().round(3).to_string())
