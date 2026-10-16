"""Holdout for the style x pass/run-defense tests: 2016-22.  Only EPA-based ratings reach back that far (no coverage charting before 2022),
so this checks the pass-defense / run-defense rating results found on 2023-25 (role_defense_split.py) on seven seasons that were not used to find them.

  python3 -m app.engine.roles_data 2016 2017 2018 2019 2020 2021 2022   # role tables for the holdout seasons (once)
  python3 backtest/role_defense_holdout.py          # 2016-22 holdout of the 2023-25 pass/run-defense results (downloads lean play-by-play once)
  python3 backtest/role_defense_holdout.py pooled   # all ten seasons: style and volume-tier tests, fake-label placebo, three eras
  python3 backtest/role_defense_holdout.py espn5    # points minus ESPN's projection, 2021-25
  python3 backtest/role_defense_holdout.py blitz22  # the RB x blitz-rate watch item on 2022 (the only earlier season with blitzer counts)
  python3 backtest/role_defense_holdout.py export   # writes style_run + tier_credit into app/engine/matchups_credit.json (the card quotes them)
Free data only (nflverse); no PFF, no per-defender stats.  Outcomes: points within the player's own season, target / carry share vs his prior-week share.
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
from app.engine import roles_data as rd  # noqa: E402
from backtest import matchup_mechanisms as mm  # noqa: E402
from backtest import role_defense_split as rds  # noqa: E402

H = list(range(2016, 2023))
OUT = rds.OUT
CREDIT = ROOT / "app" / "engine" / "matchups_credit.json"
LEAN = ["game_id", "season", "week", "season_type", "posteam", "defteam", "play_type", "qb_dropback", "rush_attempt", "two_point_attempt", "epa"]


def lean_plays(y: int) -> pd.DataFrame:
    p = OUT / f"pbp_lean_{y}.csv.gz"
    if p.exists():
        return pd.read_csv(p)
    b = rd._csv(f"{rd.BASE}/pbp/play_by_play_{y}.csv.gz", LEAN, gz=True)
    b = b[(b["season_type"] == "REG") & b["play_type"].isin(["pass", "run"]) & (b["two_point_attempt"].fillna(0) != 1)].copy()
    b["db"] = (b["qb_dropback"] == 1).astype(int)
    b["run"] = ((b["rush_attempt"] == 1) & (b["qb_dropback"] != 1)).astype(int)
    b = b[(b["db"] == 1) | (b["run"] == 1)][mm.COLS]
    b.to_csv(p, index=False, compression="gzip")
    return b


def build_h() -> pd.DataFrame:
    rds.SEAS = H
    S = rds.style_table()
    frames = []
    for y in H:
        f = pd.read_csv(rd.cache_dir() / str(y) / "roles_weekly.csv")
        if "season" not in f:
            f["season"] = y
        frames.append(f)
    w = pd.concat(frames, ignore_index=True)
    w = w[w["pos"].isin(rds.POS)].copy()
    w = w.merge(S, left_on=["season", "week", "gsis", "pos"], right_on=["season", "game_week", "gsis", "pos"], how="inner")
    P = pd.concat([lean_plays(y) for y in [H[0] - 1] + H], ignore_index=True)
    P = P[P["week"] <= 18]
    old = mm.SEAS
    mm.SEAS = H
    try:
        R = mm.weekly_ratings(P, "def").rename(columns={"team": "opp"})
    finally:
        mm.SEAS = old
    w = w.merge(R[["season", "week", "opp", "pass_r", "run_r"]], on=["season", "week", "opp"], how="left")
    w["tgt_share"] = np.where(w["t_tgt"] > 0, w["tgt"] / w["t_tgt"], np.nan)
    w["car_share"] = np.where(w["t_car"] > 0, w["car"] / w["t_car"], np.nan)
    w["d_tgt_sh"] = w["tgt_share"] - w["tgt_sh_prior"]
    w["d_car_sh"] = w["car_share"] - w["car_sh_prior"]
    w["fe_ppr"] = w["ppr"]
    D = w.drop_duplicates(["season", "week", "opp"])[["season", "week", "opp", "pass_r", "run_r"]]
    Z = D[["season", "week", "opp"]].copy()
    for f in ("pass_r", "run_r"):
        c = D[f] - D.groupby(["season", "week"])[f].transform("mean")
        Z["z_" + f] = c / c.std()
    return w.merge(Z, on=["season", "week", "opp"], how="left")


def contrast(out: dict, c: np.ndarray) -> tuple[float, float, float]:
    k = len(out["b"])
    V = out["V"][-k:, -k:]
    est = float(c @ out["b"])
    se = float(np.sqrt(c @ V @ c))
    return est, se, est / se


def run(w: pd.DataFrame, label: str, ncheck: bool = True) -> dict:
    """Omnibus per position/outcome/rating, plus the contrasts that the 2023-25 data suggested (fixed in advance of this run)."""
    res = {}
    for pos in rds.POS:
        d = w[w["pos"] == pos].dropna(subset=["p0", "p1", "p2"]).sort_values(["gsis", "season", "week"]).reset_index(drop=True)
        d["ps"] = (d["gsis"] + d["season"].astype(str)).astype("category").cat.codes
        d["n_ps"] = d.groupby("ps")["ppr"].transform("size")
        cl = d["gsis"].astype("category").cat.codes.to_numpy()
        P = d[["p0", "p1", "p2"]].to_numpy(float)
        Pt, _ = rds.tier_groups(d, pos) if pos != "QB" else (None, None)
        feats = ["pass_r"] + (["run_r"] if pos == "RB" else [])
        outs = ["fe_ppr", "d_tgt_sh"] + (["d_car_sh"] if pos == "RB" else [])
        for o in outs:
            for f in feats:
                m = (d[o].notna() & d["z_" + f].notna() & ((d["n_ps"] >= 4) if o == "fe_ppr" else True)).to_numpy()
                if m.sum() < 300:
                    continue
                x, y = d.loc[m, "z_" + f].to_numpy(float), d.loc[m, o].to_numpy(float)
                fe = d.loc[m, "ps"].to_numpy() if o == "fe_ppr" else None
                r = rds.fit(P[m], x, y, cl[m], want_v=True, fe=fe)
                if r is None:
                    continue
                res[(pos, o, f, "style")] = dict(n=r["n"], G=r["G"], p=rds.chi2_sf(r["chi2"], 2), pooled=r["b0"], b=r["b"], se=r["se"], out=r)
                if Pt is not None:
                    rt = rds.fit(Pt[m], x, y, cl[m], want_v=True, fe=fe)
                    if rt:
                        res[(pos, o, f, "tier")] = dict(n=rt["n"], G=rt["G"], p=rds.chi2_sf(rt["chi2"], 2), pooled=rt["b0"], b=rt["b"], se=rt["se"], out=rt)
    return res


CONTRASTS = [  # (label, pos, outcome, feature, grouping, weights over the 3 groups, expected sign from 2023-25)
    ("RB all-purpose minus passing-down, pass-defense rating (points)", "RB", "fe_ppr", "pass_r", "style", np.array([1.0, -1.0, 0.0])),
    ("RB early-down runner minus the other two, run-defense rating (points)", "RB", "fe_ppr", "run_r", "style", np.array([-0.5, -0.5, 1.0])),
    ("RB lead+split minus change-of-pace, pass-defense rating (points)", "RB", "fe_ppr", "pass_r", "tier", np.array([0.5, 0.5, -1.0])),
    ("RB lead+split minus change-of-pace, run-defense rating (points)", "RB", "fe_ppr", "run_r", "tier", np.array([0.5, 0.5, -1.0])),
]


def main() -> None:
    w_h = build_h()
    w_h.to_csv(OUT / "games_holdout.csv.gz", index=False, compression="gzip")
    print("holdout rows", len(w_h), w_h.groupby("pos").size().to_dict(), "seasons", sorted(w_h.season.unique()))
    w_d = rds.add_z(pd.read_csv(OUT / "games.csv.gz"))
    w_d["fe_ppr"] = w_d["ppr"]
    r_h, r_d = run(w_h, "holdout"), run(w_d, "discovery")
    print("\n== pre-specified contrasts (2023-25 suggested these) ==")
    for lab, pos, o, f, g, c in CONTRASTS:
        row = []
        for nm, R in (("2023-25", r_d), ("2016-22", r_h)):
            k = (pos, o, f, g)
            if k not in R:
                row.append(f"{nm}: n/a")
                continue
            e, s, z = contrast(R[k]["out"], c)
            row.append(f"{nm}: {e:+.2f} ± {s:.2f} (z {z:+.1f})")
        print(f"{lab}\n    " + "   ".join(row))
    print("\n== omnibus p, style groups, 2016-22 (7 seasons) ==")
    rows = [(k, v["p"], v["n"], v["G"]) for k, v in r_h.items() if k[3] == "style"]
    rows.sort(key=lambda t: t[1])
    for k, p, n, G in rows:
        print(" ", k[:3], f"p={p:.3f}", f"n={n}", f"G={G}")
    print(f"style tests: {len(rows)}; p<.05: {sum(p < .05 for _, p, _, _ in rows)} (chance ~{0.05 * len(rows):.1f}); p<.01: {sum(p < .01 for _, p, _, _ in rows)}")
    rows = [(k, v["p"], v["n"], v["G"]) for k, v in r_h.items() if k[3] == "tier"]
    rows.sort(key=lambda t: t[1])
    print("== omnibus p, volume tiers, 2016-22 ==")
    for k, p, n, G in rows[:6]:
        print(" ", k[:3], f"p={p:.3f}", f"n={n}", f"G={G}")
    print(f"tier tests: {len(rows)}; p<.05: {sum(p < .05 for _, p, _, _ in rows)} (chance ~{0.05 * len(rows):.1f})")
    print("\n== pooled slope per +1 sd of defense softness, 2016-22 vs 2023-25 (points within player) ==")
    for pos in rds.POS:
        for f in ("pass_r", "run_r"):
            k = (pos, "fe_ppr", f, "style")
            if k in r_h and k in r_d:
                print(f"  {pos} {f}: 2016-22 {r_h[k]['pooled']:+.2f}  2023-25 {r_d[k]['pooled']:+.2f}")


# ───────────────────── all ten seasons, pass/run-defense rating only ─────────────────────
ERAS = {"e16_19": (2016, 2019), "e20_22": (2020, 2022), "e23_25": (2023, 2025)}
COMMON = ["season", "week", "gsis", "pos", "p0", "p1", "p2", "ppr", "fe_ppr", "d_tgt_sh", "d_car_sh", "tgt_sh_prior", "car_sh_prior", "snap_prior", "z_pass_r", "z_run_r"]


def pooled(reps: int = 300) -> None:
    rng = np.random.default_rng(21)
    w_h = pd.read_csv(OUT / "games_holdout.csv.gz")
    w_d = rds.add_z(pd.read_csv(OUT / "games.csv.gz"))
    w_d["fe_ppr"] = w_d["ppr"]
    W = pd.concat([w_h[COMMON], w_d[COMMON]], ignore_index=True)
    rows, data = [], {}
    for pos in rds.POS:
        d = W[W["pos"] == pos].dropna(subset=["p0", "p1", "p2"]).sort_values(["gsis", "season", "week"]).reset_index(drop=True)
        d["ps"] = (d["gsis"] + d["season"].astype(str)).astype("category").cat.codes
        d["n_ps"] = d.groupby("ps")["ppr"].transform("size")
        cl = d["gsis"].astype("category").cat.codes.to_numpy()
        st = rds._starts(cl)
        n_cl = np.diff(np.r_[st, len(d)])
        cor = np.repeat(np.arange(len(st)), n_cl)
        rank = np.arange(len(d)) - st[cor]
        Pst = d[["p0", "p1", "p2"]].to_numpy(float)
        Ptr, tnames = rds.tier_groups(d, pos) if pos != "QB" else (None, None)
        snames = rds.roles.load_model()["style"][pos]["names"]
        data[pos] = (d, cl, st, n_cl, cor, rank, {"style": Pst, "tier": Ptr}, {"style": snames, "tier": tnames})
        outs = ["fe_ppr"] if pos == "QB" else ["fe_ppr", "d_tgt_sh"] + (["d_car_sh"] if pos == "RB" else [])
        for o in outs:
            for f in ["pass_r"] + (["run_r"] if pos == "RB" else []):
                for g in ("style", "tier"):
                    P = data[pos][6][g]
                    if P is None:
                        continue
                    m = (d[o].notna() & d["z_" + f].notna() & ((d["n_ps"] >= 4) if o == "fe_ppr" else True)).to_numpy()
                    x, y = d.loc[m, "z_" + f].to_numpy(float), d.loc[m, o].to_numpy(float)
                    fe = d.loc[m, "ps"].to_numpy() if o == "fe_ppr" else None
                    r = rds.fit(P[m], x, y, cl[m], want_v=True, fe=fe)
                    row = dict(pos=pos, outcome=o, feature=f, grouping=g, n=r["n"], G=r["G"], chi2=r["chi2"], p_wald=rds.chi2_sf(r["chi2"], 2), pooled=r["b0"], pooled_se=r["se0"], mde=r["mde"])
                    for s in range(3):
                        row.update({f"name{s}": data[pos][7][g][s], f"b{s}": r["b"][s], f"se{s}": r["se"][s]})
                    for era, (a, b) in ERAS.items():
                        sel = d.loc[m, "season"].between(a, b).to_numpy()
                        h = rds.fit(P[m][sel], x[sel], y[sel], cl[m][sel], want_v=True, fe=None if fe is None else fe[sel])
                        for s in range(3):
                            row[f"{era}_b{s}"] = h["b"][s] if h else np.nan
                            row[f"{era}_se{s}"] = h["se"][s] if h else np.nan
                        row[f"{era}_pooled"] = h["b0"] if h else np.nan
                    rows.append(row)
    R = pd.DataFrame(rows)
    for g in ("style", "tier"):
        mk = R["grouping"] == g
        R.loc[mk, "q_wald"] = rds.bh(R.loc[mk, "p_wald"].to_numpy())
    chi = np.full((reps, len(R)), np.nan)
    for b in range(reps):
        perm = {}
        for pos in rds.POS:
            d, cl, st, n_cl, cor, rank, Ps, _ = data[pos]
            perm[pos] = {g: (rds.perm_P(Ps[g], st, n_cl, rank, cor, rng) if Ps[g] is not None else None) for g in Ps}
        for j, r in enumerate(R.itertuples(index=False)):
            d, cl = data[r.pos][0], data[r.pos][1]
            m = (d[r.outcome].notna() & d["z_" + r.feature].notna() & ((d["n_ps"] >= 4) if r.outcome == "fe_ppr" else True)).to_numpy()
            out = rds.fit(perm[r.pos][r.grouping][m], d.loc[m, "z_" + r.feature].to_numpy(float), d.loc[m, r.outcome].to_numpy(float), cl[m],
                          fe=d.loc[m, "ps"].to_numpy() if r.outcome == "fe_ppr" else None)
            if out:
                chi[b, j] = out["chi2"]
    R["p_placebo"] = [(np.nansum(chi[:, j] >= R["chi2"].iloc[j]) + 1) / (np.isfinite(chi[:, j]).sum() + 1) for j in range(len(R))]
    R.to_csv(OUT / "results_pooled10.csv", index=False)
    np.save(OUT / "placebo_chi2_pooled10.npy", chi)
    pd.set_option("display.width", 250)
    for g in ("style", "tier"):
        S = R[R.grouping == g].sort_values("p_placebo")
        print(f"\n== {g} groups, 2016-25, {len(S)} tests: p_placebo<.05: {(S.p_placebo < .05).sum()}, <.01: {(S.p_placebo < .01).sum()}, q<.10: {(S.q_wald < .10).sum()} ==")
        for r in S.itertuples():
            sc = 1 if r.outcome == "fe_ppr" else 100
            line = " | ".join(f"{getattr(r, f'name{s}')[:16]:16s} {getattr(r, f'b{s}') * sc:+.2f}±{getattr(r, f'se{s}') * sc:.2f}" for s in range(3))
            eras = " ; ".join(f"{e}: " + "/".join(f"{getattr(r, f'{e}_b{s}') * sc:+.2f}" for s in range(3)) for e in ERAS)
            print(f"{r.pos} {r.outcome:8s} {r.feature:6s} n={r.n} p={r.p_wald:.3f} placebo={r.p_placebo:.3f} q={r.q_wald:.2f} pooled={r.pooled * sc:+.2f}±{r.pooled_se * sc:.2f}\n     {line}\n     by era {eras}")


# ───────── after ESPN's number: 2021-25 (ESPN's weekly projection exists from 2021) ─────────
def espn5(reps: int = 300) -> None:
    from backtest import vegas_signal as vs
    rng = np.random.default_rng(33)
    w_h = pd.read_csv(OUT / "games_holdout.csv.gz")
    w_h = w_h[w_h["season"] >= 2021].copy()
    pr = vs.rows()[["season", "week", "espn_id", "espn_proj"]]
    w_h = w_h.merge(pr, on=["season", "week", "espn_id"], how="left")
    w_h["resid"] = w_h["ppr"] - w_h["espn_proj"]
    w_d = rds.add_z(pd.read_csv(OUT / "games.csv.gz"))
    cols = ["season", "week", "gsis", "pos", "p0", "p1", "p2", "ppr", "resid", "espn_proj", "tgt_sh_prior", "car_sh_prior", "snap_prior", "z_pass_r", "z_run_r"]
    W = pd.concat([w_h[cols], w_d[cols]], ignore_index=True)
    rows, data = [], {}
    for pos in rds.POS:
        d = W[(W["pos"] == pos) & W["resid"].notna()].dropna(subset=["p0", "p1", "p2"]).sort_values(["gsis", "season", "week"]).reset_index(drop=True)
        cl = d["gsis"].astype("category").cat.codes.to_numpy()
        st = rds._starts(cl)
        n_cl = np.diff(np.r_[st, len(d)])
        cor = np.repeat(np.arange(len(st)), n_cl)
        rank = np.arange(len(d)) - st[cor]
        Ptr, tnames = rds.tier_groups(d, pos) if pos != "QB" else (None, None)
        Ps = {"style": d[["p0", "p1", "p2"]].to_numpy(float), "tier": Ptr}
        names = {"style": rds.roles.load_model()["style"][pos]["names"], "tier": tnames}
        data[pos] = (d, cl, st, n_cl, cor, rank, Ps)
        for f in ["pass_r"] + (["run_r"] if pos == "RB" else []):
            for g in ("style", "tier"):
                if Ps[g] is None:
                    continue
                m = d["z_" + f].notna().to_numpy()
                x, y = d.loc[m, "z_" + f].to_numpy(float), d.loc[m, "resid"].to_numpy(float)
                r = rds.fit(Ps[g][m], x, y, cl[m], want_v=True)
                row = dict(pos=pos, feature=f, grouping=g, n=r["n"], G=r["G"], chi2=r["chi2"], p_wald=rds.chi2_sf(r["chi2"], 2), pooled=r["b0"], pooled_se=r["se0"])
                for s in range(3):
                    row.update({f"name{s}": names[g][s], f"b{s}": r["b"][s], f"se{s}": r["se"][s], f"mean_proj{s}": float((Ps[g][m][:, s] * d.loc[m, "espn_proj"].to_numpy()).sum() / Ps[g][m][:, s].sum())})
                rows.append(row)
    R = pd.DataFrame(rows)
    chi = np.full((reps, len(R)), np.nan)
    for b in range(reps):
        perm = {pos: {g: (rds.perm_P(data[pos][6][g], data[pos][2], data[pos][3], data[pos][5], data[pos][4], rng) if data[pos][6][g] is not None else None) for g in ("style", "tier")} for pos in rds.POS}
        for j, r in enumerate(R.itertuples(index=False)):
            d, cl = data[r.pos][0], data[r.pos][1]
            m = d["z_" + r.feature].notna().to_numpy()
            out = rds.fit(perm[r.pos][r.grouping][m], d.loc[m, "z_" + r.feature].to_numpy(float), d.loc[m, "resid"].to_numpy(float), cl[m])
            if out:
                chi[b, j] = out["chi2"]
    R["p_placebo"] = [(np.nansum(chi[:, j] >= R["chi2"].iloc[j]) + 1) / (np.isfinite(chi[:, j]).sum() + 1) for j in range(len(R))]
    R.to_csv(OUT / "results_espn5.csv", index=False)
    print(f"points minus ESPN's projection, 2021-25, {len(R)} tests; p_placebo<.05: {(R.p_placebo < .05).sum()}")
    for r in R.sort_values("p_placebo").itertuples():
        line = " | ".join(f"{getattr(r, f'name{s}')[:18]:18s} {getattr(r, f'b{s}'):+.2f}±{getattr(r, f'se{s}'):.2f} (proj {getattr(r, f'mean_proj{s}'):.1f})" for s in range(3))
        print(f"{r.pos} {r.feature:6s} {r.grouping:5s} n={r.n} p={r.p_wald:.3f} placebo={r.p_placebo:.3f} pooled={r.pooled:+.2f}±{r.pooled_se:.2f}\n     {line}")


# ───────── the RB x blitz-rate watch item: one extra season (2022 FTN charting has blitzer counts) ─────────
def blitz22() -> None:
    from app.engine import matchups_data as md
    from backtest import vegas_signal as vs
    p = OUT / "pbp_2022_blitz.csv.gz"
    b = pd.read_csv(p) if p.exists() else md.build(2022)
    if not p.exists():
        b.to_csv(p, index=False, compression="gzip")
    b = b[b["db"] == 1].copy()
    b["blitz"] = (b["n_blitzers"].fillna(0) > 0).astype(float).where(b["n_blitzers"].notna())
    G = b.groupby(["week", "defteam"]).agg(db=("db", "size"), blitz_g=("blitz", "mean")).reset_index()
    G = G[G["db"] >= 15]
    rows = []
    for w in range(2, 19):
        past = G[G["week"] < w]
        lg = past["blitz_g"].mean() if len(past) else np.nan
        for t in G["defteam"].unique():
            pt = past[past["defteam"] == t]
            n = len(pt)
            rows.append({"week": w, "opp": t, "blitz_td": (n * pt["blitz_g"].mean() + rds.PRIOR_G * lg) / (n + rds.PRIOR_G) if n else lg})
    B = pd.DataFrame(rows)
    B["z_blitz_td"] = (B["blitz_td"] - B.groupby("week")["blitz_td"].transform("mean"))
    B["z_blitz_td"] /= B["z_blitz_td"].std()
    w_h = pd.read_csv(OUT / "games_holdout.csv.gz")
    w_h = w_h[(w_h["season"] == 2022) & (w_h["pos"] == "RB")].merge(B, on=["week", "opp"], how="left")
    pr = vs.rows()[["season", "week", "espn_id", "espn_proj"]]
    w_h = w_h.merge(pr, on=["season", "week", "espn_id"], how="left")
    w_h["resid"] = w_h["ppr"] - w_h["espn_proj"]
    d = w_h.dropna(subset=["p0", "z_blitz_td"]).sort_values(["gsis", "week"]).reset_index(drop=True)
    d["ps"] = d["gsis"].astype("category").cat.codes
    d["n_ps"] = d.groupby("ps")["ppr"].transform("size")
    cl = d["gsis"].astype("category").cat.codes.to_numpy()
    P, names = rds.tier_groups(d, "RB")
    print(f"2022 RB games {len(d)}, players {d['gsis'].nunique()}; blitz rate sd across defense-weeks (z scale only): the 2023-25 estimates were fe_ppr  lead -0.95 | split -0.65 | change of pace +0.00 (per +1 sd of blitz rate to date)")
    for o in ("fe_ppr", "resid"):
        m = (d[o].notna() & ((d["n_ps"] >= 4) if o == "fe_ppr" else True)).to_numpy()
        r = rds.fit(P[m], d.loc[m, "z_blitz_td"].to_numpy(float), d.loc[m, o].to_numpy(float), cl[m], want_v=True, fe=d.loc[m, "ps"].to_numpy() if o == "fe_ppr" else None)
        print(f"  {o}: " + " | ".join(f"{names[s][:20]:20s} {r['b'][s]:+.2f}±{r['se'][s]:.2f}" for s in range(3)) + f"   (n={r['n']}, p_omnibus={rds.chi2_sf(r['chi2'], 2):.2f}, pooled {r['b0']:+.2f}±{r['se0']:.2f})")
        c = np.array([0.5, 0.5, -1.0])
        k = len(r["b"])
        V = r["V"][-k:, -k:]
        est = float(c @ r["b"])
        se = float(np.sqrt(c @ V @ c))
        print(f"     lead+split minus change-of-pace: {est:+.2f} ± {se:.2f} (expected negative)")


# ───────── numbers the Matchups card quotes (written into app/engine/matchups_credit.json) ─────────
def export() -> None:
    """style_run: the one replicated style interaction (RB x run defense, 2016-25).  tier_credit: how far ESPN's projection moves with the
    defense by role size (2023-25), a sanity check on the card's 'share of the projection' scaling."""
    import json
    from backtest import vegas_signal as vs
    R = pd.read_csv(OUT / "results_pooled10.csv")
    r = R[(R.pos == "RB") & (R.feature == "run_r") & (R.outcome == "fe_ppr") & (R.grouping == "style")].iloc[0]
    names = [r[f"name{i}"] for i in range(3)]
    sr = {"seasons": "2016-25", "feature": "run_r", "unit": "points per +1 sd of run-defense softness, compared with the player's own season",
          "n_games": int(r.n), "p_wald": float(r.p_wald), "p_placebo": float(r.p_placebo), "q": float(r.q_wald),
          "styles": {names[i]: {"per_sd": float(r[f"b{i}"]), "se": float(r[f"se{i}"])} for i in range(3)},
          "eras": {e: {names[i]: float(r[f"{e}_b{i}"]) for i in range(3)} for e in ERAS}}
    # ESPN's own movement with the defense by role size (projection minus trailing-4 average, 2023-25, weeks 4+)
    P = mm.plays()
    Rt = mm.weekly_ratings(P, "def").rename(columns={"team": "opp"})
    raw = vs.rows()
    raw = raw[raw.position.isin(rds.POS) & raw.season.isin(rds.SEAS) & (raw.week >= 4)].copy()
    X = raw.merge(Rt[["season", "week", "opp", "pass_r", "run_r"]], on=["season", "week", "opp"]).dropna(subset=["pass_r", "run_r"]).sort_values(["espn_id", "season", "week"]).reset_index(drop=True)
    X["trail"] = X.groupby(["espn_id", "season"])["points"].transform(lambda x: x.shift(1).rolling(4, min_periods=3).mean())
    X["credit_y"] = X["espn_proj"] - X["trail"]
    g = pd.read_csv(OUT / "games.csv.gz")[["season", "week", "espn_id", "car_sh_prior", "tgt_sh_prior", "snap_prior"]]
    X = X.merge(g, on=["season", "week", "espn_id"], how="left")

    def tier(row):
        if pd.isna(row.car_sh_prior):
            return None
        if row.position == "RB":
            return "Lead back" if row.car_sh_prior >= .6 else "Split backfield" if row.car_sh_prior >= .4 else "Change of pace"
        if row.position == "WR":
            return "Alpha target share" if row.tgt_sh_prior >= .25 else "Starter" if row.tgt_sh_prior >= .17 else "Low volume"
        if row.position == "TE":
            return "Featured" if row.tgt_sh_prior >= .17 else "Starter" if row.snap_prior >= 60 else "Rotational"
        return None

    X["tier"] = [tier(t) for t in X.itertuples()]
    sd_ref = json.loads(CREDIT.read_text())["sd_ref"]
    out = {}
    for pos, xs in (("RB", ["pass_r", "run_r"]), ("WR", ["pass_r"]), ("TE", ["pass_r"])):
        for t, d in X[X.position == pos].dropna(subset=["credit_y", "tier"]).groupby("tier"):
            if len(d) < 80:
                continue
            by = ["season", "week", "position"]
            Y = (d["credit_y"] - d.groupby(by)["credit_y"].transform("mean")).to_numpy(float)
            Xm = (d[xs] - d.groupby(by)[xs].transform("mean")).to_numpy(float)
            b = np.linalg.lstsq(Xm, Y, rcond=None)[0]
            e = Y - Xm @ b
            u, inv = np.unique(d["espn_id"].to_numpy(), return_inverse=True)
            S = np.zeros((len(u), Xm.shape[1]))
            np.add.at(S, inv, Xm * e[:, None])
            ix = np.linalg.inv(Xm.T @ Xm)
            se = np.sqrt(np.diag(ix @ (S.T @ S) @ ix))
            out.setdefault(pos, {})[t] = {"n": int(len(d)), "mean_proj": float(d["espn_proj"].mean()),
                                          **{x: {"per_sd": float(b[i] * sd_ref[x.split("_")[0]]), "se": float(se[i] * sd_ref[x.split("_")[0]])} for i, x in enumerate(xs)}}
    cur = json.loads(CREDIT.read_text())
    cur["style_run"] = sr
    cur["tier_credit"] = out
    CREDIT.write_text(json.dumps(cur, indent=1))
    print("wrote style_run and tier_credit into", CREDIT)
    print(json.dumps(sr["styles"], indent=1)[:600])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "main"
    {"main": main, "pooled": pooled, "espn5": espn5, "blitz22": blitz22, "export": export}[cmd]()
