"""Which matchup mechanisms predict anything beyond ESPN's weekly projection?  (2023-25, weeks 4+; coverage needs 2023+)

  credit : defense ratings built from every play (app/engine/matchups_rating.py): 
           (a) does a defense's trailing EPA allowed predict the next game's offense EPA per dropback?
           (b) what is left of (actual - ESPN projection) once ESPN has priced the defense?
           (c) how much does ESPN's projection already move with the defense? (projection minus the player's last-4 average
               regressed on the rating; this is what the Matchups card calls "already in the number")
           (d) does the defense change the spread of outcomes around the projection?
  lens   : the man/zone lens (a player's shrunk man-minus-zone target-rate gap x the opponent's expected man rate): does it
           predict the player's targets in a game, out of sample (history = earlier seasons only)?

Writes app/engine/matchups_credit.json, which the Matchups card reads.
  python3 backtest/matchup_mechanisms.py credit
  python3 backtest/matchup_mechanisms.py lens
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
from app.engine import matchups_data as md  # noqa: E402
from app.engine import matchups_rating as mr  # noqa: E402
from backtest import vegas_signal as vs  # noqa: E402

SEAS = [2023, 2024, 2025]
OUT = ROOT / "app" / "engine" / "matchups_credit.json"
SPEC = {"QB": ["pass_r"], "RB": ["pass_r", "run_r"], "WR": ["pass_r"], "TE": ["pass_r"]}  # fixed before looking at results
COLS = ["game_id", "season", "week", "posteam", "defteam", "db", "run", "epa"]
EMPTY = pd.DataFrame({"epa": pd.Series(dtype=float), "n": pd.Series(dtype=float)})


def plays() -> pd.DataFrame:
    P = pd.concat([md.load(y)[COLS] for y in SEAS], ignore_index=True)
    return P[P["week"] <= 18]


def weekly_ratings(P: pd.DataFrame, side: str = "def") -> pd.DataFrame:
    """Rating of each team before each week: earlier weeks of the season + last season. side='off' rates offenses instead."""
    if side == "off":
        P = P.rename(columns={"defteam": "posteam", "posteam": "defteam"})
    rows = []
    for s in SEAS:
        ps, pp = P[P["season"] == s], P[P["season"] == s - 1]
        prior = {k: (mr.team_table(pp, k) if len(pp) else EMPTY) for k in ("pass", "run")}
        for w in range(2, 19):
            cur = ps[ps["week"] < w]
            rp = mr.rate(prior["pass"], mr.team_table(cur, "pass"))
            rr = mr.rate(prior["run"], mr.team_table(cur, "run")).reindex(rp.index)
            rows.append(pd.DataFrame({"season": s, "week": w, "team": rp.index, "pass_r": (rp - rp.mean()).values, "run_r": (rr - rr.mean()).values}))
    return pd.concat(rows, ignore_index=True)


def cl_idx(df: pd.DataFrame):
    keys = df.groupby(["season", "week"]).indices
    return keys, list(keys)


def boot(df: pd.DataFrame, f, boots: int = 1000, seed: int = 0):
    keys, kk = cl_idx(df)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(boots):
        ii = np.concatenate([keys[kk[p]] for p in rng.integers(0, len(kk), len(kk))])
        out.append(f(df.iloc[ii]))
    return np.percentile(np.array(out), [2.5, 97.5], axis=0)


def ols_centered(df: pd.DataFrame, ycol: str, xcols: list[str]) -> np.ndarray:
    by = ["season", "week", "position"]
    Y = (df[ycol] - df.groupby(by)[ycol].transform("mean")).to_numpy(float)
    X = (df[xcols] - df.groupby(by)[xcols].transform("mean")).to_numpy(float)
    return np.linalg.lstsq(X, Y, rcond=None)[0]


def fit(df: pd.DataFrame, ycol: str, xcols: list[str], boots: int = 1000):
    df = df.reset_index(drop=True)
    b = ols_centered(df, ycol, xcols)
    lo, hi = boot(df, lambda d: ols_centered(d, ycol, xcols), boots)
    return b, lo, hi


def ols_prop(df: pd.DataFrame, ycol: str, xcols: list[str]) -> np.ndarray:
    """Same as ols_centered, but the effect scales with the player's own projection: y = b * (rating * espn_proj).
    b is a share of the projection per unit of rating, so a 16-point back and a 6-point back are not forced to share one number."""
    by = ["season", "week", "position"]
    Y = (df[ycol] - df.groupby(by)[ycol].transform("mean")).to_numpy(float)
    Z = df[xcols].mul(df["espn_proj"], axis=0)
    X = (Z - Z.groupby([df["season"], df["week"], df["position"]]).transform("mean")).to_numpy(float)
    return np.linalg.lstsq(X, Y, rcond=None)[0]


def fit_prop(df: pd.DataFrame, ycol: str, xcols: list[str], boots: int = 1000):
    df = df.reset_index(drop=True)
    b = ols_prop(df, ycol, xcols)
    lo, hi = boot(df, lambda d: ols_prop(d, ycol, xcols), boots)
    return b, lo, hi


def credit() -> dict:
    P = plays()
    R = weekly_ratings(P, "def")
    Ro = weekly_ratings(P, "off")
    res: dict = {"fit": {"seasons": SEAS, "k_plays": mr.K_PLAYS, "rho": mr.RHO, "spec": SPEC}}
    # (a) mechanism: offense pass EPA per dropback in the game vs the defense's trailing rating (+ offense control)
    pas = P[P["db"] == 1]
    G = pas.groupby(["season", "week", "posteam", "defteam"]).agg(n=("epa", "size"), epa=("epa", "sum")).reset_index()
    G = G.merge(R.rename(columns={"team": "defteam", "pass_r": "def_r"})[["season", "week", "defteam", "def_r"]], on=["season", "week", "defteam"])
    G = G.merge(Ro.rename(columns={"team": "posteam", "pass_r": "off_r"})[["season", "week", "posteam", "off_r"]], on=["season", "week", "posteam"])
    G["y"] = G["epa"] / G["n"]
    G = G[(G["week"] >= 4) & (G["n"] >= 15)].reset_index(drop=True)

    def wls(d):
        X = np.column_stack([np.ones(len(d)), d[["def_r", "off_r"]].to_numpy(float)]); W = d["n"].to_numpy(float)
        return np.linalg.solve(X.T @ (X * W[:, None]), X.T @ (d["y"].to_numpy(float) * W))[1]
    b = wls(G); lo, hi = boot(G, wls)
    res["mechanism"] = {"slope": float(b), "ci": [float(lo), float(hi)], "n_games": int(len(G)), "sd_rating": float(G["def_r"].std())}
    print(f"(a) offense EPA/dropback on defense rating: slope {b:+.2f} [{lo:+.2f},{hi:+.2f}] n={len(G)} games, sd rating {G['def_r'].std():.3f}")
    # rating spread across teams, weeks >= 4 (descriptive scale for the card)
    late = R[R["week"] >= 4]
    sd_ref = {"pass": float(late.groupby(["season", "week"])["pass_r"].std().mean()), "run": float(late.groupby(["season", "week"])["run_r"].std().mean())}
    res["sd_ref"] = sd_ref
    # player-week rows
    raw = vs.rows()
    raw = raw[raw["position"].isin(SPEC) & raw["season"].isin(SEAS) & (raw["week"] >= 4)].copy()
    raw["resid"] = raw["points"] - raw["espn_proj"]
    X = raw.merge(R.rename(columns={"team": "opp"})[["season", "week", "opp", "pass_r", "run_r"]], on=["season", "week", "opp"], how="inner").dropna(subset=["pass_r", "run_r"])
    X = X.sort_values(["espn_id", "season", "week"]).reset_index(drop=True)
    X["trail"] = X.groupby(["espn_id", "season"])["points"].transform(lambda s: s.shift(1).rolling(4, min_periods=3).mean())
    X["credit_y"] = X["espn_proj"] - X["trail"]
    res["miss_sd"] = {p: float(X[X["position"] == p]["resid"].std()) for p in SPEC}
    res["rows"] = {p: int((X["position"] == p).sum()) for p in SPEC}
    res["credit"], res["left_over"] = {}, {}
    for pos, xc in SPEC.items():
        g = X[X["position"] == pos]
        gc = g.dropna(subset=["credit_y"])
        b, lo, hi = fit(gc, "credit_y", xc)
        res["credit"][pos] = {c: {"per_unit": float(b[i]), "per_sd": float(b[i] * sd_ref[c.split("_")[0]]), "ci_per_sd": [float(lo[i] * sd_ref[c.split("_")[0]]), float(hi[i] * sd_ref[c.split("_")[0]])]} for i, c in enumerate(xc)}
        res["credit"][pos]["n"] = int(len(gc))
        # the same measurement as a share of the player's projection (what the card now uses: a bigger role has more to move)
        bp, lop, hip = fit_prop(gc, "credit_y", xc)
        for i, c in enumerate(xc):
            sdc = sd_ref[c.split("_")[0]]
            res["credit"][pos][c].update(pct_per_unit=float(bp[i]), pct_per_sd=float(bp[i] * sdc), ci_pct_per_sd=[float(lop[i] * sdc), float(hip[i] * sdc)])
        res["credit"][pos]["mean_proj"] = float(gc["espn_proj"].mean())
        print(f"   as a share of his projection: " + ", ".join(f"{c} {bp[i] * sd_ref[c.split('_')[0]] * 100:+.1f}% per sd [{lop[i] * sd_ref[c.split('_')[0]] * 100:+.1f},{hip[i] * sd_ref[c.split('_')[0]] * 100:+.1f}]" for i, c in enumerate(xc)) + f" (mean projection {gc['espn_proj'].mean():.1f})")
        b2, lo2, hi2 = fit(g, "resid", xc)
        res["left_over"][pos] = {c: {"per_sd": float(b2[i] * sd_ref[c.split("_")[0]]), "ci_per_sd": [float(lo2[i] * sd_ref[c.split("_")[0]]), float(hi2[i] * sd_ref[c.split("_")[0]])]} for i, c in enumerate(xc)}
        res["left_over"][pos]["n"] = int(len(g))
        print(f"{pos}: ESPN already credits " + ", ".join(f"{c} {b[i] * sd_ref[c.split('_')[0]]:+.2f} [{lo[i] * sd_ref[c.split('_')[0]]:+.2f},{hi[i] * sd_ref[c.split('_')[0]]:+.2f}] pts/sd" for i, c in enumerate(xc)) + f" (n={len(gc)})"
              + " | left over " + ", ".join(f"{c} {b2[i] * sd_ref[c.split('_')[0]]:+.2f} [{lo2[i] * sd_ref[c.split('_')[0]]:+.2f},{hi2[i] * sd_ref[c.split('_')[0]]:+.2f}]" for i, c in enumerate(xc)) + f" (n={len(g)})")
    # pooled left-over (all four positions, the pre-specified features), as the headline
    allr = []
    for pos, xc in SPEC.items():
        g = X[X["position"] == pos]
        b2, _, _ = fit(g, "resid", xc, boots=10)
        allr.append((pos, len(g), float(sum(b2[i] * sd_ref[c.split("_")[0]] for i, c in enumerate(xc)))))
    # (d) spread of outcomes by defense tercile
    X["terc"] = X.groupby(["season", "week", "position"])["pass_r"].transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=["tough", "mid", "soft"]))
    d = {}
    for pos in SPEC:
        t = X[X["position"] == pos].groupby("terc").agg(n=("resid", "size"), mean=("resid", "mean"), sd=("resid", "std"))
        d[pos] = {k: {"n": int(r["n"]), "mean": float(r["mean"]), "sd": float(r["sd"])} for k, r in t.iterrows()}
        print(pos, "spread by tercile:", {k: (round(v["mean"], 2), round(v["sd"], 2)) for k, v in d[pos].items()})
    res["spread"] = d
    return res


def lens() -> dict:
    from app.engine import matchups as m
    ids = m._ids()
    hist = []
    for y in SEAS:
        d = md.load(y); d = d[d["db"] == 1].copy(); d["pidx"] = np.arange(len(d)) + y * 1_000_000; hist.append(d)
    H = pd.concat(hist, ignore_index=True)
    H["man"] = np.where(H["defense_man_zone_type"].isna(), np.nan, (H["defense_man_zone_type"] == "MAN_COVERAGE").astype(float))
    oo = H[["pidx", "season", "week", "man", "offense_players", "defteam", "receiver", "pass_attempt", "epa"]].dropna(subset=["offense_players"]).copy()
    tg = oo[oo["receiver"].notna() & (oo["pass_attempt"] == 1)][["pidx", "receiver", "epa"]].rename(columns={"receiver": "gsis"})
    oo = oo.drop(columns=["receiver", "pass_attempt", "epa"])
    oo["gsis"] = oo["offense_players"].str.split(";"); oo = oo.explode("gsis").drop(columns="offense_players")
    oo["pos"] = oo["gsis"].map(ids["position"]); oo = oo[oo["pos"].isin(["RB", "WR", "TE"])]
    oo = oo.merge(tg, on=["pidx", "gsis"], how="left"); oo["tgt"] = oo["epa"].notna().astype(int)
    # the shrinkage helper in the engine wants the same columns it gets in production
    oo["ppr"] = 0.0
    lgman = {y: float(H[(H["season"] == y) & H["man"].notna()]["man"].mean()) for y in SEAS}
    tm = H[H["man"].notna()].groupby(["season", "defteam"])["man"].mean().unstack(0)
    a, b = tm[2023].dropna(), tm[2024].dropna(); a, b = a.align(b, join="inner")
    pers = float(((a - a.mean()) * (b - b.mean())).sum() / ((a - a.mean()) ** 2).sum())
    rows = []
    for S in (2024, 2025):
        prior = oo[oo["season"] < S]
        eb = m._eb(prior)
        pm = prior[prior["man"] == 1].groupby("gsis").agg(db_m=("pidx", "size"), t_m=("tgt", "sum"))
        pz = prior[prior["man"] == 0].groupby("gsis").agg(db_z=("pidx", "size"), t_z=("tgt", "sum"))
        pl = pm.join(pz, how="inner"); pl["pos"] = pl.index.map(ids["position"])
        pl = pl[(pl["db_m"] + pl["db_z"] >= 150) & (pl["db_m"] >= 30) & (pl["db_z"] >= 30)]
        pa = (pl["t_m"] + pl["t_z"]) / (pl["db_m"] + pl["db_z"])
        dd = pl["t_m"] / pl["db_m"] - pl["t_z"] / pl["db_z"]
        s2 = pa * (1 - pa) * (1 / pl["db_m"] + 1 / pl["db_z"])
        tau2 = pl["pos"].map(lambda p: eb[p]["tr_mz"]["tau2"]); mean_d = pl["pos"].map(lambda p: eb[p]["tr_mz"]["mean"])
        pl["d_sh"] = mean_d + tau2 / (tau2 + s2) * (dd - mean_d)
        tp = H[(H["season"] == S - 1) & H["man"].notna()].groupby("defteam")["man"].mean()
        exp_man = lgman[S - 1] + pers * (tp - lgman[S - 1])
        cur = oo[oo["season"] == S]
        g = cur.groupby(["gsis", "week"]).agg(db_on=("pidx", "size"), tgt=("tgt", "sum"), man_sh=("man", "mean"), defteam=("defteam", lambda s: s.mode().iat[0])).reset_index()
        allp = oo[oo["season"] <= S].groupby(["gsis", "season", "week"]).agg(db=("pidx", "size"), t=("tgt", "sum")).reset_index().sort_values(["gsis", "season", "week"])
        allp["cdb"] = allp.groupby("gsis")["db"].cumsum() - allp["db"]; allp["ct"] = allp.groupby("gsis")["t"].cumsum() - allp["t"]
        allp = allp[allp["season"] == S][["gsis", "week", "cdb", "ct"]]
        g = g.merge(allp, on=["gsis", "week"]).merge(pl[["d_sh", "pos"]], left_on="gsis", right_index=True)
        g["base_rate"] = g["ct"] / g["cdb"]; g["exp_man"] = g["defteam"].map(exp_man); g["season"] = S
        rows.append(g)
    G = pd.concat(rows, ignore_index=True)
    G = G[(G["cdb"] >= 150) & (G["db_on"] >= 15) & G["exp_man"].notna() & G["man_sh"].notna()].reset_index(drop=True)
    G["y"] = G["tgt"] - G["base_rate"] * G["db_on"]
    G["x_exp"] = G["d_sh"] * (G["exp_man"] - G["season"].map(lambda s: lgman[s - 1])) * G["db_on"]
    G["x_known"] = G["d_sh"] * (G["man_sh"] - G["season"].map(lambda s: lgman[s])) * G["db_on"]

    def slope(df, c):
        x = df[c] - df[c].mean(); y = df["y"] - df["y"].mean(); return float((x * y).sum() / (x * x).sum())
    out = {"n_player_games": int(len(G)), "players": int(G["gsis"].nunique()), "sd_targets_miss": float(G["y"].std()), "persist_man": pers,
           "sd_expected_effect": float(G["x_exp"].std()), "p90_expected_effect": float(G["x_exp"].abs().quantile(0.9)), "sd_known_effect": float(G["x_known"].std())}
    for c, name in (("x_exp", "expected"), ("x_known", "known")):
        s = slope(G, c); lo, hi = boot(G, lambda d: slope(d, c))
        out[f"slope_{name}"] = float(s); out[f"ci_{name}"] = [float(lo), float(hi)]
        print(f"lens, {name} coverage: slope {s:+.2f} [{lo:+.2f},{hi:+.2f}]  sd(effect)={G[c].std():.3f} targets  n={len(G)}")
    print(f"sd of a player's target miss per game: {G['y'].std():.2f}")
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    cur = json.loads(OUT.read_text()) if OUT.exists() else {}
    if which in ("credit", "all"):
        cur.update(credit())
    if which in ("lens", "all"):
        cur["lens"] = lens()
    OUT.write_text(json.dumps(cur, indent=1))
    print("wrote", OUT)
