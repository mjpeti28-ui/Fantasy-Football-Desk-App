"""Exploratory PFF Premium Stats analysis: 2026 outlier lists, stat stability, ML residual models, archetypes.

Reads the cached PFF facets (tools/fetch_pff_stats.py; 2026 windows w01-06, w04-06, w01-03 and 2021-25
w07-12 on top of the backtest windows). 2026 ESPN rosters, points and the week-6 ledger come from the main
checkout. Writes a self-contained HTML report with player-level data to .cache/pff/report/ (gitignored,
holds PFF-derived data) and prints aggregate tables only.

  A. 2026 lists (usage > production, production > usage, role movers, efficiency extremes), each checked
     on 2021-25 checkpoints: does the flag predict future_ppg - live (est_ppg + 1/3 ECR)?
  B. Split-half (wk 1-6 vs 7-12) and 3-week block stability of each stat, and its link to fantasy ppg.
  C. LOSO-by-season models of future_ppg - live (ridge, lasso, GBM, RF, MLP), hyperparameters picked on
     inner LOSO folds; PCA + k-means archetypes per position.

  python backtest/pff_explore.py
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import silhouette_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backtest import ecr_signal as es  # noqa: E402
from backtest import pff_stats as ps  # noqa: E402

MAIN = Path.home() / "Developer" / "Fantasy-Football-Desk-App"  # live 2026 data lives in the main checkout
REPORT = ps.CACHE / "report"
SEASONS = es.SEASONS
FACETS = ["receiving/summary", "rushing/summary", "passing/summary"]
MY_TEAM = 3
RNG = np.random.default_rng(0)


# ------------------------------------------------------------------ features

@lru_cache(None)
def _load(facet: str, y: int, lo: int, hi: int) -> pd.DataFrame:
    return ps.load(facet, y, lo, hi)


@lru_cache(None)
def feats(y: int, lo: int, hi: int) -> pd.DataFrame:
    """One row per PFF player_id over weeks lo..hi: usage, efficiency and grade features plus name/pos/team."""
    rec, rus, pas = (_load(f, y, lo, hi) for f in FACETS)
    # shares are per game played (player per-game volume over team per-game volume), so missed games don't dilute them
    tg = pd.concat([rec, rus, pas]).groupby("team")["player_game_count"].max()
    team_db = pas.groupby("team")["dropbacks"].sum() / tg
    team_tgt = rec.groupby("team")["targets"].sum() / tg
    team_att = rus.groupby("team")["attempts"].sum() / tg
    r, u, q = rec.set_index("player_id"), rus.set_index("player_id"), pas.set_index("player_id")
    rg, ug, qg = r["player_game_count"], u["player_game_count"], q["player_game_count"]
    R = pd.DataFrame({
        "route_share": r["routes"] / rg / r["team"].map(team_db), "tgt_share": r["targets"] / rg / r["team"].map(team_tgt),
        "tprr": r["targets"] / r["routes"].replace(0, np.nan), "yprr": r["yprr"], "adot": r["avg_depth_of_target"],
        "slot_rate": r["slot_rate"], "inline_rate": r["inline_rate"], "yac_rec": r["yards_after_catch_per_reception"],
        "drop_rate": r["drop_rate"], "rec_grade": r["grades_offense"], "route_grade": r["grades_pass_route"],
        "targets_pg": r["targets"] / rg, "routes_pg": r["routes"] / rg, "rec_td_pg": r["touchdowns"] / rg, "rec_games": rg,
    })
    U = pd.DataFrame({
        "rush_share": u["attempts"] / ug / u["team"].map(team_att), "rush_pg": u["attempts"] / ug, "ypa": u["ypa"],
        "yco": u["yco_attempt"], "elusive": u["elusive_rating"], "breakaway": u["breakaway_percent"],
        "rush_grade": u["grades_offense"], "run_grade": u["grades_run"], "rush_td_pg": u["touchdowns"] / ug,
        "scr_pg": u["scrambles"] / ug, "rush_games": ug,
    })
    Q = pd.DataFrame({
        "db_pg": q["dropbacks"] / qg, "pass_grade": q["grades_pass"], "btt": q["btt_rate"], "twp": q["twp_rate"],
        "adot_pass": q["avg_depth_of_target"], "sack_pct": q["sack_percent"], "ttt": q["avg_time_to_throw"],
        "acc": q["accuracy_percent"], "ypa_pass": q["ypa"], "pass_games": qg,
    })
    out = R.join(U, how="outer").join(Q, how="outer")
    out["air_pg"] = out["targets_pg"] * out["adot"]
    meta = pd.concat([rec, rus, pas])[["player_id", "player", "position", "team"]]
    meta["pos"] = meta["position"].map(ps.POS)
    meta = meta.dropna(subset=["pos"]).groupby("player_id").agg(
        player=("player", "first"), pos=("pos", lambda s: s.mode().iat[0]), team=("team", "first"))
    out = meta.join(out, how="inner").replace([np.inf, -np.inf], np.nan)
    out["pff_games"] = out[["rec_games", "rush_games", "pass_games"]].max(axis=1)
    out.index.name = "pff_id"
    return out


SHARES = ["route_share", "tgt_share", "rush_share", "targets_pg", "rush_pg", "db_pg", "pff_games"]


@lru_cache(None)
def weekly(y: int) -> pd.DataFrame:
    base = MAIN if y == 2026 else ROOT
    d = pd.read_csv(base / "history" / "actuals" / str(y) / "players_weekly.csv")
    return d[d["kind"] == "actual"]


def window_ppg(y: int, lo: int, hi: int) -> pd.DataFrame:
    d = weekly(y)
    d = d[d["week"].between(lo, hi) & d["played"].astype(bool)]
    g = d.groupby("espn_id").agg(pts=("points", "sum"), gp=("points", "size"))
    g["ppg"] = g["pts"] / g["gp"]
    return g[["ppg", "gp"]]


@lru_cache(None)
def id_map(y: int) -> pd.Series:
    """pff_id -> espn_id. 2026 reuses pff_stats.id_map against the main checkout's nflverse ids and the w01-06 window."""
    if y != 2026:
        return ps.id_map(y).set_index("pff_id")["espn_id"]
    orig_load, orig_root = ps.load, ps.ROOT
    ps.load, ps.ROOT = (lambda fc, s, lo, hi: orig_load(fc, s, 1, 6)), MAIN
    try:
        return ps.id_map(2026).set_index("pff_id")["espn_id"]
    finally:
        ps.load, ps.ROOT = orig_load, orig_root


def window_table(y: int, lo: int, hi: int) -> pd.DataFrame:
    f = feats(y, lo, hi).copy()
    f["espn_id"] = f.index.map(id_map(y))
    return f.join(window_ppg(y, lo, hi), on="espn_id")


def eligible(d: pd.DataFrame, min_g: int = 3) -> pd.Series:
    """Real-volume players: RB by carries/targets, WR/TE by routes/targets, QB by dropbacks."""
    pos = d["pos"] if "pos" in d else d["position"]
    games = d["gp"].fillna(0) >= min_g
    vol = np.select(
        [pos.isin(["WR", "TE"]), pos == "RB", pos == "QB"],
        [(d["routes_pg"] >= 15) | (d["targets_pg"] >= 3), (d["rush_pg"] >= 5) | (d["targets_pg"] >= 2.5), d["db_pg"] >= 20],
        False)
    return games & vol


# ------------------------------------------------------------------ historical checkpoint rows

def hist_rows() -> pd.DataFrame:
    """usage_rows checkpoints (all est_ppg > 0) with live blend, extended PFF features over 1..W-1 and
    last-3 shares, and fantasy ppg/games over 1..W-1."""
    x = ps.build_rows()
    keep = list(pd.read_csv(ps.OUT / "usage_rows_2025.csv", nrows=1).columns) + ["pff_id", "ecr", "ecr_ppg", "live"]
    x = x[[c for c in keep if c in x.columns]]
    parts = []
    for y in SEASONS:
        for W in range(3, 14):
            std = window_table(y, 1, W - 1).drop(columns=["espn_id"])
            l3 = feats(y, max(1, W - 3), W - 1)[SHARES].add_suffix("_l3")
            f = std.join(l3).reset_index()
            f["season"], f["week"] = y, W
            parts.append(f)
    F = pd.concat(parts, ignore_index=True).drop(columns=["ppg", "gp"])
    x = x.merge(F, on=["season", "week", "pff_id"], how="left")
    P = []
    for y in SEASONS:
        for W in range(3, 14):
            p = window_ppg(y, 1, W - 1).reset_index()
            p["season"], p["week"] = y, W
            P.append(p)
    x = x.merge(pd.concat(P), on=["season", "week", "espn_id"], how="left")
    x["pos"] = x["position"]
    return x


# ------------------------------------------------------------------ 2026 table

def rows_2026() -> pd.DataFrame:
    f = window_table(2026, 1, 6)
    f = f.join(feats(2026, 4, 6)[SHARES].add_suffix("_l3")).join(feats(2026, 1, 3)[SHARES].add_suffix("_f3"))
    led = pd.read_csv(MAIN / "history" / "ledger" / "2026" / "wk06_players.csv")[["espn_id", "est_ppg", "status", "ecr_rank"]]
    f = f.merge(led, on="espn_id", how="left").set_index(f.index)
    ro = pd.read_csv(MAIN / "history" / "actuals" / "2026" / "rosters_by_week.csv")
    ro = ro[ro["week"] == ro["week"].max()].drop_duplicates("espn_id").set_index("espn_id")["team_id"]
    names = json.loads((MAIN / "history" / "actuals" / "2026" / "settings.json").read_text())["team_names"]
    tid = f["espn_id"].map(ro)
    f["owner"] = np.where(tid == MY_TEAM, "MAX", np.where(tid.notna(), tid.map(lambda t: names.get(str(int(t)), "") if pd.notna(t) else ""), "FA"))
    f.loc[f["espn_id"].isna(), "owner"] = "?"
    return f


# ------------------------------------------------------------------ A. usage model + flags

USAGE = {"WR": ["route_share", "tgt_share", "tprr", "targets_pg", "adot", "air_pg"],
         "RB": ["rush_pg", "rush_share", "targets_pg", "route_share", "tgt_share"]}
USAGE["TE"] = USAGE["WR"]
EFF = {"WR": ["yprr", "rec_grade"], "TE": ["yprr", "rec_grade"], "RB": ["rush_grade", "elusive", "yco"]}
ROLE = {"WR": "route_share", "TE": "route_share", "RB": "rush_share"}


def usage_model():
    return make_pipeline(SimpleImputer(strategy="median"),
                         HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=40, random_state=0))


def fit_usage(H: pd.DataFrame, T: pd.DataFrame):
    """Expected ppg from usage only. LOSO out-of-sample for history H, fit on all of H for 2026 table T."""
    H["exp_ppg"], T["exp_ppg"] = np.nan, np.nan
    for pos, cols in USAGE.items():
        hp = H["elig"] & (H["pos"] == pos) & H["ppg"].notna()
        for s in SEASONS:
            tr, te = hp & (H["season"] != s), hp & (H["season"] == s)
            m = usage_model().fit(H.loc[tr, cols], H.loc[tr, "ppg"], histgradientboostingregressor__sample_weight=H.loc[tr, "gp"])
            H.loc[te, "exp_ppg"] = m.predict(H.loc[te, cols])
        m = usage_model().fit(H.loc[hp, cols], H.loc[hp, "ppg"], histgradientboostingregressor__sample_weight=H.loc[hp, "gp"])
        tp = T["elig"] & (T["pos"] == pos)
        T.loc[tp, "exp_ppg"] = m.predict(T.loc[tp, cols])
    for d in (H, T):
        d["res"] = d["ppg"] - d["exp_ppg"]
        d["role_d"] = np.nan
        for pos, c in ROLE.items():
            k = d["pos"] == pos
            d.loc[k, "role_d"] = (d.loc[k, c + "_l3"] - d.loc[k, c]).where(d.loc[k, "pff_games_l3"] >= 2)


def eff_z(d: pd.DataFrame, by) -> pd.Series:
    z = pd.Series(np.nan, index=d.index)
    for pos, cols in EFF.items():
        k = d["elig"] & (d["pos"] == pos)
        g = d[k]
        zz = [(g[c] - g.groupby(by)[c].transform("mean")) / g.groupby(by)[c].transform("std") for c in cols] if by else \
             [(g[c] - g[c].mean()) / g[c].std() for c in cols]
        z[k] = pd.concat(zz, axis=1).mean(axis=1)
    return z


def boot(df: pd.DataFrame, flag: pd.Series, col: str, B: int = 2000):
    """Cluster (player-season) bootstrap of the future_played-weighted mean of col for flagged rows and of
    flagged minus unflagged. Returns (mean_f, lo, hi, diff, dlo, dhi)."""
    cl, _ = pd.factorize(df["season"].astype(str) + "_" + df["espn_id"].astype(str))
    K = cl.max() + 1
    w, v, f = df["future_played"].to_numpy(float), df[col].to_numpy(float), flag.to_numpy(bool)
    swf, svf = np.bincount(cl, w * f, K), np.bincount(cl, w * v * f, K)
    swu, svu = np.bincount(cl, w * ~f, K), np.bincount(cl, w * v * ~f, K)
    idx = RNG.integers(0, K, (B, K))
    cnt = np.zeros((B, K))
    np.add.at(cnt, (np.repeat(np.arange(B), K), idx.ravel()), 1)
    mf, mu = cnt @ svf / (cnt @ swf), cnt @ svu / (cnt @ swu)
    pf, pu = svf.sum() / swf.sum(), svu.sum() / swu.sum()
    return pf, *np.nanpercentile(mf, [2.5, 97.5]), pf - pu, *np.nanpercentile(mf - mu, [2.5, 97.5])


def flag_defs(H: pd.DataFrame):
    """Historical cutoffs (pooled per position) so the same rule applies to 2026. Returns {name: (fn, sign)}
    where fn(d) -> bool mask and sign is the direction a useful flag should move future - live."""
    cut = {}
    for pos in USAGE:
        g = H[H["elig"] & (H["pos"] == pos)]
        cut[pos] = {"res_lo": g["res"].quantile(0.1), "res_hi": g["res"].quantile(0.9),
                    "role_lo": g["role_d"].quantile(0.1), "role_hi": g["role_d"].quantile(0.9),
                    "eff_lo": g["eff_z"].quantile(0.1), "eff_hi": g["eff_z"].quantile(0.9)}

    def mk(key, op):
        def fn(d):
            m = pd.Series(False, index=d.index)
            for pos, c in cut.items():
                k = d["elig"] & (d["pos"] == pos)
                col = {"res": "res", "role": "role_d", "eff": "eff_z"}[key.split("_")[0]]
                m |= k & (d[col] <= c[key] if op == "le" else d[col] >= c[key])
            return m
        return fn
    return cut, {"usage>production (res bottom 10%)": (mk("res_lo", "le"), +1),
                 "production>usage (res top 10%)": (mk("res_hi", "ge"), -1),
                 "role riser (L3 - STD share top 10%)": (mk("role_hi", "ge"), +1),
                 "role faller (bottom 10%)": (mk("role_lo", "le"), -1),
                 "efficiency high (z top 10%)": (mk("eff_hi", "ge"), +1),
                 "efficiency low (z bottom 10%)": (mk("eff_lo", "le"), -1)}


def flag_history(H: pd.DataFrame, defs) -> pd.DataFrame:
    E = H[H["elig"] & H["pos"].isin(list(USAGE)) & (H["future_played"] >= 3) & H["week"].between(5, 13)].copy()
    E["d_est"], E["d_live"] = E["future_ppg"] - E["est_ppg"], E["future_ppg"] - E["live"]
    out = []
    for name, (fn, sign) in defs.items():
        fl = fn(E)
        for pos in ["RB", "WR", "TE"]:
            k = E["pos"] == pos
            g, f = E[k], fl[k]
            if "role" in name:
                g, f = g[g["week"] >= 6], f[g["week"] >= 6]
            if f.sum() < 10:
                continue
            est = boot(g, f, "d_est")
            liv = boot(g, f, "d_live")
            seas = sum(np.sign(np.average(gg.loc[f[gg.index], "d_live"], weights=gg.loc[f[gg.index], "future_played"])
                               - np.average(gg.loc[~f[gg.index], "d_live"], weights=gg.loc[~f[gg.index], "future_played"])) == sign
                       for _, gg in g.groupby("season") if f[gg.index].sum() >= 3)
            pred = (liv[4] > 0 if sign > 0 else liv[5] < 0)
            out.append({"flag": name, "pos": pos, "rows": int(f.sum()),
                        "players": g.loc[f, ["season", "espn_id"]].drop_duplicates().shape[0],
                        "fut-est": est[0], "fut-est CI": f"[{est[1]:+.2f}, {est[2]:+.2f}]",
                        "fut-live": liv[0], "fut-live CI": f"[{liv[1]:+.2f}, {liv[2]:+.2f}]",
                        "vs unflagged": liv[3], "vs unflagged CI": f"[{liv[4]:+.2f}, {liv[5]:+.2f}]",
                        "seasons right sign": seas, "label": "historically predictive" if pred else "descriptive only"})
    return pd.DataFrame(out)


def res_slopes(H: pd.DataFrame) -> pd.DataFrame:
    """Continuous version: weighted slope of (future_ppg - live) on the usage residual, cluster-bootstrap CI."""
    E = H[H["elig"] & H["pos"].isin(list(USAGE)) & (H["future_played"] >= 3) & H["week"].between(5, 13) & H["res"].notna()].copy()
    E["d_live"] = E["future_ppg"] - E["live"]
    out = []
    for pos, g in E.groupby("pos"):
        cl, _ = pd.factorize(g["season"].astype(str) + "_" + g["espn_id"].astype(str))
        x, y, w = g["res"].to_numpy(), g["d_live"].to_numpy(), g["future_played"].to_numpy(float)

        def slope(i):
            xm, ym = np.average(x[i], weights=w[i]), np.average(y[i], weights=w[i])
            return np.sum(w[i] * (x[i] - xm) * (y[i] - ym)) / np.sum(w[i] * (x[i] - xm) ** 2)
        groups = [np.flatnonzero(cl == c) for c in range(cl.max() + 1)]
        bs = [slope(np.concatenate([groups[j] for j in RNG.integers(0, len(groups), len(groups))])) for _ in range(500)]
        out.append({"pos": pos, "slope": slope(np.arange(len(g))), "CI": f"[{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]",
                    "res_sd": np.sqrt(np.cov(x, aweights=w))})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ B. stability

STATS = {
    "WR": ["route_share", "tgt_share", "tprr", "targets_pg", "yprr", "adot", "slot_rate", "yac_rec", "drop_rate",
           "rec_grade", "route_grade", "rec_td_pg", "ppg"],
    "RB": ["rush_share", "rush_pg", "targets_pg", "route_share", "ypa", "yco", "elusive", "breakaway", "rush_grade",
           "rush_td_pg", "ppg"],
    "QB": ["db_pg", "rush_pg", "scr_pg", "adot_pass", "btt", "twp", "sack_pct", "ttt", "acc", "pass_grade", "ypa_pass", "ppg"],
}
STATS["TE"] = STATS["WR"] + ["inline_rate"]


def stability() -> pd.DataFrame:
    rows = []
    for y in SEASONS:
        a, b = window_table(y, 1, 6), window_table(y, 7, 12)
        bl = [(window_table(y, 4, 6), window_table(y, 7, 9)), (window_table(y, 7, 9), window_table(y, 10, 12))]
        for pos, cols in STATS.items():
            A = a[eligible(a) & (a["pos"] == pos)]
            J = A.join(b[eligible(b)], rsuffix="_b", how="inner")
            for c in cols:
                blk = []
                for p, q in bl:
                    P = p[eligible(p, 2) & (p["pos"] == pos)].join(q[eligible(q, 2)], rsuffix="_b", how="inner")
                    blk.append(P[c].corr(P[c + "_b"], method="spearman"))
                rows.append({"season": y, "pos": pos, "stat": c, "n": len(J),
                             "split_half": J[c].corr(J[c + "_b"], method="spearman"), "block3": np.nanmean(blk),
                             "same_ppg": A[c].corr(A["ppg"], method="spearman"),
                             "next_ppg": J[c].corr(J["ppg_b"], method="spearman")})
    S = pd.DataFrame(rows)
    return S.groupby(["pos", "stat"]).agg(n=("n", "mean"), split_half=("split_half", "mean"), block3=("block3", "mean"),
                                          same_ppg=("same_ppg", "mean"), next_ppg=("next_ppg", "mean"),
                                          split_half_min=("split_half", "min")).reset_index()


# ------------------------------------------------------------------ C. ML on future_ppg - live

FE = {"WR": ["route_share", "tgt_share", "tprr", "yprr", "adot", "slot_rate", "yac_rec", "drop_rate", "rec_grade",
             "route_grade", "targets_pg", "rec_td_pg", "route_share_l3", "tgt_share_l3", "targets_pg_l3"],
      "RB": ["rush_share", "rush_pg", "ypa", "yco", "elusive", "breakaway", "rush_grade", "run_grade", "targets_pg",
             "route_share", "tgt_share", "yprr", "rush_td_pg", "rush_share_l3", "rush_pg_l3", "targets_pg_l3"],
      "QB": ["db_pg", "pass_grade", "btt", "twp", "adot_pass", "sack_pct", "ttt", "acc", "ypa_pass", "rush_pg", "scr_pg",
             "db_pg_l3", "rush_pg_l3"]}
FE["TE"] = FE["WR"] + ["inline_rate"]
BASE = ["live", "est_ppg", "ppg", "gp", "week"]


def _pipe(model, scale=True):
    steps = [SimpleImputer(strategy="median")] + ([StandardScaler()] if scale else []) + [model]
    return make_pipeline(*steps)


MODELS = {
    "ridge": [lambda a=a: _pipe(Ridge(alpha=a)) for a in (1, 10, 100, 1000, 10000)],
    "ridge_base": [lambda a=a: _pipe(Ridge(alpha=a)) for a in (1, 10, 100, 1000, 10000)],
    "gbm_base": [lambda d=d, lr=lr: _pipe(HistGradientBoostingRegressor(max_depth=d, learning_rate=lr, max_iter=150,
                                                                         min_samples_leaf=80, l2_regularization=1.0, random_state=0), False)
                 for d in (2, 3) for lr in (0.02, 0.06)],
    "lasso": [lambda a=a: _pipe(Lasso(alpha=a, max_iter=5000)) for a in (0.003, 0.01, 0.03, 0.1, 0.3)],
    "gbm": [lambda d=d, lr=lr: _pipe(HistGradientBoostingRegressor(max_depth=d, learning_rate=lr, max_iter=150,
                                                                    min_samples_leaf=80, l2_regularization=1.0, random_state=0), False)
            for d in (2, 3) for lr in (0.02, 0.06)],
    "rf": [lambda m=m: _pipe(RandomForestRegressor(n_estimators=150, min_samples_leaf=m, max_features=0.33, max_samples=0.5,
                                                   n_jobs=-1, random_state=0), False) for m in (30, 100, 300)],
    "mlp": [lambda h=h, a=a: _pipe(MLPRegressor(hidden_layer_sizes=h, alpha=a, early_stopping=True, validation_fraction=0.15,
                                                learning_rate_init=1e-3, max_iter=400, n_iter_no_change=15, random_state=0))
            for h in ((32,), (64, 32)) for a in (0.01, 0.1, 1.0, 10.0)],
}


def _fit(make, X, y, w):
    m = make()
    name = m.steps[-1][0]
    return m.fit(X, y, **{f"{name}__sample_weight": w})


def loso_model(g: pd.DataFrame, cols, configs) -> tuple[pd.Series, list]:
    """Outer LOSO by season; config chosen by weighted MSE over inner LOSO folds of the training seasons."""
    X, y, w = g[cols], g["future_ppg"] - g["live"], g["future_played"].to_numpy(float)
    pred, chosen = pd.Series(np.nan, index=g.index), []
    for s in SEASONS:
        tr = (g["season"] != s).to_numpy()
        inner = [t for t in SEASONS if t != s]
        best, best_i = np.inf, 0
        if len(configs) > 1:
            for i, mk in enumerate(configs):
                err = 0.0
                for t in inner:
                    a, b = tr & (g["season"] != t).to_numpy(), (g["season"] == t).to_numpy()
                    p = _fit(mk, X[a], y[a], w[a]).predict(X[b])
                    err += np.sum(w[b] * (y[b] - p) ** 2)
                if err < best:
                    best, best_i = err, i
        chosen.append(best_i)
        pred[~tr] = _fit(configs[best_i], X[tr], y[tr], w[tr]).predict(X[~tr])
    return pred, chosen


def score(g: pd.DataFrame, p: pd.Series):
    rc = {s: es.rank_corr(gg.assign(p=p[gg.index]), "p", "future_ppg", ("season", "week")) for s, gg in g.groupby("season")}
    mae = {s: float(np.average((p[gg.index] - gg["future_ppg"]).abs(), weights=gg["future_played"])) for s, gg in g.groupby("season")}
    return rc, mae


def ml_table(H: pd.DataFrame, only):
    D = H[H["future_played"] >= 3].copy()
    lines, preds = [], {}
    for pos in ["QB", "RB", "WR", "TE"]:
        g = D[D["position"] == pos].copy()
        base = {k: score(sub, sub["live"]) for k, sub in (("all", g), ("est>5", g[g["est_ppg"] > 5]))}
        for name in only:
            configs = MODELS[name]
            cols = BASE if name.endswith("_base") else BASE + FE[pos]
            p, chosen = loso_model(g, cols, configs)
            preds[(pos, name)] = p
            for k, sub in (("all", g), ("est>5", g[g["est_ppg"] > 5])):
                rc, mae = score(sub, sub["live"] + p[sub.index])
                brc, bmae = base[k]
                drc = {s: rc[s] - brc[s] for s in SEASONS}
                dmae = {s: mae[s] - bmae[s] for s in SEASONS}
                lines.append({"pos": pos, "rows": k, "model": name, "n": len(sub), "base_rc": np.mean(list(brc.values())),
                              "d_rc": np.mean(list(drc.values())), "rc_up": sum(v > 0 for v in drc.values()),
                              "d_mae": np.mean(list(dmae.values())), "mae_down": sum(v < 0 for v in dmae.values()),
                              "d_rc_by_season": " ".join(f"{v:+.3f}" for v in drc.values()), "configs": chosen})
            print(f"  ml {pos} {name} done {pd.Timestamp.now():%H:%M:%S}", flush=True)
    return pd.DataFrame(lines), preds


# ------------------------------------------------------------------ archetypes

PROFILE = {"WR": ["route_share", "tgt_share", "tprr", "yprr", "adot", "slot_rate", "yac_rec", "rec_grade", "rec_td_pg"],
           "TE": ["route_share", "tgt_share", "tprr", "yprr", "adot", "slot_rate", "inline_rate", "yac_rec", "rec_grade"],
           "RB": ["rush_share", "rush_pg", "targets_pg", "route_share", "ypa", "yco", "elusive", "breakaway", "rush_grade"],
           "QB": ["db_pg", "rush_pg", "adot_pass", "btt", "twp", "sack_pct", "ttt", "acc", "pass_grade"]}


def archetypes(T: pd.DataFrame):
    """Fit on 2021-25 weeks 1-12 profiles (eligible, >= 4 games); assign 2026 weeks 1-6 profiles."""
    out, assign, figs = [], [], {}
    for pos, cols in PROFILE.items():
        hist = []
        for y in SEASONS:
            a = window_table(y, 1, 12)
            a = a[eligible(a, 4) & (a["pos"] == pos)].copy()
            a = a.join(window_ppg(y, 13, 17).rename(columns={"ppg": "ppg_late", "gp": "gp_late"}), on="espn_id")
            hist.append(a.assign(season=y))
        A = pd.concat(hist).dropna(subset=cols)
        sc = StandardScaler().fit(A[cols])
        Z = sc.transform(A[cols])
        pca = PCA(n_components=0.85, random_state=0).fit(Z)
        P = pca.transform(Z)
        ks = {k: silhouette_score(P, KMeans(k, n_init=20, random_state=0).fit_predict(P)) for k in range(3, 7)}
        k = max(ks, key=ks.get)
        km = KMeans(k, n_init=20, random_state=0).fit(P)
        A["cl"] = km.labels_
        cz = pd.DataFrame(Z, columns=cols, index=A.index).groupby(A["cl"].to_numpy()).mean()
        t = T[T["elig"] & (T["pos"] == pos)].dropna(subset=cols).copy()
        t["cl"] = km.predict(pca.transform(sc.transform(t[cols])))
        pt = pca.transform(sc.transform(t[cols]))
        t["pc1"], t["pc2"] = pt[:, 0], pt[:, 1] if pt.shape[1] > 1 else 0
        assign.append(t)
        for c in range(k):
            zc = cz.loc[c].sort_values(key=abs, ascending=False)
            ex = t[t["cl"] == c].sort_values("ppg", ascending=False)["player"].head(5).tolist()
            m = A[A["cl"] == c]
            out.append({"pos": pos, "cluster": c, "k": k, "silhouette": round(ks[k], 3), "hist_n": len(m), "n_2026": int((t["cl"] == c).sum()),
                        "profile": ", ".join(f"{f} {v:+.1f}" for f, v in zc.head(4).items()),
                        "hist_ppg_wk1_12": m["ppg"].mean(), "hist_ppg_wk13_17": m["ppg_late"].mean(),
                        "late_minus_early": (m["ppg_late"] - m["ppg"]).mean(), "examples_2026": ", ".join(ex)})
        figs[pos] = (pca.explained_variance_ratio_[:2].sum(), t)
    return pd.DataFrame(out), pd.concat(assign), figs


# ------------------------------------------------------------------ report

def main():
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.offline import get_plotlyjs

    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 300)
    REPORT.mkdir(parents=True, exist_ok=True)
    print("building rows ...", flush=True)
    H = hist_rows()
    H["elig"] = eligible(H)
    T = rows_2026()
    T["elig"] = eligible(T)
    print(f"2026: {len(T)} PFF players, {T['espn_id'].notna().sum()} matched to ESPN, {T['elig'].sum()} eligible")
    fit_usage(H, T)
    H["eff_z"] = eff_z(H, ["season", "week"])
    T["eff_z"] = eff_z(T, None)
    cut, defs = flag_defs(H)
    FH = flag_history(H, defs)
    SL = res_slopes(H)
    print("\n== A. historical check of each flag (future - live, ppg; weeks 5-13, future_played >= 3)")
    print(FH.round(3).to_string(index=False))
    print("\n== A. slope of future-live on usage residual")
    print(SL.round(3).to_string(index=False))

    show = ["player", "pos", "team", "owner", "status", "gp", "ppg", "exp_ppg", "res", "est_ppg", "route_share", "tgt_share",
            "tprr", "rush_share", "targets_pg", "route_share_l3", "rush_share_l3", "role_d", "yprr", "rec_grade", "rush_grade",
            "elusive", "yco", "eff_z", "rec_td_pg", "rush_td_pg"]
    E = T[T["elig"] & T["pos"].isin(list(USAGE))]
    lists = {
        "1. Usage > production (buy-low / waiver)": E[(E["exp_ppg"] >= 7) & (E["status"] != "ir")].sort_values("res").head(25),
        "2. Production > usage (sell-high)": E[E["ppg"] >= 9].sort_values("res", ascending=False).head(25),
        "3a. Role risers (last 3 vs season)": E[E["status"] != "ir"].sort_values("role_d", ascending=False).head(20),
        "3b. Role fallers": E.sort_values("role_d").head(20),
        "4a. Efficiency high": E.sort_values("eff_z", ascending=False).head(20),
        "4b. Efficiency low": E.sort_values("eff_z").head(20),
    }
    for k, v in lists.items():
        v[show].to_csv(REPORT / f"list_{k[:2].strip('. ')}.csv", index=False)

    print("\n== B. stability", flush=True)
    S = stability()
    print(S.sort_values(["pos", "split_half"], ascending=[True, False]).round(3).to_string(index=False))

    T["who"] = np.where(T["owner"].isin(["MAX", "FA", "?"]), T["owner"], "rostered")
    print("\n== C. archetypes", flush=True)
    CL, CA, cfigs = archetypes(T)
    print(CL.round(2).drop(columns=["examples_2026"]).to_string(index=False))
    CL.to_csv(REPORT / "clusters.csv", index=False)

    print("\n== C. ML (future - live)", flush=True)
    ml_path = REPORT / "ml.csv"
    old = pd.read_csv(ml_path) if "--reuse-ml" in sys.argv and ml_path.exists() else pd.DataFrame(columns=["model"])
    todo = [m for m in MODELS if m not in set(old["model"])]
    ML = pd.concat([old, ml_table(H, todo)[0]], ignore_index=True) if todo else old
    print(ML.drop(columns=["configs"]).round(4).to_string(index=False))
    ML.to_csv(REPORT / "ml.csv", index=False)
    FH.to_csv(REPORT / "flags.csv", index=False)
    S.to_csv(REPORT / "stability.csv", index=False)
    SL.to_csv(REPORT / "slopes.csv", index=False)

    # ---------------- HTML
    COL = {"MAX": "#d62728", "FA": "#2ca02c", "?": "#bbbbbb"}
    cmap = {**COL, "rostered": "#7f7f7f"}
    figs = []

    def add(title, fig, note=""):
        fig.update_layout(template="plotly_white", height=460, margin=dict(l=40, r=20, t=50, b=40), title=title)
        figs.append(f"<div class='fig'>{pio.to_html(fig, include_plotlyjs=False, full_html=False)}<p class='note'>{note}</p></div>")

    E = T[T["elig"] & T["pos"].isin(list(USAGE))].copy()
    f = px.scatter(E, x="exp_ppg", y="ppg", color="who", facet_col="pos", color_discrete_map=cmap,
                   hover_data=["player", "team", "owner", "route_share", "tgt_share", "rush_share", "targets_pg", "est_ppg"])
    for i in range(1, 4):
        f.add_shape(type="line", x0=0, y0=0, x1=25, y1=25, line=dict(dash="dot", color="#999"), row=1, col=i)
    add("2026 wk 1-6: actual ppg vs usage-expected ppg", f,
        "Below the diagonal = usage > production (buy-low); above = production > usage (sell-high). Expected ppg is a GBM on usage only "
        "(route/target share, targets per route, targets/game, aDOT for WR/TE; carries, rush share, targets, route share for RB) fit on 2021-25.")
    f = px.scatter(E, x=E["route_share"].where(E["pos"] != "RB", E["rush_share"]),
                   y=E["route_share_l3"].where(E["pos"] != "RB", E["rush_share_l3"]), color="who", facet_col="pos",
                   color_discrete_map=cmap, hover_data=["player", "team", "owner"], labels={"x": "share wk 1-6", "y": "share wk 4-6"})
    for i in range(1, 4):
        f.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dot", color="#999"), row=1, col=i)
    add("Role movers: route share (WR/TE) or rush share (RB), last 3 vs season", f)
    fh = FH.copy()
    fh["err_lo"], fh["err_hi"] = [fh["vs unflagged"] - fh["vs unflagged CI"].str.extract(r"\[([-+\d.]+),")[0].astype(float),
                                  fh["vs unflagged CI"].str.extract(r", ([-+\d.]+)\]")[0].astype(float) - fh["vs unflagged"]]
    f = px.bar(fh, x="flag", y="vs unflagged", color="pos", barmode="group", error_y="err_hi", error_y_minus="err_lo")
    add("Historical check: flagged minus unflagged mean of (future ppg - live forecast), 95% cluster-bootstrap CI", f,
        "2021-25 checkpoints weeks 5-13. A useful buy flag is above zero, a sell flag below. Error bars crossing zero = descriptive only.")
    S2 = S[S["stat"] != "ppg"]
    f = px.bar(S.sort_values("split_half"), x="split_half", y="stat", color="block3", facet_col="pos", orientation="h",
               color_continuous_scale="Viridis", facet_col_spacing=0.06)
    f.update_yaxes(matches=None, showticklabels=True)
    f.update_layout(height=620)
    add("Stability: Spearman r, weeks 1-6 vs weeks 7-12 (mean over 2021-25); color = adjacent 3-week block r", f)
    f = px.scatter(S2, x="split_half", y="next_ppg", color="pos", text="stat", hover_data=["same_ppg", "block3", "n"])
    f.update_traces(textposition="top center", textfont_size=9)
    f.update_layout(height=600)
    add("Sticky and useful? split-half stability vs Spearman r with fantasy ppg in weeks 7-12", f,
        "Top right = stable and tied to future points. Note ppg itself sits at about 0.5-0.6 for skill players.")
    mlp = ML.copy()
    f = px.bar(mlp, x="model", y="d_rc", color="pos", barmode="group", facet_col="rows", hover_data=["rc_up", "d_mae", "d_rc_by_season"])
    f.add_hrect(y0=0.008, y1=0.019, fillcolor="green", opacity=0.12, line_width=0)
    add("ML residual models: change in within-week Spearman vs the live blend (LOSO by season)", f,
        "Green band = what adding FantasyPros ECR to est_ppg was worth (+0.008 to +0.019). Models target future_ppg - live, so any gain is new information.")
    for pos, (ev, t) in cfigs.items():
        f = px.scatter(t, x="pc1", y="pc2", color=t["cl"].astype(str), symbol="who", hover_data=["player", "team", "owner", "ppg"])
        add(f"{pos} archetypes (2026 wk 1-6, k-means on PCA of the per-position PFF profile; PC1+PC2 = {ev:.0%} of variance)", f)

    def tab(df, cols=None, n=None):
        d = df if cols is None else df[[c for c in cols if c in df.columns]]
        d = d.head(n) if n else d
        return d.to_html(index=False, float_format=lambda v: f"{v:.2f}", classes="t", na_rep="")

    lcols = {"1": ["player", "pos", "team", "owner", "status", "gp", "ppg", "exp_ppg", "res", "est_ppg", "route_share", "tgt_share", "tprr", "rush_share", "targets_pg"],
             "2": ["player", "pos", "team", "owner", "status", "gp", "ppg", "exp_ppg", "res", "est_ppg", "rec_td_pg", "rush_td_pg", "route_share", "rush_share"],
             "3": ["player", "pos", "team", "owner", "status", "route_share", "route_share_l3", "rush_share", "rush_share_l3", "role_d", "ppg", "est_ppg"],
             "4": ["player", "pos", "team", "owner", "status", "eff_z", "yprr", "rec_grade", "rush_grade", "elusive", "yco", "route_share", "rush_share", "ppg", "est_ppg"]}
    lab = {"1": "usage>production", "2": "production>usage", "3": "role", "4": "efficiency"}
    html_lists = []
    for k, v in lists.items():
        fam = k[0]
        verdict = FH[FH["flag"].str.startswith(lab[fam].split()[0])][["flag", "pos", "label", "vs unflagged", "vs unflagged CI"]]
        html_lists.append(f"<h3>{k}</h3>{tab(v, lcols[fam], 15)}<p class='note'>Historical check:</p>{tab(verdict)}")
    css = ("body{font-family:-apple-system,Helvetica,sans-serif;max-width:1250px;margin:20px auto;padding:0 16px;color:#222}"
           "table.t{border-collapse:collapse;font-size:12px;margin:6px 0 14px}table.t td,table.t th{border:1px solid #ddd;padding:3px 6px}"
           "table.t th{background:#f3f3f3}.note{color:#555;font-size:13px}h2{border-bottom:2px solid #333;margin-top:40px}")
    body = f"""<h1>PFF Premium Stats: exploration (2026 wk 1-6, history 2021-25)</h1>
<p class='note'>Generated by backtest/pff_explore.py on 2026-10-16. Owner column: MAX = your team, FA = free agent, else the league team name
(ESPN rosters at week {pd.read_csv(MAIN / 'history/actuals/2026/rosters_by_week.csv')['week'].max()}). est_ppg is the week-6 ledger v2 forecast.
"live" in historical checks = est_ppg blended 1/3 with rank-matched FantasyPros ECR.</p>
{VERDICT}
<h2>A. 2026 actionable lists</h2>{''.join(figs[:3])}{''.join(html_lists)}
<h3>Continuous check: slope of (future - live) on the usage residual</h3>{tab(SL)}
<h3>All historical flag checks</h3>{tab(FH)}
<h2>B. Stability and links to fantasy points</h2>{''.join(figs[3:5])}{tab(S.sort_values(['pos', 'split_half'], ascending=[True, False]))}
<h2>C. ML models of future_ppg - live</h2>{figs[5]}{tab(ML.drop(columns=['configs']))}
<h2>C. Archetypes</h2>{''.join(figs[6:])}{tab(CL)}"""
    (REPORT / "index.html").write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>PFF exploration</title>"
                                       f"<script>{get_plotlyjs()}</script><style>{css}</style></head><body>{body}</body></html>")
    print(f"\nreport: {REPORT / 'index.html'}")


VERDICT = """<h2>Verdict</h2><ul>
<li><b>Lists are descriptive, not edges.</b> On 2021-25 none of the six flags (usage&gt;production, production&gt;usage, role riser/faller,
efficiency high/low) moves future ppg against the live forecast reliably. The only CI that clears zero is WR role risers
(+0.54 ppg vs unflagged, CI +0.01 to +1.09, 5/5 seasons), and with 18 tests one marginal hit is what chance gives you. The continuous
usage-residual slope is about 0 at every position. v2 already absorbs usage through xfp.</li>
<li><b>Sticky stats:</b> WR slot rate, targets/game, target share, TPRR; TE inline/slot rate, targets/game; RB carries/game, rush share,
targets/game; QB rush attempts and time to throw. <b>Noise:</b> drop rate, TD rates, YAC/rec, ypa, YCO, breakaway, elusive rating, TWP rate.
Grades sit in the middle (about 0.55 for WR, 0.3 for RB/TE).</li>
<li><b>ML:</b> no model beats the live blend for QB/RB/WR. The MLP is the worst everywhere (it overfits; early stopping validates on
rows from the same players). The one exception is linear models for TE with est&gt;5 (+0.025 rank corr, 5/5 seasons, MAE down 5/5), driven by
rec grade (+) and TD rate (-, TD regression). It hurts across all TE rows (-0.015) and isn't stable when trained on est&gt;5 only. Promising, not ready.</li>
<li><b>For the Desk:</b> worth a "usage vs production" scatter and a role-mover panel as context (clearly labeled descriptive). Show route share,
target share and TPRR on player cards. Don't feed grades or efficiency into valuation. Possible narrow follow-up: TE grade plus TD-rate regression.</li></ul>"""

if __name__ == "__main__":
    main()
