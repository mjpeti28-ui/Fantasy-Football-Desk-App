"""Does a usage/profile-aware statistical layer add information on top of the current v2 model?

Reuses the project's leave-one-season-out machinery: backtest/model_rows.py already holds the CURRENT
model's forecasts (equal blend of ours / ESPN / ECR, availability model) at every checkpoint W = 3..13
of 2021-2025, with realised points afterwards. This script bolts on nflverse usage and profile features
known before W, then tests whether any of them improves on est_ppg out of sample.

  python3 backtest/ml_signal.py build                 # dataset -> history/backtest/multi/ml_rows.csv
  python3 backtest/ml_signal.py reg                   # regression models, LOSO (+ forward chaining)
  python3 backtest/ml_signal.py clf                   # logistic / boosted classifiers for a 'hit' outcome
  python3 backtest/ml_signal.py decide                # waiver replay + trade-direction tests
  python3 backtest/ml_signal.py placebo               # shuffled-feature noise floor
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import nflverse_data as nd  # noqa: E402
import valuation as v  # noqa: E402

OUT = ROOT / "history" / "backtest" / "multi"
SEASONS = [2021, 2022, 2023, 2024, 2025]
SKILL = ["QB", "RB", "WR", "TE"]
CHECKS = range(3, 14)
LAST = v.FANTASY_LAST_WEEK


# ----------------------------------------------------------------------------------------------- build
def usage_feats(U: pd.DataFrame, W: int) -> pd.DataFrame:
    g = U[(U["week"] < W) & ((U["offense_pct"].fillna(0) > 0) | U["fantasy_points_ppr"].notna())].copy()
    g = g.sort_values(["espn_id", "week"])
    g["opp"] = g["carries"].fillna(0) + g["targets"].fillna(0)
    l3 = g.groupby("espn_id").tail(3)
    l2 = g.groupby("espn_id").tail(2)
    gb = g.groupby("espn_id")
    return pd.DataFrame({
        "u_games": gb.size(),
        "tpg": gb["targets"].mean(),
        "cpg": gb["carries"].mean(),
        "apg": gb["attempts"].mean(),
        "tsh": gb["target_share"].mean(),
        "ays": gb["air_yards_share"].mean(),
        "wopr": gb["wopr"].mean(),
        "snap_pg": gb["offense_pct"].mean(),
        "opp_pg": gb["opp"].mean(),
        "xfp_pg": gb["xfp"].mean(),
        "fp_pg": gb["fantasy_points_ppr"].mean(),
        "xfp_l3": l3.groupby("espn_id")["xfp"].mean(),
        "tsh_l3": l3.groupby("espn_id")["target_share"].mean(),
        "snap_l2": l2.groupby("espn_id")["offense_pct"].mean(),
        "fp_l3": l3.groupby("espn_id")["fantasy_points_ppr"].mean(),
    })


def build_season(season: int, ids: pd.DataFrame) -> pd.DataFrame:
    m = pd.read_csv(OUT / f"model_rows_{season}.csv")
    m = m[m["position"].isin(SKILL) & (m["est_ppg"] > 0)].copy()
    U = nd.usage_table(season)
    ur = pd.read_csv(OUT / f"usage_rows_{season}.csv", usecols=["espn_id", "week", "rostered", "added_now"])
    ur = ur.rename(columns={"week": "W"})
    parts = []
    for W, g in m.groupby("W"):
        f = usage_feats(U, W)
        gg = g.merge(f, left_on="espn_id", right_index=True, how="left")
        parts.append(gg)
    d = pd.concat(parts, ignore_index=True)
    d = d.merge(ur, on=["espn_id", "W"], how="left")
    d["rostered"] = d["rostered"].fillna(False).astype(bool)
    d["added_now"] = d["added_now"].fillna(False).astype(bool)
    # outcomes
    pts = np.column_stack([d[f"pts_w{w}"] if f"pts_w{w}" in d else np.zeros(len(d)) for w in range(3, LAST + 1)])
    pl = np.column_stack([d[f"play_w{w}"] if f"play_w{w}" in d else np.zeros(len(d)) for w in range(3, LAST + 1)])
    xa = np.column_stack([d[f"xa_w{w}"] if f"xa_w{w}" in d else np.zeros(len(d)) for w in range(3, LAST + 1)])
    wk = np.arange(3, LAST + 1)[None, :]
    live = wk >= d["W"].to_numpy()[:, None]
    pts = np.where(live & np.isfinite(pts), pts, 0.0)
    pl = np.where(live & np.isfinite(pl), pl, 0.0)
    xa = np.where(live & np.isfinite(xa), xa, 0.0)
    d["future_points"] = pts.sum(1)
    d["future_played"] = pl.sum(1)
    d["exp_games"] = xa.sum(1)
    d["future_ppg"] = np.where(d["future_played"] > 0, (pts * pl).sum(1) / np.clip(d["future_played"], 1, None), np.nan)
    d["v2_total"] = d["est_ppg"] * d["exp_games"]
    # profile
    d = d.merge(ids, on="espn_id", how="left")
    d["rookie"] = (d["draft_year"] == season).astype(float)
    d["log_pick"] = np.log(d["draft_pick"].fillna(260).clip(1, 260))
    d["age"] = season - pd.to_datetime(d["birthdate"], errors="coerce").dt.year
    d["age"] = d["age"].fillna(d["age"].median())
    keep = ["season", "W", "espn_id", "name", "position", "est_ppg", "espn_now", "ecr_rank", "avail_now", "bye", "team_id",
            "rostered", "added_now", "future_points", "future_played", "future_ppg", "exp_games", "v2_total", "rookie",
            "log_pick", "age"] + [c for c in usage_feats(U, 5).columns]
    return d[keep]


def build():
    ids = pd.read_csv(ROOT / "history" / "nflverse" / "ids.csv", usecols=["espn_id", "draft_year", "draft_pick", "birthdate"])
    ids = ids.dropna(subset=["espn_id"]).drop_duplicates("espn_id")
    ids["espn_id"] = ids["espn_id"].astype("int64")
    out = pd.concat([build_season(s, ids) for s in SEASONS], ignore_index=True)
    out.to_csv(OUT / "ml_rows.csv", index=False)
    print(len(out), "rows;", out.groupby("season").size().to_dict())
    print("usage coverage", out["u_games"].notna().mean().round(3), " rookies", int(out["rookie"].sum()))


# ----------------------------------------------------------------------------------------------- modelling
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

USAGE = ["tpg", "cpg", "tsh", "ays", "wopr", "snap_pg", "snap_l2", "opp_pg"]
XFP = ["xfp_pg", "xfp_l3", "gap"]
PROFILE = ["rookie", "log_pick", "age", "u_games"]
MKT = ["espn_now_f", "ecr_f"]
FEATS = {
    "base": ["est_ppg"],
    "mkt": ["est_ppg"] + MKT,
    "xfp": ["est_ppg"] + XFP,
    "usage": ["est_ppg"] + USAGE,
    "profile": ["est_ppg"] + PROFILE,
    "full": ["est_ppg"] + XFP + USAGE + PROFILE,
}


def load() -> pd.DataFrame:
    d = pd.read_csv(OUT / "ml_rows.csv")
    has = d["u_games"].fillna(0) > 0
    for c in ["tpg", "cpg", "apg", "tsh", "ays", "wopr", "snap_pg", "snap_l2", "opp_pg", "tsh_l3"]:
        d[c] = d[c].fillna(0.0)
    for c in ["xfp_pg", "xfp_l3", "fp_pg", "fp_l3"]:
        d[c] = d[c].where(has & d[c].notna(), d["est_ppg"])
    d["gap"] = d["fp_pg"] - d["xfp_pg"]
    d["u_games"] = d["u_games"].fillna(0.0)
    d["espn_now_f"] = d["espn_now"].fillna(d["est_ppg"])
    d["ecr_f"] = np.log(d["ecr_rank"].fillna(150.0))
    return d


def spear(a, b):
    return pd.Series(a).corr(pd.Series(b), method="spearman")


def fit_reg(tr: pd.DataFrame, cols, kind="ols", alpha=None):
    """Per-position weighted linear fit; returns dict pos -> (scaler, model)."""
    out = {}
    for pos in SKILL:
        a = tr[tr["position"] == pos]
        sc = StandardScaler().fit(a[cols])
        m = Ridge(alpha=1e-6 if kind == "ols" else alpha)
        m.fit(sc.transform(a[cols]), a["future_ppg"], sample_weight=a["future_played"])
        out[pos] = (sc, m)
    return out


def pred_reg(mod, te: pd.DataFrame, cols):
    p = np.full(len(te), np.nan)
    for pos, (sc, m) in mod.items():
        b = (te["position"] == pos).to_numpy()
        if b.any():
            p[b] = m.predict(sc.transform(te.loc[b, cols]))
    return np.clip(p, 0, None)


def inner_ridge_alpha(tr, cols, grid=(1, 10, 100, 1000, 5000)):
    best, be = grid[0], 1e18
    for a in grid:
        err = 0.0
        for s in tr["season"].unique():
            m = fit_reg(tr[tr["season"] != s], cols, "ridge", a)
            te = tr[tr["season"] == s]
            err += float((te["future_played"] * (pred_reg(m, te, cols) - te["future_ppg"]) ** 2).sum())
        if err < be:
            best, be = a, err
    return best


def design_gb(df, cols):
    X = df[cols].copy()
    for p in SKILL:
        X["pos_" + p] = (df["position"] == p).astype(float)
    return X


GB_GRID = [dict(max_depth=2, max_iter=150, learning_rate=0.05, min_samples_leaf=80),
           dict(max_depth=3, max_iter=250, learning_rate=0.04, min_samples_leaf=60)]


def fit_gb(tr, cols, cfg):
    m = HistGradientBoostingRegressor(l2_regularization=1.0, random_state=0, **cfg)
    m.fit(design_gb(tr, cols), tr["future_ppg"], sample_weight=tr["future_played"])
    return m


def inner_gb_cfg(tr, cols):
    best, be = GB_GRID[0], 1e18
    for cfg in GB_GRID:
        err = 0.0
        for s in tr["season"].unique():
            m = fit_gb(tr[tr["season"] != s], cols, cfg)
            te = tr[tr["season"] == s]
            err += float((te["future_played"] * (np.clip(m.predict(design_gb(te, cols)), 0, None) - te["future_ppg"]) ** 2).sum())
        if err < be:
            best, be = cfg, err
    return best


def regression_predictions(d: pd.DataFrame, train_rule="loso", test_seasons=None, models=None) -> pd.DataFrame:
    d = d.copy()
    models = models or ["base", "mkt", "xfp", "usage", "profile", "full", "full_ridge", "full_gb"]
    for m in models:
        d["pred_" + m] = np.nan
    ok = (d["future_played"] >= 3)
    for held in (test_seasons or SEASONS):
        if train_rule == "loso":
            trs = [s for s in SEASONS if s != held]
        else:  # forward chaining
            trs = [s for s in SEASONS if s < held]
            if len(trs) < 2:
                continue
        tr = d[d["season"].isin(trs) & ok]
        te_mask = d["season"] == held
        te = d[te_mask]
        for m in models:
            if m in FEATS:
                mod = fit_reg(tr, FEATS[m], "ols")
                d.loc[te_mask, "pred_" + m] = pred_reg(mod, te, FEATS[m])
            elif m == "full_ridge":
                a = inner_ridge_alpha(tr, FEATS["full"])
                mod = fit_reg(tr, FEATS["full"], "ridge", a)
                d.loc[te_mask, "pred_" + m] = pred_reg(mod, te, FEATS["full"])
            elif m == "full_gb":
                cfg = inner_gb_cfg(tr, FEATS["full"])
                gb = fit_gb(tr, FEATS["full"], cfg)
                d.loc[te_mask, "pred_" + m] = np.clip(gb.predict(design_gb(te, FEATS["full"])), 0, None)
    return d


def group_rhos(d, models, mask=None):
    g0 = d[(d["future_played"] >= 3)]
    if mask is not None:
        g0 = g0[mask.loc[g0.index]]
    rows = []
    for (s, W, pos), g in g0.groupby(["season", "W", "position"]):
        if len(g) < 8:
            continue
        r = {"season": s, "W": W, "position": pos, "n": len(g)}
        for m in models:
            r["rho_" + m] = spear(g["pred_" + m], g["future_ppg"])
        rows.append(r)
    return pd.DataFrame(rows)


def boot_delta(gr, m, base="base", reps=2000, seed=0):
    x = (gr["rho_" + m] - gr["rho_" + base]).to_numpy()
    w = gr["n"].to_numpy(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (reps, len(x)))
    b = np.array([np.average(x[i], weights=w[i]) for i in idx])
    return np.average(x, weights=w), np.percentile(b, 2.5), np.percentile(b, 97.5)


def error_table(d, models):
    g = d[(d["future_played"] >= 3)]
    rows = []
    for m in models:
        e = g["pred_" + m] - g["future_ppg"]
        slope = np.polyfit(g["pred_" + m], g["future_ppg"], 1)[0]
        rows.append({"model": m, "MAE": e.abs().mean(), "RMSE": np.sqrt((e ** 2).mean()), "calib_slope": slope,
                     "R2": 1 - (e ** 2).sum() / ((g["future_ppg"] - g["future_ppg"].mean()) ** 2).sum()})
    return pd.DataFrame(rows).round(4)


def cmd_reg():
    d = load()
    models = ["base", "mkt", "xfp", "usage", "profile", "full", "full_ridge", "full_gb"]
    p = regression_predictions(d, "loso", models=models)
    p.to_csv(OUT / "ml_reg_preds.csv", index=False)
    pd.set_option("display.width", 220)
    print("== LOSO, all positions ==")
    print(error_table(p, models).to_string(index=False))
    gr = group_rhos(p, models)
    print("\nwithin-group Spearman (mean) and paired delta vs base [95% CI, groups bootstrap]:")
    for m in models:
        mu = np.average(gr["rho_" + m], weights=gr["n"])
        if m == "base":
            print(f"  {m:12s} {mu:.4f}")
        else:
            dm, lo, hi = boot_delta(gr, m)
            print(f"  {m:12s} {mu:.4f}  delta {dm:+.4f} [{lo:+.4f}, {hi:+.4f}]")


def cmd_segments():
    p = pd.read_csv(OUT / "ml_reg_preds.csv")
    models = ["base", "xfp", "usage", "full", "full_ridge", "full_gb"]
    segs = {
        "rostered": p["rostered"],
        "free agents": ~p["rostered"],
        "W 3-5": p["W"] <= 5,
        "W 6-9": p["W"].between(6, 9),
        "W 10-13": p["W"] >= 10,
        "rookies": p["rookie"] == 1,
        "veterans": p["rookie"] == 0,
        "no 2023": p["season"] != 2023,
        "WR": p["position"] == "WR", "RB": p["position"] == "RB", "TE": p["position"] == "TE", "QB": p["position"] == "QB",
        "low volume (est<8)": p["est_ppg"] < 8,
        "high volume (est>=8)": p["est_ppg"] >= 8,
    }
    rows = []
    for name, mask in segs.items():
        gr = group_rhos(p, models, mask)
        if len(gr) < 5:
            continue
        rec = {"segment": name, "groups": len(gr), "base_rho": np.average(gr["rho_base"], weights=gr["n"])}
        for m in models[1:]:
            dm, lo, hi = boot_delta(gr, m, reps=800)
            rec[m] = f"{dm:+.3f} [{lo:+.3f},{hi:+.3f}]"
        rows.append(rec)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("\nPer-season delta (full_ridge vs base, mean within-group rho):")
    gr = group_rhos(p, ["base", "full", "full_ridge", "full_gb"])
    t = gr.groupby("season").apply(lambda x: pd.Series({m: np.average(x["rho_" + m] - x["rho_base"], weights=x["n"]) for m in ["full", "full_ridge", "full_gb"]}))
    print(t.round(4).to_string())


def cmd_forward():
    d = load()
    models = ["base", "xfp", "usage", "full", "full_ridge", "full_gb"]
    p = regression_predictions(d, "forward", test_seasons=[2023, 2024, 2025], models=models)
    p = p[p["season"].isin([2023, 2024, 2025])]
    print("forward-chaining (train on earlier seasons only), test 2023-25")
    print(error_table(p, models).to_string(index=False))
    gr = group_rhos(p, models)
    for m in models[1:]:
        dm, lo, hi = boot_delta(gr, m)
        print(f"  {m:12s} delta {dm:+.4f} [{lo:+.4f}, {hi:+.4f}]")


# ----------------------------------------------------------------------------------------------- classification
from sklearn.metrics import log_loss, roc_auc_score  # noqa: E402


def ece(y, p, bins=10):
    q = pd.cut(p, np.linspace(0, 1, bins + 1), include_lowest=True)
    t = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q").agg(n=("y", "size"), y=("y", "mean"), p=("p", "mean")).dropna()
    return float((t["n"] * (t["y"] - t["p"]).abs()).sum() / t["n"].sum())


def fit_logit(tr, y, cols, C=1.0):
    out = {}
    for pos in SKILL:
        b = (tr["position"] == pos).to_numpy()
        sc = StandardScaler().fit(tr.loc[b, cols])
        m = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(tr.loc[b, cols]), y[b])
        out[pos] = (sc, m)
    return out


def pred_logit(mod, te, cols):
    p = np.full(len(te), np.nan)
    for pos, (sc, m) in mod.items():
        b = (te["position"] == pos).to_numpy()
        if b.any():
            p[b] = m.predict_proba(sc.transform(te.loc[b, cols]))[:, 1]
    return p


def inner_C(tr, y, cols, grid=(0.003, 0.03, 0.3, 3.0)):
    best, be = grid[0], 1e18
    for C in grid:
        ll = 0.0
        for s in tr["season"].unique():
            a = (tr["season"] != s).to_numpy()
            m = fit_logit(tr[a], y[a], cols, C)
            pp = np.clip(pred_logit(m, tr[~a], cols), 1e-6, 1 - 1e-6)
            yy = y[~a]
            ll += float(-(yy * np.log(pp) + (1 - yy) * np.log(1 - pp)).sum())
        if ll < be:
            best, be = C, ll
    return best


def make_target(task, tr, te):
    if task == "hit":     # top 35% of the position's pool by rest-of-season ppg (threshold from training seasons)
        thr = tr.groupby("position")["future_ppg"].quantile(0.65)
        return (tr["future_ppg"] >= tr["position"].map(thr)).astype(int).to_numpy(), (te["future_ppg"] >= te["position"].map(thr)).astype(int).to_numpy()
    if task == "surprise":  # does he beat what the model implies? (median ratio from training seasons)
        r = (tr["future_ppg"] / tr["est_ppg"]).groupby(tr["position"]).median()
        return (tr["future_ppg"] > tr["est_ppg"] * tr["position"].map(r)).astype(int).to_numpy(), (te["future_ppg"] > te["est_ppg"] * te["position"].map(r)).astype(int).to_numpy()
    if task == "bust":    # rest-of-season ppg below 70% of the model estimate
        return (tr["future_ppg"] < 0.7 * tr["est_ppg"]).astype(int).to_numpy(), (te["future_ppg"] < 0.7 * te["est_ppg"]).astype(int).to_numpy()
    if task == "boom":    # above 130% of the model estimate
        return (tr["future_ppg"] > 1.3 * tr["est_ppg"]).astype(int).to_numpy(), (te["future_ppg"] > 1.3 * te["est_ppg"]).astype(int).to_numpy()
    raise ValueError(task)


def run_clf(p, task, seasons=SEASONS):
    p = p[p["future_played"] >= 3].copy().reset_index(drop=True)
    models = ["L0", "L1", "L1r", "GB"]
    for m in models:
        p["pc_" + m] = np.nan
    p["y"] = np.nan
    p["base_rate"] = np.nan
    for held in seasons:
        tr = p[p["season"] != held]
        te_mask = (p["season"] == held).to_numpy()
        te = p[te_mask]
        ytr, yte = make_target(task, tr, te)
        p.loc[te_mask, "y"] = yte
        p.loc[te_mask, "base_rate"] = ytr.mean()
        p.loc[te_mask, "pc_L0"] = pred_logit(fit_logit(tr, ytr, FEATS["base"]), te, FEATS["base"])
        p.loc[te_mask, "pc_L1"] = pred_logit(fit_logit(tr, ytr, FEATS["full"], 1.0), te, FEATS["full"])
        C = inner_C(tr, ytr, FEATS["full"])
        p.loc[te_mask, "pc_L1r"] = pred_logit(fit_logit(tr, ytr, FEATS["full"], C), te, FEATS["full"])
        gb = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05, min_samples_leaf=80, l2_regularization=1.0, random_state=0)
        gb.fit(design_gb(tr, FEATS["full"]), ytr)
        p.loc[te_mask, "pc_GB"] = gb.predict_proba(design_gb(te, FEATS["full"]))[:, 1]
    return p, models


def boot_brier(p, m, base="L0", reps=2000, seed=1):
    p = p.assign(d=(p["pc_" + m] - p["y"]) ** 2 - (p["pc_" + base] - p["y"]) ** 2)
    g = p.groupby(["season", "W", "position"])["d"].agg(["sum", "size"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (reps, len(g)))
    s, n = g["sum"].to_numpy(), g["size"].to_numpy()
    b = np.array([s[i].sum() / n[i].sum() for i in idx])
    return s.sum() / n.sum(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def cmd_clf():
    p0 = pd.read_csv(OUT / "ml_reg_preds.csv")
    pd.set_option("display.width", 220)
    for task in ["hit", "surprise", "bust", "boom"]:
        p, models = run_clf(p0, task)
        p.to_csv(OUT / f"ml_clf_{task}.csv", index=False)
        print(f"\n=== task: {task}  (base rate {p['y'].mean():.3f}, n={len(p)}) ===")
        rows = []
        for m in models:
            pp = np.clip(p["pc_" + m], 1e-6, 1 - 1e-6)
            bs = float(((pp - p["y"]) ** 2).mean())
            ref = float(((p["base_rate"] - p["y"]) ** 2).mean())
            rows.append({"model": m, "AUC": roc_auc_score(p["y"], pp), "Brier": bs, "BSS_vs_rate": 1 - bs / ref,
                         "logloss": log_loss(p["y"], pp), "ECE": ece(p["y"].to_numpy(), pp.to_numpy())})
        print(pd.DataFrame(rows).round(4).to_string(index=False))
        for m in models[1:]:
            d, lo, hi = boot_brier(p, m)
            print(f"  dBrier {m} vs L0: {d:+.5f} [{lo:+.5f}, {hi:+.5f}]  (negative = better)")
        aucs = []
        for pos in SKILL:
            q = p[p["position"] == pos]
            aucs.append({"pos": pos, "AUC L0": roc_auc_score(q["y"], q["pc_L0"]), "AUC L1r": roc_auc_score(q["y"], q["pc_L1r"]), "AUC GB": roc_auc_score(q["y"], q["pc_GB"])})
        print(pd.DataFrame(aucs).round(3).to_string(index=False))


# ----------------------------------------------------------------------------------------------- decisions
def cmd_decide():
    p = pd.read_csv(OUT / "ml_reg_preds.csv")
    pd.set_option("display.width", 220)
    ml_models = ["xfp", "usage", "full", "full_ridge", "full_gb"]
    z = lambda s: (s - s.mean()) / (s.std() or 1.0)
    # ---- waiver replay
    recs = []
    fa = p[~p["rostered"] & p["pred_base"].notna() & (p["position"].isin(SKILL))]
    for (yr, W, pos), g in fa.groupby(["season", "W", "position"]):
        k = int(g["added_now"].sum())
        if k == 0:
            continue
        g = g.copy()
        rec = {"season": yr, "W": W, "position": pos, "k": k, "managers": g.loc[g["added_now"], "future_points"].mean(),
               "best": g["future_points"].nlargest(k).mean()}
        g["cur"] = z(g["espn_now_f"]) + z(g["v2_total"])
        rec["current"] = g.nlargest(k, "cur")["future_points"].mean()
        rec["v2_only"] = g.nlargest(k, "v2_total")["future_points"].mean()
        rec["espn_only"] = g.nlargest(k, "espn_now_f")["future_points"].mean()
        for m in ml_models + ["mkt"]:
            ratio = (g["pred_" + m] / g["pred_base"].clip(lower=0.5)).clip(0.5, 2.0)
            g["u"] = g["v2_total"] * ratio
            g["bl"] = z(g["espn_now_f"]) + z(g["u"])
            rec["ml_" + m] = g.nlargest(k, "bl")["future_points"].mean()
            rec["mlonly_" + m] = g.nlargest(k, "u")["future_points"].mean()
        recs.append(rec)
    r = pd.DataFrame(recs)
    r.to_csv(OUT / "ml_waiver_replay.csv", index=False)
    cols = [c for c in r.columns if c not in ("season", "W", "position", "k")]
    wsum = r[cols].mul(r["k"], axis=0).sum() / r["k"].sum()
    print("Waiver replay: realised points W..17 per pickup (all skill positions)")
    print(wsum.round(1).to_string())
    for pos in SKILL:
        q = r[r["position"] == pos]
        print(pos, "k=", int(q["k"].sum()), (q[["managers", "current", "ml_full_ridge", "ml_full_gb", "ml_xfp"]].mul(q["k"], axis=0).sum() / q["k"].sum()).round(1).to_dict())
    rng = np.random.default_rng(2)
    for m in ["ml_xfp", "ml_usage", "ml_full", "ml_full_ridge", "ml_full_gb", "ml_mkt"]:
        dlt = ((r[m] - r["current"]) * r["k"]).to_numpy()
        kk = r["k"].to_numpy(float)
        idx = rng.integers(0, len(r), (3000, len(r)))
        b = np.array([dlt[i].sum() / kk[i].sum() for i in idx])
        print(f"  {m:14s} minus current: {dlt.sum() / kk.sum():+.2f} pts/pickup  [{np.percentile(b, 2.5):+.2f}, {np.percentile(b, 97.5):+.2f}]   seasons>0: {int((r.assign(d=r[m]-r['current']).groupby('season')['d'].mean() > 0).sum())}/5")
    # ---- trade-direction proxy: pairs of rostered players
    rost = p[p["rostered"] & p["pred_base"].notna() & p["position"].isin(SKILL)].copy()
    rost["tot_base"] = rost["pred_base"] * rost["exp_games"]
    for m in ml_models + ["mkt"]:
        rost["tot_" + m] = rost["pred_" + m] * rost["exp_games"]
    rost["tot_cur"] = rost["v2_total"]
    rng = np.random.default_rng(5)
    res = {m: [] for m in ["cur", "base"] + ml_models + ["mkt"]}
    meta = []
    for (yr, W), g in rost.groupby(["season", "W"]):
        n = len(g)
        i = rng.integers(0, n, 4000)
        j = rng.integers(0, n, 4000)
        keep = (i != j) & (g["position"].to_numpy()[i] != g["position"].to_numpy()[j])
        i, j = i[keep], j[keep]
        real = g["future_points"].to_numpy()[i] - g["future_points"].to_numpy()[j]
        meta.append(pd.DataFrame({"season": yr, "W": W, "real": real}))
        for m in res:
            t = g["tot_" + m].to_numpy()
            res[m].append(t[i] - t[j])
    meta = pd.concat(meta, ignore_index=True)
    print("\nTrade-direction proxy: random cross-position pairs of rostered players; does the sign of predicted ROS-points difference match realised? (ties/zero-real removed)")
    rows = []
    bins = [0, 5, 10, 20, 40, 80, 1e9]
    for m, parts in res.items():
        pr = np.concatenate(parts)
        ok = meta["real"].to_numpy() != 0
        acc = float((np.sign(pr[ok]) == np.sign(meta["real"].to_numpy()[ok])).mean())
        rec = {"model": m, "acc_all": acc}
        ab = np.abs(pr)
        for lo, hi in zip(bins[:-1], bins[1:]):
            mm = ok & (ab >= lo) & (ab < hi)
            rec[f"{lo}-{hi if hi < 1e8 else 'inf'}"] = float((np.sign(pr[mm]) == np.sign(meta["real"].to_numpy()[mm])).mean()) if mm.sum() > 50 else np.nan
        rows.append(rec)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    # paired bootstrap on accuracy difference vs current by (season, W) cluster
    ok = meta["real"].to_numpy() != 0
    cl = meta["season"].astype(str) + "_" + meta["W"].astype(str)
    for m in ["base"] + ml_models + ["mkt"]:
        pr_m = np.concatenate(res[m]); pr_c = np.concatenate(res["cur"])
        hit_m = (np.sign(pr_m) == np.sign(meta["real"].to_numpy())) & ok
        hit_c = (np.sign(pr_c) == np.sign(meta["real"].to_numpy())) & ok
        df = pd.DataFrame({"cl": cl, "d": hit_m.astype(float) - hit_c.astype(float), "n": ok.astype(float)})
        gg = df.groupby("cl").agg(d=("d", "sum"), n=("n", "sum"))
        rng2 = np.random.default_rng(3)
        idx = rng2.integers(0, len(gg), (2000, len(gg)))
        dd, nn = gg["d"].to_numpy(), gg["n"].to_numpy()
        b = np.array([dd[i].sum() / nn[i].sum() for i in idx])
        print(f"  {m:12s} accuracy minus current: {dd.sum() / nn.sum():+.4f} [{np.percentile(b, 2.5):+.4f}, {np.percentile(b, 97.5):+.4f}]")
    # rank corr of ROS totals among rostered players per (season, W)
    print("\nRank corr of predicted vs realised ROS points among rostered skill players (mean over season-weeks):")
    for m in ["cur", "base"] + ml_models + ["mkt"]:
        rho = [spear(g["tot_" + m], g["future_points"]) for _, g in rost.groupby(["season", "W"])]
        print(f"  {m:12s} {np.nanmean(rho):.4f}")


# ----------------------------------------------------------------------------------------------- placebo
def cmd_placebo(reps=25):
    d = load()
    base_models = ["base", "full"]
    real = regression_predictions(d, "loso", models=base_models)
    gr = group_rhos(real, base_models)
    dm, lo, hi = boot_delta(gr, "full")
    print(f"real 'full' (OLS) vs base: delta {dm:+.4f}")
    rng = np.random.default_rng(11)
    shuffle_cols = XFP + USAGE + PROFILE
    deltas = []
    for r_ in range(reps):
        q = d.copy()
        for _, idx in q.groupby(["season", "W", "position"]).indices.items():
            perm = rng.permutation(idx)
            q.iloc[idx, [q.columns.get_loc(c) for c in shuffle_cols]] = q.iloc[perm][shuffle_cols].to_numpy()
        pp = regression_predictions(q, "loso", models=base_models)
        g2 = group_rhos(pp, base_models)
        deltas.append(np.average(g2["rho_full"] - g2["rho_base"], weights=g2["n"]))
    deltas = np.array(deltas)
    print(f"placebo (features shuffled within season-week-position), {reps} reps: mean delta {deltas.mean():+.4f}, 5th-95th pct [{np.percentile(deltas, 5):+.4f}, {np.percentile(deltas, 95):+.4f}]")
    print("real delta below placebo mean means the real features hurt MORE than random columns of the same distribution do")


if __name__ == "__main__":
    c = sys.argv[1]
    {"build": build, "reg": cmd_reg, "seg": cmd_segments, "fwd": cmd_forward, "clf": cmd_clf, "decide": cmd_decide, "placebo": cmd_placebo}[c]()
