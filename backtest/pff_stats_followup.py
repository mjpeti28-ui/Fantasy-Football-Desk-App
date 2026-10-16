"""Follow-ups to pff_stats.py: (1) PFF feature groups on low-history rows, (2) a gradient-boosted check.

Same rows, leakage guard (features from weeks < W only), LOSO-by-season folds and metrics as
pff_stats.py: within season-week-position Spearman vs future_ppg, MAE weighted by future_played,
rows with future_played >= 3 and est_ppg > 0.

1. Subsets: n <= 3 games observed before W; W <= 5; rookies (draft_year == season, ids.csv).
   Each is scored two ways: OLS fit on the subset only, and OLS fit on all rows, scored on the subset.
2. HistGradientBoostingRegressor per position (depth 3, lr 0.05, min leaf 50, <= 400 iters,
   iterations chosen on the last training season, then refit on all training seasons):
   (a) est_ppg + ECR rank, (b) (a) + all PFF features; compared with the LOSO OLS live blend.
   Held-out permutation importance (drop in rank corr) for PFF features in (b).
  python backtest/pff_stats_followup.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backtest import ecr_signal as es  # noqa: E402
from backtest import pff_stats as ps  # noqa: E402

SEASONS, SKILL, OUT = ps.SEASONS, ps.SKILL, ps.OUT
REC = ["rec_grade", "route_grade", "yprr", "tprr", "route_share", "tgt_share"]
RUS = ["rush_grade", "run_grade", "elusive", "yco", "rush_share", "rush_pg"]
PAS = ["pass_grade_off", "pass_grade", "btt", "twp", "adot", "db_pg"]
_l3 = lambda cs: cs + [c + "_l3" for c in cs]  # noqa: E731
PFF = {"WR": _l3(REC), "TE": _l3(REC), "RB": _l3(RUS + REC), "QB": _l3(PAS + ["rush_share", "rush_pg"])}


def scores(g: pd.DataFrame, p: str) -> dict:
    """Per season: (within week rank corr, weighted MAE) of column p vs future_ppg."""
    return {y: (es.rank_corr(s, p, "future_ppg", ("season", "week")),
                float(np.average((s[p] - s["future_ppg"]).abs(), weights=s["future_played"])))
            for y, s in g.groupby("season")}


def summarize(sc: dict, b: dict) -> dict:
    drc = [sc[y][0] - b[y][0] for y in SEASONS if y in sc]
    dmae = [sc[y][1] - b[y][1] for y in SEASONS if y in sc]
    return {"d_rc": np.nanmean(drc), "d_mae": np.mean(dmae), "rc_up": sum(v > 0 for v in drc),
            "mae_down": sum(v < 0 for v in dmae), "seasons": len(drc),
            "d_rc_by_season": " ".join(f"{v:+.3f}" for v in drc)}


# ------------------------------------------------------------------ 1. low-history subsets

def subset_report(x: pd.DataFrame, masks: dict) -> pd.DataFrame:
    lines = []
    for pos in SKILL:
        g = x[x["position"] == pos]
        allc = sorted({c for v in ps.GROUPS[pos].values() for c in v})
        g = ps.impute(g, allc)
        groups = dict(ps.GROUPS[pos], all=allc)
        for sname, mfn in masks.items():
            m = mfn(g)
            for fit in ("subset", "all"):
                fit_rows = g[m] if fit == "subset" else g
                for base in ("est_ppg", "live"):
                    h = fit_rows.copy()
                    h["p0"] = es.loso_reg(h, [base], "future_ppg", "future_played")
                    for name, cols in groups.items():
                        h["p1"] = es.loso_reg(h, [base] + cols, "future_ppg", "future_played")
                        s = h.loc[m[m].index.intersection(h.index)]
                        b = scores(s, "p0")
                        lines.append({"subset": sname, "fit": fit, "pos": pos, "base": base, "group": name,
                                      "rows": len(s), "base_rc": np.nanmean([v[0] for v in b.values()]),
                                      **summarize(scores(s, "p1"), b)})
    return pd.DataFrame(lines)


# ------------------------------------------------------------------ 2. GBM

GBM_KW = dict(max_depth=3, learning_rate=0.05, min_samples_leaf=50, l2_regularization=1.0, random_state=0)


def gbm_fit(tr: pd.DataFrame, cols) -> HistGradientBoostingRegressor:
    """Pick iterations on the last training season, then refit on all training seasons."""
    last = tr["season"].max()
    a, v = tr[tr["season"] < last], tr[tr["season"] == last]
    m = HistGradientBoostingRegressor(max_iter=400, early_stopping=False, **GBM_KW)
    m.fit(a[cols], a["future_ppg"], sample_weight=a["future_played"])
    err = [np.average((p - v["future_ppg"]) ** 2, weights=v["future_played"]) for p in m.staged_predict(v[cols])]
    n = max(int(np.argmin(err)) + 1, 20)
    m = HistGradientBoostingRegressor(max_iter=n, early_stopping=False, **GBM_KW)
    m.fit(tr[cols], tr["future_ppg"], sample_weight=tr["future_played"])
    m.n_chosen = n
    return m


def gbm_loso(g: pd.DataFrame, cols, perm=None, rng=None):
    pred = pd.Series(np.nan, index=g.index)
    imp, iters = {}, []
    for held in SEASONS:
        tr, te = g[g["season"] != held], g[g["season"] == held].copy()
        m = gbm_fit(tr, cols)
        iters.append(m.n_chosen)
        pred[te.index] = m.predict(te[cols])
        if perm:
            te["p"] = pred[te.index]
            base = es.rank_corr(te, "p", "future_ppg", ("season", "week"))
            for c in perm:
                drops = []
                for _ in range(3):
                    t2 = te.copy()
                    t2[c] = rng.permutation(t2[c].to_numpy())
                    t2["p"] = m.predict(t2[cols])
                    drops.append(base - es.rank_corr(t2, "p", "future_ppg", ("season", "week")))
                imp.setdefault(c, []).append(np.mean(drops))
    return pred, imp, iters


def gbm_report(x: pd.DataFrame, masks: dict):
    rng = np.random.default_rng(0)
    lines, imps = [], []
    for pos in SKILL:
        g = x[x["position"] == pos].copy()
        for c in PFF[pos]:
            g[c] = g[c].replace([np.inf, -np.inf], np.nan)
        g["lin_live"] = es.loso_reg(g, ["live"], "future_ppg", "future_played")
        g["gbm_a"], _, it_a = gbm_loso(g, ["est_ppg", "ecr"])
        g["gbm_b"], imp, it_b = gbm_loso(g, ["est_ppg", "ecr"] + PFF[pos], perm=PFF[pos], rng=rng)
        for sname, mfn in {"all": lambda d: d["week"] > 0, **masks}.items():
            s = g[mfn(g)]
            sc = {k: scores(s, k) for k in ("lin_live", "gbm_a", "gbm_b")}
            for comp, (k1, k0) in {"a - live": ("gbm_a", "lin_live"), "b - a": ("gbm_b", "gbm_a"),
                                   "b - live": ("gbm_b", "lin_live")}.items():
                lines.append({"rows": sname, "pos": pos, "comp": comp, "n": len(s),
                              "base_rc": np.nanmean([v[0] for v in sc[k0].values()]),
                              **summarize(sc[k1], sc[k0])})
        for c, v in imp.items():
            imps.append({"pos": pos, "feature": c, "mean_drop": np.mean(v), "folds_pos": sum(d > 0 for d in v),
                         "by_season": " ".join(f"{d:+.4f}" for d in v)})
        print(f"{pos}: iterations chosen a={it_a} b={it_b}", flush=True)
    return pd.DataFrame(lines), pd.DataFrame(imps)


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 500)
    x = ps.build_rows()
    x = x[x["future_played"] >= 3].copy()
    ids = pd.read_csv(ROOT / "history" / "nflverse" / "ids.csv").dropna(subset=["espn_id", "draft_year"])
    dy = ids.drop_duplicates("espn_id").set_index(ids.drop_duplicates("espn_id")["espn_id"].astype("int64"))["draft_year"]
    x["draft_year"] = x["espn_id"].map(dy)
    masks = {"n<=3": lambda d: d["n"] <= 3, "W<=5": lambda d: d["week"] <= 5,
             "rookie": lambda d: d["draft_year"] == d["season"]}
    print("== subset rows by position")
    print(pd.DataFrame({k: x.groupby("position").apply(lambda d, f=f: int(f(d).sum())) for k, f in masks.items()}))

    sub = subset_report(x, masks)
    sub.to_csv(OUT / "pff_followup_subsets.csv", index=False)
    print("\n== subsets: PFF group on top of base (d_rc > 0, d_mae < 0 = better)")
    print(sub.round(4).to_string(index=False))

    gl, gi = gbm_report(x, masks)
    gl.to_csv(OUT / "pff_followup_gbm.csv", index=False)
    gi.to_csv(OUT / "pff_followup_gbm_importance.csv", index=False)
    print("\n== GBM comparisons")
    print(gl.round(4).to_string(index=False))
    print("\n== GBM (b) held-out permutation importance, top 6 per position (drop in rank corr)")
    print(gi.sort_values("mean_drop", ascending=False).groupby("pos").head(6).round(4).to_string(index=False))
