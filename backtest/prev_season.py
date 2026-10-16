"""Does last season's production improve the in-season forecast on top of v2?

Uses the per-checkpoint rows from usage_signal.py (v2 estimate fit leave-one-season-out,
rest-of-season outcome). Adds last season's ppg (weeks 1-17, >=4 games) and last-4-weeks
ppg, then LOSO per-position regressions of rest-of-season ppg. 2021 has no prior season in
the cache, so it's excluded. Also replays the waiver backtest with the adjusted forecast.

  python3 backtest/prev_season.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backtest import multi_season as ms  # noqa: E402
from backtest import usage_signal as us  # noqa: E402

OUT = ROOT / "history" / "backtest" / "multi"


def prev_features() -> pd.DataFrame:
    out = []
    for y in range(2021, 2026):
        d = pd.read_csv(ROOT / "history" / "actuals" / str(y) / "players_weekly.csv")
        a = d[(d["kind"] == "actual") & d["week"].between(1, 17) & (d["played"] == True)]  # noqa: E712
        f = a.groupby("espn_id")["points"].agg(prev_ppg="mean", prev_g="count")
        l = a[a["week"] >= 14].groupby("espn_id")["points"].agg(prev_late="mean", prev_lg="count")
        f = f.join(l).reset_index()
        f["season"] = y + 1
        out.append(f)
    return pd.concat(out, ignore_index=True)


FEATS = {
    "v2": ["est_ppg"],
    "v2+prev": ["est_ppg", "prev_ppg_f", "has_prev"],
    "v2+prev+late": ["est_ppg", "prev_ppg_f", "prev_late_f", "has_prev"],
    "v2+prev x week": ["est_ppg", "prev_ppg_f", "has_prev", "prev_x_early"],
}


def prep(d: pd.DataFrame) -> pd.DataFrame:
    d = d.merge(prev_features(), on=["season", "espn_id"], how="left")
    d["has_prev"] = (d["prev_g"].fillna(0) >= 4).astype(float)
    d["prev_ppg_f"] = np.where(d["has_prev"] > 0, d["prev_ppg"], d["est_ppg"])
    ok_late = (d["has_prev"] > 0) & (d["prev_lg"].fillna(0) >= 2)
    d["prev_late_f"] = np.where(ok_late, d["prev_late"], d["prev_ppg_f"])
    # lets last season matter more early and fade as the season goes on
    d["prev_x_early"] = (d["prev_ppg_f"] - d["est_ppg"]) * (14 - d["week"]) / 11
    return d


def loso(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    for m in FEATS:
        d["pred_" + m] = np.nan
    for held in sorted(d["season"].unique()):
        tr = d[(d["season"] != held) & (d["future_played"] >= 3) & (d["est_ppg"] > 0)]
        for pos in us.SKILL:
            a = tr[tr["position"] == pos]
            b = (d["season"] == held) & (d["position"] == pos)
            for m, cols in FEATS.items():
                beta = us.ols(a[cols].to_numpy(float), a["future_ppg"].to_numpy(float), a["future_played"].to_numpy(float))
                d.loc[b, "pred_" + m] = us.predict(beta, d.loc[b, cols].to_numpy(float)).clip(min=0)
    return d


def evaluate(d: pd.DataFrame) -> pd.DataFrame:
    out = []
    g0 = d[(d["future_played"] >= 3) & d["rostered"]]
    for pos in us.SKILL:
        for span, g in (("wk3-5", g0[g0["week"] <= 5]), ("wk6-13", g0[g0["week"] > 5])):
            g = g[g["position"] == pos]
            rec = {"position": pos, "span": span, "n": len(g)}
            for m in FEATS:
                rec[m] = np.nanmean([ms.spearman(x["pred_" + m], x["future_ppg"]) for _, x in g.groupby(["season", "week"]) if len(x) > 5])
            out.append(rec)
    return pd.DataFrame(out)


def bootstrap_gain(d: pd.DataFrame, m: str, n: int = 2000) -> tuple:
    g0 = d[(d["future_played"] >= 3) & d["rostered"]]
    per = []
    for (s, w, p), x in g0.groupby(["season", "week", "position"]):
        if len(x) > 5:
            per.append(ms.spearman(x["pred_" + m], x["future_ppg"]) - ms.spearman(x["pred_v2"], x["future_ppg"]))
    per = np.array(per)
    rng = np.random.default_rng(0)
    bs = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(n)]
    return per.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


if __name__ == "__main__":
    d = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob("usage_rows_*.csv"))], ignore_index=True)
    d = d[d["season"] >= 2022]
    d = loso(prep(d))
    pd.set_option("display.width", 200)
    print(evaluate(d).round(3).to_string(index=False))
    for m in list(FEATS)[1:]:
        mean, lo, hi = bootstrap_gain(d, m)
        print(f"{m}: rank-corr gain vs v2 {mean:+.3f} (95% {lo:+.3f} to {hi:+.3f})")
    d["pred_v2+prev x week"] = d["pred_v2+prev x week"]
    r = us.waiver_replay(d.assign(**{"pred_v2+xfp+role": d["pred_v2+prev x week"]}), "v2+xfp+role")
    print("\nWaiver replay with v2+prev x week (column v2u):")
    print(us.summarize_waivers(r)[["managers", "v2_total", "v2u", "best", "k"]].to_string())
