"""Does nflverse usage (xFP, snap share, touches) improve rest-of-season forecasts?

Build step (per season, cached): for every checkpoint W = 3..13 and every QB/RB/WR/TE
with a v2 forecast, record v2's per-game estimate (params fit without that season),
usage features from games before W, whether the player was rostered going into W
(free agent = candidate pickup), and what happened from W through week 17.

Analysis step: per position, fit future_ppg ~ features on the other seasons (LOSO) and
compare held-out rank correlation / MAE against v2 alone, separately for rostered
players and the free-agent pool. Then replay the waiver backtest with the new rule.

  python3 backtest/usage_signal.py build 2021 [2022 ...]
  python3 backtest/usage_signal.py analyze
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nflverse_data as nd  # noqa: E402
import valuation as v  # noqa: E402
from backtest import multi_season as ms  # noqa: E402

OUT = ROOT / "history" / "backtest" / "multi"
SKILL = ["QB", "RB", "WR", "TE"]
CHECKS = range(3, 14)


def usage_features(U: pd.DataFrame, W: int) -> pd.DataFrame:
    """Per-player usage summary from games before week W (only games he appeared in)."""
    g = U[(U["week"] < W) & ((U["offense_pct"].fillna(0) > 0) | U["fantasy_points_ppr"].notna())].copy()
    g["opp"] = g["carries"].fillna(0) + g["targets"].fillna(0)
    g = g.sort_values(["espn_id", "week"])
    last3 = g.groupby("espn_id").tail(3)
    last2 = g.groupby("espn_id").tail(2)
    f = pd.DataFrame({
        "u_games": g.groupby("espn_id").size(),
        "xfp_pg": g.groupby("espn_id")["xfp"].mean(),
        "fp_pg": g.groupby("espn_id")["fantasy_points_ppr"].mean(),
        "snap_pg": g.groupby("espn_id")["offense_pct"].mean(),
        "opp_pg": g.groupby("espn_id")["opp"].mean(),
        "xfp_l3": last3.groupby("espn_id")["xfp"].mean(),
        "fp_l3": last3.groupby("espn_id")["fantasy_points_ppr"].mean(),
        "opp_l3": last3.groupby("espn_id")["opp"].mean(),
        "tshare_l3": last3.groupby("espn_id")["target_share"].mean(),
        "snap_l2": last2.groupby("espn_id")["offense_pct"].mean(),
    })
    return f


def build(season: int) -> pd.DataFrame:
    rows = pd.read_csv(OUT / "player_rows.csv")
    p = ms.fit_all(rows, [y for y in ms.SEASONS if y != season])
    s = v.load_season(season)
    U = nd.usage_table(season)
    wk = s.weekly
    recs = []
    for W in CHECKS:
        try:
            t = v.player_table(s, W, p, rostered=set(s.roster_at(W - 1)["espn_id"]))
        except Exception as e:  # e.g. broken projection week
            print("skip", season, W, e)
            continue
        t = t[t["position"].isin(SKILL)].copy()
        rost_prev = set(s.roster_at(W - 1)["espn_id"])
        rost_now = set(s.roster_at(W)["espn_id"])
        fut = wk[(wk["week"] >= W) & (wk["week"] <= v.FANTASY_LAST_WEEK)]
        pl = fut[fut["played"]].groupby("espn_id")["points"].agg(["sum", "count"])
        t["v2_total"] = v.expected_ros_points(t, W, v.FANTASY_LAST_WEEK)
        t = t.join(usage_features(U, W))
        t["season"], t["week"] = season, W
        t["rostered"] = t.index.isin(rost_prev)
        t["added_now"] = t.index.isin(rost_now - rost_prev)
        t["future_points"] = t.index.map(pl["sum"]).fillna(0.0)
        t["future_played"] = t.index.map(pl["count"]).fillna(0).astype(int)
        recs.append(t.reset_index())
    d = pd.concat(recs, ignore_index=True)
    d["future_ppg"] = np.where(d["future_played"] > 0, d["future_points"] / d["future_played"].clip(lower=1), np.nan)
    return d


FEATS = {
    "v2": ["est_ppg"],
    "v2+xfp": ["est_ppg", "xfp_l3", "xfp_pg"],
    "v2+xfp+role": ["est_ppg", "xfp_l3", "xfp_pg", "snap_l2", "snap_delta", "opp_l3"],
    "v2+role": ["est_ppg", "snap_l2", "snap_delta", "opp_l3"],
}


def prep(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    has = d["u_games"].fillna(0) > 0
    for c in ("xfp_l3", "xfp_pg"):
        d[c] = d[c].where(has & d[c].notna(), d["est_ppg"])
    d["snap_l2"] = d["snap_l2"].fillna(0.0)
    d["snap_delta"] = (d["snap_l2"] - d["snap_pg"]).fillna(0.0)
    d["opp_l3"] = d["opp_l3"].fillna(0.0)
    return d


def ols(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones(len(X)), X]) * np.sqrt(w)[:, None]
    return np.linalg.lstsq(A, y * np.sqrt(w), rcond=None)[0]


def predict(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X]) @ beta


def loso_predictions(d: pd.DataFrame) -> pd.DataFrame:
    """Adds pred_<model> (per-game) columns, each fit without the held-out season.
    Training rows: rostered + pool, >=3 future games played, v2 estimate > 0."""
    d = d.copy()
    for m in FEATS:
        d["pred_" + m] = np.nan
    for held in sorted(d["season"].unique()):
        tr = d[(d["season"] != held) & (d["future_played"] >= 3) & (d["est_ppg"] > 0)]
        te = d["season"] == held
        for pos in SKILL:
            a = tr[tr["position"] == pos]
            b = te & (d["position"] == pos)
            for m, cols in FEATS.items():
                beta = ols(a[cols].to_numpy(float), a["future_ppg"].to_numpy(float), a["future_played"].to_numpy(float))
                d.loc[b, "pred_" + m] = predict(beta, d.loc[b, cols].to_numpy(float)).clip(min=0)
    return d


def evaluate(d: pd.DataFrame) -> pd.DataFrame:
    out = []
    for grp, g0 in (("rostered", d[d["rostered"]]), ("free agents", d[~d["rostered"]])):
        g0 = g0[g0["future_played"] >= 3]
        for pos in SKILL:
            g = g0[g0["position"] == pos]
            rec = {"group": grp, "position": pos, "n": len(g)}
            for m in FEATS:
                rhos = [ms.spearman(x["pred_" + m], x["future_ppg"]) for _, x in g.groupby(["season", "week"]) if len(x) > 5]
                rec["rho " + m] = np.nanmean(rhos)
                rec["mae " + m] = float(np.abs(g["pred_" + m] - g["future_ppg"]).mean())
            out.append(rec)
    return pd.DataFrame(out)


def waiver_replay(d: pd.DataFrame, model: str) -> pd.DataFrame:
    """Same protocol as waiver_backtest.py: per season/week/position, each rule picks as many
    free agents as managers actually added; score = realized points from W to week 17."""
    recs = []
    for (yr, W, pos), g in d[~d["rostered"]].groupby(["season", "week", "position"]):
        k = int(g["added_now"].sum())
        if k == 0:
            continue
        g = g.copy()
        ratio = (g["pred_" + model] / g["est_ppg"].where(g["est_ppg"] > 0)).fillna(1.0)
        g["v2u"] = g["v2_total"] * ratio
        z = lambda s: (s - s.mean()) / (s.std() or 1.0)
        g["blend"] = z(g["espn_now"].fillna(0)) + z(g["v2_total"])
        g["blend_u"] = z(g["espn_now"].fillna(0)) + z(g["v2u"])
        g["xfp_l3_rule"] = g["xfp_l3"]
        rec = {"season": yr, "week": W, "position": pos, "k": k,
               "managers": g.loc[g["added_now"], "future_points"].mean(),
               "best": g["future_points"].nlargest(k).mean()}
        for rule in ("v2_total", "v2u", "espn_now", "blend", "blend_u", "xfp_l3_rule"):
            rec[rule] = g.nlargest(k, rule)["future_points"].mean()
        recs.append(rec)
    return pd.DataFrame(recs)


def summarize_waivers(r: pd.DataFrame) -> pd.DataFrame:
    cols = ["managers", "v2_total", "v2u", "espn_now", "blend", "blend_u", "xfp_l3_rule", "best"]
    w = r.assign(**{c: r[c] * r["k"] for c in cols})
    tot = w.groupby("position")[cols + ["k"]].sum()
    tot.loc["ALL"] = tot.sum()
    return tot[cols].div(tot["k"], axis=0).round(1).assign(k=tot["k"].astype(int))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        for a in sys.argv[2:]:
            d = build(int(a))
            d.to_csv(OUT / f"usage_rows_{a}.csv", index=False)
            print(a, len(d), "rows; usage coverage", round(d["u_games"].notna().mean(), 3))
    elif cmd == "analyze":
        d = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob("usage_rows_*.csv"))], ignore_index=True)
        d = loso_predictions(prep(d))
        d.to_csv(OUT / "usage_predictions.csv", index=False)
        pd.set_option("display.width", 200)
        print(evaluate(d).round(3).to_string(index=False))
        model = sys.argv[2] if len(sys.argv) > 2 else "v2+xfp+role"
        r = waiver_replay(d, model)
        r.to_csv(OUT / "usage_waiver_replay.csv", index=False)
        print(f"\nWaiver replay (model {model}), realized points W..17 per pickup:")
        print(summarize_waivers(r).to_string())
