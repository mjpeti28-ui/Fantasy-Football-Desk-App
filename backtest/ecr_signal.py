"""Does FantasyPros expert consensus (ECR) add information beyond ESPN / v2?

Weekly: Friday weekly-position ECR vs ESPN's (final, pre-kickoff) weekly projection for
that week's points, rostered QB/RB/WR/TE. Metrics: within-week rank corr, MAE of a LOSO
regression, pairwise start/sit accuracy (same team, same position), lineup points.
ROS: Friday rest-of-season ECR at checkpoint W vs v2's per-game estimate for points
from W to 17. Metrics: rank corr (ppg and total), MAE, who wins when they disagree,
and whether expert disagreement (ECR sd) flags bigger misses.
  python3 backtest/ecr_signal.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import nflverse_data as nd  # noqa: E402
from backtest import multi_season as ms  # noqa: E402
from backtest import usage_signal as us  # noqa: E402

OUT = ROOT / "history" / "backtest" / "multi"
SKILL = ["QB", "RB", "WR", "TE"]
SEASONS = [2021, 2022, 2023, 2024, 2025]


def weekly_rows(E: pd.DataFrame) -> pd.DataFrame:
    out = []
    for y in SEASONS:
        d = ROOT / "history" / "actuals" / str(y)
        wk = pd.read_csv(d / "players_weekly.csv")
        act = wk[(wk["kind"] == "actual")].groupby(["espn_id", "week"])["points"].sum()
        pr = pd.read_csv(d / "espn_weekly_proj.csv")
        r = pd.read_csv(d / "rosters_by_week.csv")
        r = r[r["position"].isin(SKILL)][["week", "team_id", "espn_id", "position", "lineup_slot", "name"]]
        e = E[(E["season"] == y) & E["page_type"].str.startswith("weekly")][["week", "espn_id", "ecr", "sd"]]
        x = r.merge(pr, on=["week", "espn_id"], how="left").merge(e, on=["week", "espn_id"], how="left")
        x["points"] = [act.get((i, w), 0.0) for i, w in zip(x["espn_id"], x["week"])]
        x["season"] = y
        out.append(x)
    x = pd.concat(out, ignore_index=True)
    return x[(x["espn_proj"] > 0) & x["ecr"].notna() & (x["week"] <= 17)]


def loso_reg(x: pd.DataFrame, cols, target="points", weight=None) -> pd.Series:
    pred = pd.Series(np.nan, index=x.index)
    for held in SEASONS:
        for pos in SKILL:
            tr = x[(x["season"] != held) & (x["position"] == pos)]
            te = (x["season"] == held) & (x["position"] == pos)
            if tr.empty or not te.any():
                continue
            w = tr[weight].to_numpy(float) if weight else np.ones(len(tr))
            b = us.ols(tr[cols].to_numpy(float), tr[target].to_numpy(float), w)
            pred[te] = us.predict(b, x.loc[te, cols].to_numpy(float))
    return pred


def rank_corr(x, pred_col, target, by=("season", "week", "position")):
    return float(np.mean([ms.spearman(g[pred_col], g[target]) for _, g in x.groupby(list(by)) if len(g) > 5]))


def pairwise(x, col):
    """Same team, same week, same position: how often does the rule rank the higher scorer first?"""
    right = tot = 0
    for _, g in x.groupby(["season", "week", "team_id", "position"]):
        if len(g) < 2:
            continue
        v, p = g[col].to_numpy(), g["points"].to_numpy()
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                if p[i] == p[j] or v[i] == v[j]:
                    continue
                tot += 1
                right += (v[i] > v[j]) == (p[i] > p[j])
    return right / tot, tot


def weekly_report(x):
    x = x.copy()
    x["log_ecr"] = np.log(x["ecr"])
    x["neg_ecr"] = -x["ecr"]
    x["p_espn"] = loso_reg(x, ["espn_proj"])
    x["p_ecr"] = loso_reg(x, ["log_ecr"])
    x["p_both"] = loso_reg(x, ["espn_proj", "log_ecr"])
    res = {}
    for c in ("p_espn", "p_ecr", "p_both"):
        pw, n = pairwise(x, c)
        res[c] = {"rank corr": rank_corr(x, c, "points"), "MAE": float((x[c] - x["points"]).abs().mean()), "pairwise start/sit": pw}
    res["pairs"] = n
    # partial: ECR residual vs outcome residual after ESPN
    rr = []
    for (s, w, p), g in x.groupby(["season", "week", "position"]):
        if len(g) > 8:
            b1 = np.polyfit(g["espn_proj"], g["points"], 1)
            b2 = np.polyfit(g["espn_proj"], g["log_ecr"], 1)
            rr.append(ms.spearman(g["log_ecr"] - np.polyval(b2, g["espn_proj"]), g["points"] - np.polyval(b1, g["espn_proj"])))
    res["ECR beyond ESPN (partial, - = ECR better rank helps)"] = float(np.mean(rr))
    by_pos = {pos: {c: rank_corr(x[x["position"] == pos], c, "points", ("season", "week")) for c in ("p_espn", "p_ecr", "p_both")} for pos in SKILL}
    return res, by_pos, x


def ros_rows(E):
    d = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob("usage_rows_*.csv"))], ignore_index=True)
    e = E[E["page_type"].isin(["redraft-qb", "redraft-rb", "redraft-wr", "redraft-te"])][["season", "week", "espn_id", "ecr", "sd"]]
    x = d.merge(e, on=["season", "week", "espn_id"], how="inner")
    return x[x["position"].isin(SKILL)]


def ros_report(x):
    x = x.copy()
    x["log_ecr"] = np.log(x["ecr"])
    x = x[x["est_ppg"] > 0]
    ppg = x[x["future_played"] >= 3].copy()
    ppg["p_v2"] = loso_reg(ppg, ["est_ppg"], "future_ppg", "future_played")
    ppg["p_ecr"] = loso_reg(ppg, ["log_ecr"], "future_ppg", "future_played")
    ppg["p_both"] = loso_reg(ppg, ["est_ppg", "log_ecr"], "future_ppg", "future_played")
    res = {c: {"ppg rank corr": rank_corr(ppg, c, "future_ppg"), "ppg MAE": float((ppg[c] - ppg["future_ppg"]).abs().mean())}
           for c in ("p_v2", "p_ecr", "p_both")}
    # totals (availability matters): v2_total vs ECR rank directly
    tot = x.copy()
    tot["neg_log_ecr"] = -tot["log_ecr"]
    res["total points rank corr v2_total"] = rank_corr(tot, "v2_total", "future_points")
    res["total points rank corr ECR"] = rank_corr(tot, "neg_log_ecr", "future_points")
    # disagreements: rank both within season-week-position; who is closer?
    ppg["r_v2"] = ppg.groupby(["season", "week", "position"])["p_v2"].rank(ascending=False)
    ppg["r_ecr"] = ppg.groupby(["season", "week", "position"])["ecr"].rank()
    ppg["r_act"] = ppg.groupby(["season", "week", "position"])["future_ppg"].rank(ascending=False)
    ppg["gap"] = ppg["r_v2"] - ppg["r_ecr"]  # >0: experts like him more than v2
    ppg["gap_bin"] = pd.cut(ppg["gap"], [-999, -10, -4, 4, 10, 999], labels=["v2 >> ECR", "v2 > ECR", "agree", "ECR > v2", "ECR >> v2"])
    dis = ppg.groupby("gap_bin", observed=True).apply(lambda g: pd.Series({
        "n": len(g), "v2 closer": float((np.abs(g["r_v2"] - g["r_act"]) < np.abs(g["r_ecr"] - g["r_act"])).mean()),
        "mean |err| v2": float(np.abs(g["r_v2"] - g["r_act"]).mean()), "mean |err| ECR": float(np.abs(g["r_ecr"] - g["r_act"]).mean())}),
        include_groups=False)
    # expert disagreement as an uncertainty flag
    ppg["abs_err"] = (ppg["p_both"] - ppg["future_ppg"]).abs()
    ppg["sd_rel"] = ppg["sd"] / ppg["ecr"].clip(lower=1)
    sd_q = ppg.groupby(pd.qcut(ppg["sd_rel"], 4, duplicates="drop"), observed=True)["abs_err"].mean()
    return res, dis, sd_q, ppg


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    E = nd.ecr_table()
    E = E.dropna(subset=["espn_id"]).assign(espn_id=lambda d: d["espn_id"].astype("int64"))
    wx = weekly_rows(E)
    res, by_pos, wx = weekly_report(wx)
    print("WEEKLY (next-week points, rostered players, n =", len(wx), ")")
    for k, v in res.items():
        print(" ", k, {a: round(b, 4) for a, b in v.items()} if isinstance(v, dict) else round(v, 4) if isinstance(v, float) else v)
    print(pd.DataFrame(by_pos).round(3).to_string())
    rx = ros_rows(E)
    res, dis, sdq, ppg = ros_report(rx)
    print("\nROS (from checkpoint W to week 17, n =", len(rx), ")")
    for k, v in res.items():
        print(" ", k, {a: round(b, 4) for a, b in v.items()} if isinstance(v, dict) else round(v, 4))
    print(dis.round(3).to_string())
    print("abs error by expert-disagreement quartile (sd/ecr):\n", sdq.round(3).to_string())
    wx.to_csv(OUT / "ecr_weekly.csv", index=False)
    ppg.to_csv(OUT / "ecr_ros.csv", index=False)
