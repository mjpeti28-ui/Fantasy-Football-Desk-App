"""Do Vegas lines add information to ESPN's weekly projection (and ECR)?

For each rostered player-week: team implied total = total/2 +/- spread/2 (nflverse closing
lines; spread_line > 0 = home favored), opponent implied total, spread from the team's
view, plus wind/temperature. Angles:
  additive residual  : points ~ espn_proj + implied (+ spread, + opp) per position
  multiplicative     : espn_proj * (implied / 22.5)^b
  game script        : spread interacted with position (RB favored, QB/WR trailing)
  D/ST and K         : opponent implied total / own total
Metrics: within-week rank corr, MAE, pairwise start/sit accuracy, bootstrap CIs.
  python3 backtest/vegas_signal.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backtest import ecr_signal as es  # noqa: E402
from backtest import multi_season as ms  # noqa: E402

ESPN_TEAM = {1: "ATL", 2: "BUF", 3: "CHI", 4: "CIN", 5: "CLE", 6: "DAL", 7: "DEN", 8: "DET", 9: "GB", 10: "TEN",
             11: "IND", 12: "KC", 13: "LV", 14: "LA", 15: "MIA", 16: "MIN", 17: "NE", 18: "NO", 19: "NYG", 20: "NYJ",
             21: "PHI", 22: "ARI", 23: "PIT", 24: "LAC", 25: "SF", 26: "SEA", 27: "TB", 28: "WAS", 29: "CAR",
             30: "JAX", 33: "BAL", 34: "HOU"}
POSS = ["QB", "RB", "WR", "TE", "K", "DST"]


def lines() -> pd.DataFrame:
    s = pd.read_csv(ROOT / "history" / "nflverse" / "schedules.csv")
    s = s[s["game_type"] == "REG"]
    home = pd.DataFrame({"season": s["season"], "week": s["week"], "team": s["home_team"], "opp": s["away_team"],
                         "spread": s["spread_line"], "total": s["total_line"], "wind": s["wind"], "temp": s["temp"], "roof": s["roof"]})
    away = home.assign(team=s["away_team"], opp=s["home_team"], spread=-s["spread_line"])
    L = pd.concat([home, away], ignore_index=True)
    L["implied"] = L["total"] / 2 + L["spread"] / 2
    L["opp_implied"] = L["total"] / 2 - L["spread"] / 2
    L["dome"] = L["roof"].isin(["dome", "closed"]).astype(float)
    L["wind"] = L["wind"].fillna(0).where(L["dome"] == 0, 0)
    return L


def rows() -> pd.DataFrame:
    L = lines()
    out = []
    for y in es.SEASONS:
        d = ROOT / "history" / "actuals" / str(y)
        wk = pd.read_csv(d / "players_weekly.csv")
        act = wk[wk["kind"] == "actual"]
        pts = act.groupby(["espn_id", "week"])["points"].sum()
        team = act.dropna(subset=["pro_team_id"]).sort_values("week").groupby("espn_id")["pro_team_id"].last()
        team_wk = act.dropna(subset=["pro_team_id"]).set_index(["espn_id", "week"])["pro_team_id"]
        pr = pd.read_csv(d / "espn_weekly_proj.csv")
        r = pd.read_csv(d / "rosters_by_week.csv")
        r["position"] = r["position"].replace({"D/ST": "DST"})
        r = r[r["position"].isin(POSS)][["week", "team_id", "espn_id", "position"]]
        x = r.merge(pr, on=["week", "espn_id"], how="left")
        x = x[(x["espn_proj"] > 0) & (x["week"] <= 17)]
        x["points"] = [pts.get((i, w), 0.0) for i, w in zip(x["espn_id"], x["week"])]
        tid = []
        for i, w in zip(x["espn_id"], x["week"]):
            if i < 0:
                tid.append(-i - 16000)
            else:
                tid.append(team_wk.get((i, w), team.get(i)))
        x["team"] = pd.Series(tid, index=x.index).map(ESPN_TEAM)
        x["season"] = y
        out.append(x)
    x = pd.concat(out, ignore_index=True).merge(L, on=["season", "week", "team"], how="left")
    return x.dropna(subset=["implied"])


def fit_pred(x, cols, target="points"):
    from backtest import usage_signal as us
    pred = pd.Series(np.nan, index=x.index)
    for held in es.SEASONS:
        for pos in POSS:
            tr = x[(x["season"] != held) & (x["position"] == pos)]
            te = (x["season"] == held) & (x["position"] == pos)
            if len(tr) and te.any():
                b = us.ols(tr[cols].to_numpy(float), tr[target].to_numpy(float), np.ones(len(tr)))
                pred[te] = us.predict(b, x.loc[te, cols].to_numpy(float))
    return pred


def pairwise(x, col):
    return es.pairwise(x.rename(columns={}), col)


def boot_diff_pos(g, col, target="points"):
    per = [ms.spearman(h[col], h[target]) for _, h in g.groupby(["season", "week"]) if len(h) > 5]
    return float(np.mean(per)), len(per)


def boot_diff(x, a, b, target="points"):
    per, seas = [], []
    for k, g in x[x["position"].isin(["QB", "RB", "WR", "TE"])].groupby(["season", "week", "position"]):
        if len(g) > 5:
            per.append(ms.spearman(g[b], g[target]) - ms.spearman(g[a], g[target]))
            seas.append(k[0])
    per = np.array(per)
    rng = np.random.default_rng(0)
    bs = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(2000)]
    return per.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), pd.Series(per).groupby(seas).mean().round(3).to_dict()


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    x = rows()
    x["imp_c"] = x["implied"] - 22.5
    x["opp_c"] = x["opp_implied"] - 22.5
    x["espn_x_imp"] = x["espn_proj"] * x["imp_c"] / 22.5
    x["espn_x_opp"] = x["espn_proj"] * x["opp_c"] / 22.5
    x["espn_x_spread"] = x["espn_proj"] * x["spread"] / 10
    x["espn_x_wind"] = x["espn_proj"] * np.clip(x["wind"] - 10, 0, None) / 10
    print("rows", len(x), x.groupby("position").size().to_dict(), "team match", round(x["team"].notna().mean(), 3))
    E = __import__("nflverse_data").ecr_table().dropna(subset=["espn_id"])
    E["espn_id"] = E["espn_id"].astype("int64")
    wE = E[E["page_type"].str.startswith("weekly")][["season", "week", "espn_id", "ecr"]]
    x = x.merge(wE, on=["season", "week", "espn_id"], how="left")
    x["log_ecr"] = np.log(x["ecr"].fillna(x.groupby(["season", "week", "position"])["ecr"].transform("max") + 5).fillna(60))
    variants = {
        "espn": ["espn_proj"],
        "+implied": ["espn_proj", "imp_c"],
        "+implied mult": ["espn_proj", "espn_x_imp"],
        "+opp implied": ["espn_proj", "opp_c", "espn_x_opp"],
        "+spread (script)": ["espn_proj", "espn_x_spread", "spread"],
        "+all vegas": ["espn_proj", "espn_x_imp", "espn_x_opp", "espn_x_spread", "espn_x_wind"],
        "espn+ecr": ["espn_proj", "log_ecr"],
        "espn+ecr+vegas": ["espn_proj", "log_ecr", "espn_x_imp", "espn_x_opp", "espn_x_spread", "espn_x_wind"],
    }
    for name, cols in variants.items():
        x["p_" + name] = fit_pred(x, cols)
    res = []
    for pos in POSS:
        g = x[x["position"] == pos]
        rec = {"pos": pos, "n": len(g)}
        for name in variants:
            rec[name] = es.rank_corr(g, "p_" + name, "points", ("season", "week"))
        res.append(rec)
    print(pd.DataFrame(res).round(3).to_string(index=False))
    for pos in ("K", "DST"):
        g = x[x["position"] == pos]
        base = boot_diff_pos(g, "p_espn")
        print(pos, {n: round(boot_diff_pos(g, "p_" + n)[0] - base[0], 4) for n in variants})
    for name in variants:
        if name == "espn":
            continue
        m, lo, hi, s = boot_diff(x, "p_espn", "p_" + name)
        pw, n = es.pairwise(x[x["position"].isin(["QB", "RB", "WR", "TE"])], "p_" + name)
        print(f"{name:18s} rank-corr gain {m:+.4f} [{lo:+.4f},{hi:+.4f}] seasons {s} | MAE {(x['p_'+name]-x['points']).abs().mean():.3f} | pairwise {pw:.4f}")
    pw, n = es.pairwise(x[x["position"].isin(["QB", "RB", "WR", "TE"])], "p_espn")
    print(f"espn baseline MAE {(x['p_espn']-x['points']).abs().mean():.3f} pairwise {pw:.4f} (pairs {n})")
    x.to_csv(es.OUT / "vegas_rows.csv", index=False)
