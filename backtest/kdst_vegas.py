"""This week's K and D/ST: does the betting line add to ESPN's weekly projection?

Pool: each week's top 20 K / D/ST by ESPN's weekly projection (the realistic start/stream
choices), weeks 2-17. Models are OLS, fit leave-one-season-out on 2021-25:
  DST : points ~ espn + opponent implied total
  K   : points ~ espn + team spread (+ = favored) + indoor (fixed dome, or retractable roof)
Wind is left out: nflverse only fills it in after the game. Metrics: within-week rank corr with
actual points, and the average of the model's top 5 minus ESPN's top 5 (bootstrap CI over weeks).
The final fit on all of 2021-25 is what app/engine/kdst.py uses. 2026 is a test only.
  python3 backtest/kdst_vegas.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backtest.vegas_signal import ESPN_TEAM  # noqa: E402

SEASONS, TEST = [2021, 2022, 2023, 2024, 2025], 2026
RETRACT = {"ARI", "ATL", "DAL", "HOU", "IND"}  # roof decided on game day, closed ~85% of the time
COLS = {"DST": ["espn", "opp_c"], "K": ["espn", "spread", "indoor"]}
POOL = 20


def lines() -> pd.DataFrame:
    s = pd.read_csv(ROOT / "history" / "nflverse" / "schedules.csv")
    s = s[s["game_type"] == "REG"]
    indoor = ((s["roof"] == "dome") | (s["home_team"].isin(RETRACT) & (s["roof"] != "outdoors"))).astype(float)
    home = pd.DataFrame({"season": s["season"], "week": s["week"], "team": s["home_team"], "spread": s["spread_line"],
                         "total": s["total_line"], "indoor": indoor})
    L = pd.concat([home, home.assign(team=s["away_team"], spread=-s["spread_line"])], ignore_index=True)
    L["opp_c"] = (L["total"] - L["spread"]) / 2 - 22.5
    return L.dropna(subset=["spread", "total"])


def rows(years) -> pd.DataFrame:
    out = []
    for y in years:
        d = ROOT / "history" / "actuals" / str(y)
        wk = pd.read_csv(d / "players_weekly.csv")
        wk = wk[wk["position"].isin(["K", "DST"])]
        act = wk[wk["kind"] == "actual"]
        pts = act.groupby(["espn_id", "week"])["points"].sum()
        tw = act.dropna(subset=["pro_team_id"]).groupby(["espn_id", "week"])["pro_team_id"].last()
        tl = act.dropna(subset=["pro_team_id"]).sort_values("week").groupby("espn_id")["pro_team_id"].last()
        pos = wk.groupby("espn_id")["position"].first()
        pr = pd.read_csv(d / "espn_weekly_proj.csv")
        pr = pr[pr["espn_id"].isin(pos.index) & pr["week"].between(2, min(17, int(act["week"].max()))) & (pr["espn_proj"] > 0)]
        x = pr.rename(columns={"espn_proj": "espn"}).assign(season=y)
        x["position"] = x["espn_id"].map(pos)
        x["points"] = [pts.get((i, w), 0.0) for i, w in zip(x["espn_id"], x["week"])]
        x["team"] = [ESPN_TEAM.get(-i - 16000 if i < 0 else tw.get((i, w), tl.get(i))) for i, w in zip(x["espn_id"], x["week"])]
        out.append(x)
    return pd.concat(out, ignore_index=True).merge(lines(), on=["season", "week", "team"], how="inner")


def ols(X, y):
    return np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)[0]


def pred(b, X):
    return np.column_stack([np.ones(len(X)), X]) @ b


def score(g: pd.DataFrame, col: str) -> pd.DataFrame:
    g = g[g.groupby(["season", "week"])["espn"].rank(ascending=False, method="first") <= POOL]
    sp = lambda a, b: pd.Series(list(a)).rank().corr(pd.Series(list(b)).rank())
    return g.groupby(["season", "week"]).apply(lambda h: pd.Series({
        "rc_espn": sp(h["espn"], h["points"]), "rc_model": sp(h[col], h["points"]),
        "top5": h.nlargest(5, col)["points"].mean() - h.nlargest(5, "espn")["points"].mean()})).reset_index()


def report(per: pd.DataFrame, label: str):
    rng = np.random.default_rng(0)
    d = per["top5"].to_numpy()
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    s = per.groupby("season")[["rc_espn", "rc_model"]].mean().round(3)
    print(f"{label}: rank corr ESPN {per['rc_espn'].mean():.3f} -> model {per['rc_model'].mean():.3f} "
          f"(better in {(s['rc_model'] > s['rc_espn']).sum()}/{len(s)} seasons) | top-5 gain {d.mean():+.2f} "
          f"[{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}] pts/wk, {len(d)} weeks")


if __name__ == "__main__":
    test = (ROOT / "history" / "actuals" / str(TEST) / "players_weekly.csv").exists()
    x = rows(SEASONS + ([TEST] if test else []))
    for ps, cols in COLS.items():
        g = x[x["position"] == ps].copy()
        g["model"] = np.nan
        for held in SEASONS:
            tr = g[g["season"].isin(SEASONS) & (g["season"] != held)]
            g.loc[g["season"] == held, "model"] = pred(ols(tr[cols].to_numpy(float), tr["points"].to_numpy(float)),
                                                        g.loc[g["season"] == held, cols].to_numpy(float))
        tr = g[g["season"].isin(SEASONS)]
        b = ols(tr[cols].to_numpy(float), tr["points"].to_numpy(float))
        print(f"\n{ps} fit 2021-25 (n={len(tr)}):", dict(zip(["const"] + cols, np.round(b, 3))))
        report(score(tr, "model"), f"{ps} LOSO 2021-25")
        te = g[g["season"] == TEST].copy()
        if len(te):
            te["model"] = pred(b, te[cols].to_numpy(float))
            report(score(te, "model"), f"{ps} test {TEST} wks {te['week'].min()}-{te['week'].max()}")
