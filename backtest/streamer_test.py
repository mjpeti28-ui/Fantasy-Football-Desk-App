"""Is "a replacement-level streamer is always available" true?

The valuation fills every lineup hole with waiver_replacement(): the 2nd-best free agent's
expected ppg (est x availability) x waiver_scale, fixed at the checkpoint. We check it
against what streaming actually produced, 2021-2025:

  predicted : model replacement ppg at checkpoint W (LOSO params, same code as the tools)
  stream1/2/3: each week w in W..W+3, the free agents (not rostered in week w-1) ranked by
              ESPN's projection for week w; the 1st / 2nd / 3rd best -- realized points in w
  managers  : players a team added for week w and started that week -- realized points
  coverage  : share of weeks where the 2nd-best free agent's ESPN projection >= predicted

Breakdowns by position, season, and early/late checkpoint.
  python3 backtest/streamer_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import valuation as v  # noqa: E402
from backtest import multi_season as ms  # noqa: E402

OUT = ROOT / "history" / "backtest" / "multi"
POS = ["QB", "RB", "WR", "TE", "K", "DST"]


def season_rows(yr: int, rows: pd.DataFrame) -> list:
    p = ms.fit_all(rows, [y for y in ms.SEASONS if y != yr])
    s = v.load_season(yr)
    d = ROOT / "history" / "actuals" / str(yr)
    wk = pd.read_csv(d / "players_weekly.csv")
    act = wk[wk["kind"] == "actual"].groupby(["espn_id", "week"])["points"].sum()
    pr = pd.read_csv(d / "espn_weekly_proj.csv")
    pr["position"] = pr["espn_id"].map(s.position)
    r = pd.read_csv(d / "rosters_by_week.csv")
    out = []
    for W in range(3, 14):
        rost = set(s.roster_at(W)["espn_id"])
        tbl = v.player_table(s, W, p, rostered=rost)
        repl = v.waiver_replacement(tbl, rost, scale=p.waiver_scale)
        for w in range(W, min(W + 4, v.FANTASY_LAST_WEEK + 1)):
            prev = set(r.loc[r["week"] == w - 1, "espn_id"])
            now = r[r["week"] == w]
            added_started = now[~now["espn_id"].isin(prev) & ~now["lineup_slot"].isin(["BE", "IR"])]
            fa = pr[(pr["week"] == w) & ~pr["espn_id"].isin(prev) & (pr["espn_proj"] > 0)]
            for pos in POS:
                f = fa[fa["position"] == pos].sort_values("espn_proj", ascending=False)
                real = [act.get((i, w), 0.0) for i in f["espn_id"].head(3)]
                ms_rows = added_started[added_started["espn_id"].map(s.position) == pos]
                out.append({
                    "season": yr, "checkpoint": W, "week": w, "position": pos, "predicted": repl.get(pos, np.nan),
                    "stream1": real[0] if len(real) > 0 else np.nan, "stream2": real[1] if len(real) > 1 else np.nan,
                    "stream3": real[2] if len(real) > 2 else np.nan,
                    "proj2": float(f["espn_proj"].iloc[1]) if len(f) > 1 else np.nan,
                    "managers_sum": float(sum(act.get((i, w), 0.0) for i in ms_rows["espn_id"])), "managers_n": len(ms_rows),
                })
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    rows = pd.read_csv(OUT / "player_rows.csv")
    recs = []
    for yr in ms.SEASONS:
        recs += season_rows(yr, rows)
    d = pd.DataFrame(recs)
    d.to_csv(OUT / "streamer_test.csv", index=False)
    d["covered"] = (d["proj2"] >= d["predicted"]).astype(float)
    g = d.groupby("position")
    t = pd.DataFrame({
        "predicted": g["predicted"].mean(), "stream #1": g["stream1"].mean(), "stream #2": g["stream2"].mean(),
        "stream #3": g["stream3"].mean(),
        "managers' started adds": g["managers_sum"].sum() / g["managers_n"].sum(), "n adds": g["managers_n"].sum(),
        "ESPN proj of #2": g["proj2"].mean(), "weeks #2 proj >= predicted": g["covered"].mean()})
    print("per-game points (averaged over checkpoint x next-4-week rows):\n", t.round(2).to_string())
    print("\nstream #2 / predicted by season:\n", (d.groupby(["position", "season"])["stream2"].mean() / d.groupby(["position", "season"])["predicted"].mean()).unstack().round(2).to_string())
    d["half"] = np.where(d["checkpoint"] <= 7, "wk3-7", "wk8-13")
    print("\nstream #2 / predicted by half:\n", (d.groupby(["position", "half"])["stream2"].mean() / d.groupby(["position", "half"])["predicted"].mean()).unstack().round(2).to_string())
    # bootstrap the ratio by season-week clusters
    rng = np.random.default_rng(0)
    for pos in POS:
        x = d[d["position"] == pos]
        keys = x[["season", "week"]].drop_duplicates().to_numpy()
        bs = []
        for _ in range(1000):
            k = keys[rng.integers(0, len(keys), len(keys))]
            m = pd.DataFrame(k, columns=["season", "week"]).merge(x, on=["season", "week"])
            bs.append(m["stream2"].mean() / m["predicted"].mean())
        print(f"{pos}: stream#2/predicted {x['stream2'].mean()/x['predicted'].mean():.2f} 95% [{np.percentile(bs,2.5):.2f}, {np.percentile(bs,97.5):.2f}]")
