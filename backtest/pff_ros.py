"""Does PFF's rest-of-season board add anything over FantasyPros' consensus?

2025 is the only season with saved PFF ROS boards (history/rankings/2025/*_ROS_week_2_PFF_rankings.csv,
Sep 24 - Nov 5, ~191 QB/RB/WR/TE). Each board is assigned to the first week whose Thursday game it
precedes, then scored like ``multi_season.fit_ecr_weight``: within each week x position, rank-match
the experts' order onto our own per-game estimates, blend at weight w, and take the Spearman
correlation with rest-of-season ppg (players with >= 3 later games). Only players ranked by BOTH
PFF and FantasyPros are scored, so the sources are compared on the same set.

    python backtest/pff_ros.py
"""
from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import normalize_name  # noqa: E402

SEASON = 2025
WEEK1_THURSDAY = date(2025, 9, 4)
BOARDS = ROOT / "history" / "rankings" / str(SEASON)
ROWS = ROOT / "history" / "backtest" / "multi" / f"usage_rows_{SEASON}.csv"
W = 1 / 3  # the live model's ecr_weight


def board_week(stamp: str) -> int:
    """First week whose Thursday comes after the board was saved."""
    d = datetime.strptime(stamp[:8], "%Y%m%d").date()
    return (d - WEEK1_THURSDAY).days // 7 + 2


def pff_boards() -> pd.DataFrame:
    latest = {}  # several saves can land on one week: keep the latest (files sort by stamp)
    for f in sorted(BOARDS.glob("*_ROS_week_2_PFF_rankings.csv")):
        latest[board_week(f.name)] = f
    out = []
    for wk, f in latest.items():
        b = pd.read_csv(f)
        b.columns = ["rank", "position", "name", "team"]
        b["week"] = wk
        out.append(b)
    b = pd.concat(out, ignore_index=True)
    b["pff"] = b.groupby(["week", "position"])["rank"].rank(method="first")
    b["name_norm"] = b["name"].map(normalize_name)
    return b[["week", "position", "name_norm", "pff"]]


def spearman(a: pd.Series, b: pd.Series) -> float:
    return float(a.rank().corr(b.rank()))


def rank_matched(g: pd.DataFrame, col: str) -> np.ndarray:
    vals = np.sort(g["est_ppg"].to_numpy())[::-1]
    order = g[col].rank(method="first").astype(int).to_numpy() - 1
    return vals[order]


def main() -> int:
    import nflverse_data as nd

    rows = pd.read_csv(ROWS)
    rows = rows[(rows["est_ppg"] > 0) & (rows["future_played"] >= 3)].copy()
    rows["name_norm"] = rows["name"].map(normalize_name)

    E = nd.ecr_table().dropna(subset=["espn_id"])
    E = E[(E["season"] == SEASON) & E["page_type"].isin(["redraft-qb", "redraft-rb", "redraft-wr", "redraft-te"])]
    fp = E.assign(espn_id=E["espn_id"].astype("int64")).groupby(["week", "espn_id"])["ecr"].min().rename("fp").reset_index()

    pff = pff_boards()
    x = rows.merge(fp, on=["week", "espn_id"]).merge(pff, on=["week", "position", "name_norm"])
    print(f"PFF boards cover weeks {sorted(pff['week'].unique())}; scored {len(x)} player-weeks "
          f"(ranked by both, >= 3 later games)\n")

    preds = {}
    for _, g in x.groupby(["week", "position"]):
        g = g.copy()
        g["fp_ppg"], g["pff_ppg"] = rank_matched(g, "fp"), rank_matched(g, "pff")
        g["avg"] = (g["fp"].rank() + g["pff"].rank()) / 2
        g["avg_ppg"] = rank_matched(g, "avg")
        preds[g.index[0]] = g
    x = pd.concat(preds.values())
    x["v2"] = x["est_ppg"]
    x["v2 + FP (live)"] = (1 - W) * x["est_ppg"] + W * x["fp_ppg"]
    x["v2 + PFF"] = (1 - W) * x["est_ppg"] + W * x["pff_ppg"]
    x["v2 + avg(FP, PFF)"] = (1 - W) * x["est_ppg"] + W * x["avg_ppg"]
    x["FP alone"], x["PFF alone"] = x["fp_ppg"], x["pff_ppg"]
    cols = ["v2", "FP alone", "PFF alone", "v2 + FP (live)", "v2 + PFF", "v2 + avg(FP, PFF)"]

    res = {}
    for (wk, pos), g in x.groupby(["week", "position"]):
        if len(g) > 5:
            res[(wk, pos)] = {c: spearman(g[c], g["future_ppg"]) for c in cols}
    r = pd.DataFrame(res).T
    r.index.names = ["week", "position"]
    pd.set_option("display.width", 200)
    print("Mean within-week rank corr with rest-of-season ppg (higher is better)")
    print(r.groupby(level="position").mean().round(3).to_string())
    print("\nAll positions:", r.mean().round(3).to_dict())
    print("\nWeeks where v2+PFF beat v2+FP:", int((r["v2 + PFF"] > r["v2 + FP (live)"]).sum()), "of", len(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
