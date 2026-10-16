"""League manager tendencies, 2021-2025 (team ids persist across seasons in ESPN; names change).

Per current team id: trade proposals made/received/accepted, waiver adds per season and how
those adds did vs what v2 expected (skill), whether that skill persists year to year, bench
make-up by position, and lineup-setting (points left vs ESPN-projection lineups).
  python3 backtest/league_behavior.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "history" / "backtest" / "multi"
SEASONS = [2021, 2022, 2023, 2024, 2025]


def names():
    return json.loads((ROOT / "history" / "actuals" / "2026" / "settings.json").read_text())["team_names"]


def trades():
    t = pd.read_csv(ROOT / "history" / "actuals" / "trade_transactions_all.csv")
    t = t[t["type"] == "TRADE_PROPOSAL"].drop_duplicates("tx_id")
    prop = t.groupby("proposer").size().rename("proposed")
    # the counterparty is whichever team isn't the proposer on any item
    allrows = pd.read_csv(ROOT / "history" / "actuals" / "trade_transactions_all.csv")
    allrows = allrows[allrows["type"] == "TRADE_PROPOSAL"]
    other = allrows.assign(o=np.where(allrows["from_team"] == allrows["proposer"], allrows["to_team"], allrows["from_team"]))
    recv = other.drop_duplicates("tx_id").groupby("o").size().rename("received")
    acc = t[t["status"].isin(["EXECUTED", "ACCEPTED"])]
    acc_by = pd.concat([acc["proposer"], other[other["tx_id"].isin(acc["tx_id"])].drop_duplicates("tx_id")["o"]]).value_counts().rename("completed")
    return pd.concat([prop, recv, acc_by], axis=1).fillna(0).astype(int)


def adds():
    rows = []
    for y in SEASONS:
        f = OUT / f"usage_rows_{y}.csv"
        u = pd.read_csv(f) if f.exists() else None
        r = pd.read_csv(ROOT / "history" / "actuals" / str(y) / "rosters_by_week.csv")
        for W in range(2, 14):
            prev = set(zip(r.loc[r["week"] == W - 1, "espn_id"], r.loc[r["week"] == W - 1, "team_id"]))
            now = r[r["week"] == W]
            prev_ids = {i for i, _ in prev}
            new = now[~now["espn_id"].isin(prev_ids)]
            for _, x in new.iterrows():
                rec = {"season": y, "week": W, "team_id": int(x["team_id"]), "espn_id": int(x["espn_id"]), "position": x["position"]}
                if u is not None:
                    m = u[(u["week"] == W) & (u["espn_id"] == x["espn_id"])]
                    if len(m):
                        rec["pred"] = float(m["v2_total"].iloc[0])
                        rec["real"] = float(m["future_points"].iloc[0])
                rows.append(rec)
    a = pd.DataFrame(rows)
    return a


def bench():
    rows = []
    for y in SEASONS:
        r = pd.read_csv(ROOT / "history" / "actuals" / str(y) / "rosters_by_week.csv")
        b = r[r["lineup_slot"].isin(["BE", "Bench"])]
        c = b.groupby(["team_id", "position"]).size() / b.groupby("team_id")["week"].nunique()
        rows.append(c.rename("per_week").reset_index().assign(season=y))
    return pd.concat(rows).groupby(["team_id", "position"])["per_week"].mean().unstack().round(2)


def lineups():
    rows = []
    for y in SEASONS:
        f = OUT / f"start_sit_{y}.csv"
        if f.exists():
            rows.append(pd.read_csv(f).assign(season=y))
    return pd.concat(rows) if rows else None


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    nm = {int(k): v.strip() for k, v in names().items()}
    T = trades()
    T.index = T.index.astype(int)
    print("TRADES 2021-25 (unique proposals)\n", T.rename(index=nm).sort_values("proposed", ascending=False).to_string())
    A = adds()
    per = A.groupby(["team_id", "season"]).size().unstack(fill_value=0)
    print("\nADDS per season\n", per.rename(index=nm).to_string())
    s = A.dropna(subset=["pred"]).copy()
    s["resid"] = s["real"] - s["pred"]
    sk = s.groupby(["team_id", "season"])["resid"].mean().unstack()
    print("\nADD QUALITY: realized minus v2-expected ROS points per add (+ = beat the model)\n", sk.rename(index=nm).round(1).to_string())
    pairs = [(sk[a], sk[b]) for a, b in zip(SEASONS[:-1], SEASONS[1:])]
    cor = [pd.concat(p, axis=1).dropna().corr(method="spearman").iloc[0, 1] for p in pairs]
    print("year-to-year persistence of add quality (rank corr by season pair):", np.round(cor, 2), "mean", round(float(np.nanmean(cor)), 2))
    print("\nBENCH make-up (avg bench players by position per week, 2021-25)\n", bench().rename(index=nm).to_string())
    L = lineups()
    if L is not None:
        print("\nstart_sit columns:", list(L.columns)[:12])
