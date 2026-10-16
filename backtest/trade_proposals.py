"""Every trade proposal in league history (2021-2026) vs how it was answered.

Question: do accepted proposals look better *to the responder*, and by which lens --
our team-context points model, or a naive "sum of ESPN projections" view (roughly
what the ESPN app / trade analyzers show)? n is tiny (~20 answered proposals),
so this is descriptive, not a fitted acceptance model.
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


def outcomes() -> pd.DataFrame:
    d = pd.read_csv(v.HIST / "trade_transactions_all.csv")
    props = d[d["type"] == "TRADE_PROPOSAL"]
    resp = d[d["type"] != "TRADE_PROPOSAL"].drop_duplicates("tx_id")
    res = {}
    for _, r in resp.iterrows():
        o = {"TRADE_ACCEPT": "accepted", "TRADE_UPHOLD": "accepted", "TRADE_DECLINE": "declined", "TRADE_VETO": "vetoed"}.get(r["type"])
        if o and isinstance(r["related"], str):
            prev = res.get(r["related"])
            res[r["related"]] = "vetoed" if "vetoed" in (o, prev) else o
    out = []
    for tx, g in props.groupby("tx_id"):
        g = g.dropna(subset=["espn_id"])
        if g.empty:
            continue
        proposer = int(g["proposer"].iloc[0])
        teams = set(g["from_team"].astype(int)) | set(g["to_team"].astype(int))
        responder = [t for t in teams if t != proposer]
        if len(responder) != 1:
            continue
        out.append({"season": int(g["season"].iloc[0]), "week": int(g["week"].iloc[0]), "tx_id": tx,
                    "proposer": proposer, "responder": responder[0],
                    "to_responder": g.loc[g["to_team"] == responder[0], "espn_id"].astype(int).tolist(),
                    "to_proposer": g.loc[g["to_team"] == proposer, "espn_id"].astype(int).tolist(),
                    "outcome": res.get(tx, "no answer / withdrawn")})
    return pd.DataFrame(out)


def score(df: pd.DataFrame) -> pd.DataFrame:
    rows = pd.read_csv(OUT / "player_rows.csv")
    recs = []
    for _, t in df.iterrows():
        yr, W = t["season"], t["week"]
        s = v.load_season(yr)
        p = ms.fit_all(rows, [y for y in ms.SEASONS if y != yr]) if yr in ms.SEASONS else v.ForecastParams.load()
        r = s.roster_at(W)
        tbl = v.player_table(s, W, p, rostered=set(r["espn_id"]))
        rost = {int(k): g["espn_id"].astype(int).tolist() for k, g in r.groupby("team_id")}
        repl = v.waiver_replacement(tbl, [i for ids in rost.values() for i in ids], scale=p.waiver_scale)
        weeks = range(W, v.FANTASY_LAST_WEEK + 1)
        espn = s.espn_weekly[s.espn_weekly["week"] == W].set_index("espn_id")["espn_proj"] if s.espn_weekly is not None else pd.Series(dtype=float)
        games = lambda i: sum(1 for w in weeks if w != s.byes.get(i))
        naive = lambda ids: sum(float(espn.get(i, 0.0)) * games(i) for i in ids)
        rec = dict(t)
        for side, get, give in (("responder", t["to_responder"], t["to_proposer"]), ("proposer", t["to_proposer"], t["to_responder"])):
            team = t[side]
            before = [i for i in rost.get(team, []) if i in tbl.index]
            if not set(give) <= set(rost.get(team, [])):  # roster changed before the proposal week snapshot
                rec[f"{side}_ours"] = np.nan
                rec[f"{side}_naive_espn"] = np.nan
                continue
            after = [i for i in before if i not in give] + [i for i in get if i in tbl.index]
            rec[f"{side}_ours"] = v.team_value(after, tbl, s.slots, weeks, repl, sims=150) - v.team_value(before, tbl, s.slots, weeks, repl, sims=150)
            rec[f"{side}_naive_espn"] = naive(get) - naive(give)
        rec["players_to_responder"] = ", ".join(s.name.get(i, str(i)) for i in t["to_responder"])
        rec["players_to_proposer"] = ", ".join(s.name.get(i, str(i)) for i in t["to_proposer"])
        recs.append(rec)
    return pd.DataFrame(recs)


if __name__ == "__main__":
    df = score(outcomes())
    df.drop(columns=["to_responder", "to_proposer"]).to_csv(OUT / "trade_proposals_scored.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["season", "week", "outcome", "responder_ours", "responder_naive_espn", "proposer_ours", "players_to_responder", "players_to_proposer"]
    print(df[cols].round(1).sort_values(["outcome", "season"]).to_string(index=False))
    print()
    print(df.groupby("outcome")[["responder_ours", "responder_naive_espn", "proposer_ours"]].agg(["median", "count"]).round(1).to_string())
