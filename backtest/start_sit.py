"""Start/sit backtest: how many points would each lineup-setting rule have scored?

For every team-week 2021-2026 (league scoring, actual points; DNP = 0):
- manager:  the lineup the manager actually started
- espn:     start whoever ESPN projected highest that week (its published number)
- v2:       start by v2's week forecast (ESPN-blended est x next-game availability)
- ours:     start by v2 without ESPN (preseason prior shrunk to observed ppg)
- optimal:  hindsight-best lineup from the same roster (ceiling)
Players in IR slots can't start. v2/ours use params fit without the season being
scored (leave-one-season-out); 2026 uses all-season params.

League size: 2021-2025 are 12-team seasons, 2026 is 10 teams. Compared within
the same team-week, so the size change doesn't bias any rule against another.
"""
from __future__ import annotations

import dataclasses
import itertools
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import valuation as v  # noqa: E402
from backtest import multi_season as ms  # noqa: E402

OUT = ROOT / "history" / "backtest" / "multi"


def choose(cands: List[Tuple[int, str, float]], slots: Sequence[str]) -> List[int]:
    """Pick ids maximizing predicted points for the slots (single-position slots
    first, then flex slots in both orders -- same logic as valuation.best_lineup_points)."""
    pools: Dict[str, List[Tuple[float, int]]] = {}
    for pid, pos, pred in cands:
        pools.setdefault(pos, []).append((pred, pid))
    for p in pools:
        pools[p].sort(reverse=True)
    chosen, flex = [], []
    for slot in slots:
        elig = v.SLOT_ELIGIBILITY[slot]
        if len(elig) == 1:
            if pools.get(elig[0]):
                chosen.append(pools[elig[0]].pop(0))
        else:
            flex.append(slot)
    best, best_ids = -1.0, []
    for order in set(itertools.permutations(flex)):
        pl = {p: list(x) for p, x in pools.items()}
        picks = []
        for slot in order:
            c = [(pl[p][0], p) for p in v.SLOT_ELIGIBILITY[slot] if pl.get(p)]
            if c:
                (val, pid), p = max(c)
                picks.append((val, pid))
                pl[p].pop(0)
        tot = sum(x[0] for x in picks)
        if tot > best:
            best, best_ids = tot, picks
    return [pid for _, pid in chosen + best_ids]


def run_season(season: int, rows: pd.DataFrame, weeks=range(1, 18)) -> pd.DataFrame:
    s = v.load_season(season)
    p = ms.fit_params(rows[rows["season"] != season]) if season in ms.SEASONS else v.ForecastParams.load()  # waiver scale irrelevant for lineups
    p_ours = dataclasses.replace(p, espn_weight={k: 0.0 for k in p.espn_weight})
    s_ours = dataclasses.replace(s, espn_weekly=None)
    wk = s.weekly
    out = []
    for W in weeks:
        if W > int(s.rosters["week"].max()) or W > int(wk["week"].max()):
            break
        actual = wk[wk["week"] == W].set_index("espn_id")["points"].where(wk[wk["week"] == W].set_index("espn_id")["played"], 0.0)
        espn = s.espn_weekly[s.espn_weekly["week"] == W].set_index("espn_id")["espn_proj"] if s.espn_weekly is not None else pd.Series(dtype=float)
        t2 = v.player_table(s, W, p)
        t0 = v.player_table(s_ours, W, p_ours)
        pred2 = (t2["est_ppg"] * t2["avail_now"]).where(t2["bye"] != W, 0.0)
        pred0 = (t0["est_ppg"] * t0["avail_now"]).where(t0["bye"] != W, 0.0)
        r = s.roster_at(W)
        for team, g in r.groupby("team_id"):
            g = g[g["lineup_slot"] != "IR"]
            ids = [int(i) for i in g["espn_id"]]
            pos = {int(i): s.position.get(int(i)) for i in ids}
            act = {i: float(actual.get(i, 0.0)) for i in ids}
            rec = {"season": season, "week": W, "team_id": int(team)}
            rec["manager"] = sum(act[int(i)] for i, sl in zip(g["espn_id"], g["lineup_slot"]) if sl not in ("BE", "IR"))
            for name, pred in (("espn", {i: float(espn.get(i, 0.0)) for i in ids}),
                               ("v2", {i: float(pred2.get(i, 0.0)) for i in ids}),
                               ("ours", {i: float(pred0.get(i, 0.0)) for i in ids}),
                               ("optimal", act)):
                picks = choose([(i, pos[i], pred[i]) for i in ids if pos[i] in v.POSITIONS], s.slots)
                rec[name] = sum(act[i] for i in picks)
            out.append(rec)
    return pd.DataFrame(out)


if __name__ == "__main__":
    rows = pd.read_csv(OUT / "player_rows.csv")
    for yr in [int(a) for a in sys.argv[1:]]:
        df = run_season(yr, rows)
        df.to_csv(OUT / f"start_sit_{yr}.csv", index=False)
        print(yr, len(df), df[["manager", "espn", "v2", "ours", "optimal"]].mean().round(2).to_dict())
