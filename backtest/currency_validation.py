"""Does the points currency (valuation.py) predict team outcomes and trade outcomes
better than the current combined z-score? Leave-one-season-out, 2021-2025.

Two tests per season x checkpoint:
1. Team outcomes: predicted rest-of-regular-season lineup points vs actual points scored.
2. Hypothetical trades: random 1-for-1 and 2-for-1 swaps between two real rosters.
   Predicted change in each team's expected lineup points (weeks W-17) vs the
   realized change: best possible lineups each week using players' ACTUAL points,
   with and without the trade. (Hindsight-optimal lineups, so manager lineup
   mistakes don't pollute the comparison. The receiving side of a 2-for-1 drops
   its lowest-forecast player, chosen before anything is known.)

League size: 2021-2025 are 12-team seasons. Everything is ranked/compared
within a season, so the 10-team 2026 league doesn't change the logic, but
absolute point deltas from 12-team seasons shouldn't be read as 10-team numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import valuation as v  # noqa: E402
from backtest import multi_season as ms  # noqa: E402

OUT = ROOT / "history" / "backtest" / "multi"
SKILL = {"QB", "RB", "WR", "TE"}


def params_excluding(season: int, rows: pd.DataFrame) -> v.ForecastParams:
    return ms.fit_all(rows, [yr for yr in ms.SEASONS if yr != season])


TARGET = "hindsight"
DEPTH = 2
REPL_MULT: Dict[str, float] = {}
USE_WAIVER_SCALE = True  # or "espn": lineups picked each week by ESPN's projection, scored on actual points


_STREAM = {}


def stream_table(s: v.Season) -> Dict:
    """(week, pos) -> (ESPN projection, realized points) of the 2nd-best free agent by ESPN's
    projection for that week (pool: not rostered the week before). What streaming really gave."""
    if s.season in _STREAM:
        return _STREAM[s.season]
    d = v.HIST / str(s.season)
    pr = pd.read_csv(d / "espn_weekly_proj.csv")
    pr["position"] = pr["espn_id"].map(s.position)
    r = pd.read_csv(d / "rosters_by_week.csv")
    act = s.weekly[s.weekly["played"]].groupby(["espn_id", "week"])["points"].sum()
    out = {}
    for w in range(1, v.FANTASY_LAST_WEEK + 1):
        prev = set(r.loc[r["week"] == w - 1, "espn_id"])
        fa = pr[(pr["week"] == w) & ~pr["espn_id"].isin(prev) & (pr["espn_proj"] > 0)]
        for pos in v.POSITIONS:
            f = fa[fa["position"] == pos].sort_values("espn_proj", ascending=False)
            if len(f) > 1:
                i = int(f["espn_id"].iloc[1])
                out[(w, pos)] = (float(f["espn_proj"].iloc[1]), float(act.get((i, w), 0.0)))
    _STREAM[s.season] = out
    return out


def realized_lineup_points(s: v.Season, ids: List[int], weeks, replacement: Dict[str, float]) -> float:
    wk = s.weekly[s.weekly["week"].isin(list(weeks)) & s.weekly["played"] & s.weekly["espn_id"].isin(ids)]
    total = 0.0
    if TARGET == "stream":
        # realistic: ESPN-picked lineups, holes filled by the streamer actually available that
        # week (2nd-best free agent by ESPN's projection), scored on his real points
        from backtest.start_sit import choose

        st = stream_table(s)
        for w in weeks:
            act = wk[wk["week"] == w].set_index("espn_id")["points"]
            e = s.espn_weekly[s.espn_weekly["week"] == w].set_index("espn_id")["espn_proj"] if s.espn_weekly is not None else pd.Series(dtype=float)
            cands = [(int(i), s.position[int(i)], float(e.get(i, 0.0))) for i in ids if s.position.get(int(i)) in v.POSITIONS]
            streamers = {}
            for k, pos in enumerate(v.POSITIONS):
                if (w, pos) in st:
                    sid = -(k + 1) * 10**9
                    streamers[sid] = st[(w, pos)][1]
                    cands.append((sid, pos, st[(w, pos)][0]))
            picks = choose(cands, s.slots)
            total += sum(streamers[i] if i in streamers else float(act.get(i, 0.0)) for i in picks)
        return total
    if TARGET == "espn":
        from backtest.start_sit import choose

        for w in weeks:
            act = wk[wk["week"] == w].set_index("espn_id")["points"]
            e = s.espn_weekly[s.espn_weekly["week"] == w].set_index("espn_id")["espn_proj"] if s.espn_weekly is not None else pd.Series(dtype=float)
            cands = [(int(i), s.position[int(i)], float(e.get(i, 0.0))) for i in ids if s.position.get(int(i)) in v.POSITIONS]
            # a waiver streamer per position, valued at the replacement level, always available
            cands += [(-(k + 1) * 10**9, pos, replacement.get(pos, 0.0)) for k, pos in enumerate(v.POSITIONS)]
            picks = choose(cands, s.slots)
            streamers = {-(k + 1) * 10**9: pos for k, pos in enumerate(v.POSITIONS)}
            total += sum(replacement.get(streamers[i], 0.0) if i in streamers else float(act.get(i, 0.0)) for i in picks)
        return total
    for w in weeks:
        g = wk[wk["week"] == w]
        pts = {p: [replacement.get(p, 0.0)] for p in v.POSITIONS}
        for pid, val in zip(g["espn_id"], g["points"]):
            pts[s.position[int(pid)]].append(float(val))
        for p in pts:
            pts[p].sort(reverse=True)
        total += v.best_lineup_points(pts, s.slots)
    return total


def team_rosters(s: v.Season, week: int) -> Dict[int, List[int]]:
    r = s.roster_at(week)
    return {int(t): g["espn_id"].astype(int).tolist() for t, g in r.groupby("team_id")}


def old_combined(s: v.Season, week: int, tbl: pd.DataFrame, rosters: Dict[int, List[int]], tmp: Path) -> Dict[int, float]:
    """The current model's combined z-score, fed the same forecasts (as a projections file)."""
    import main

    proj = pd.DataFrame({
        "playerName": tbl["name"], "position": tbl["position"].str.lower(),
        "fantasyPoints": v.expected_ros_points(tbl, week, v.FANTASY_LAST_WEEK),
        "teamName": None,
    }).sort_values("fantasyPoints", ascending=False)
    proj.to_csv(tmp / "_proj.csv", index=False)
    pd.DataFrame({"Rank": range(1, len(proj) + 1), "Position": proj["position"].str.upper(),
                  "Name": proj["playerName"], "Team": None}).to_csv(tmp / "_rank.csv", index=False)
    custom = {}
    for t, ids in rosters.items():
        d = {p: [] for p in v.POSITIONS}
        for i in ids:
            if i in tbl.index:
                d[tbl.at[i, "position"]].append(tbl.at[i, "name"])
        custom[f"T{t}"] = d
    try:
        res = main.evaluate_league(str(tmp / "_rank.csv"), projections_path=str(tmp / "_proj.csv"),
                                   custom_rosters=custom, save_snapshot=False)
    except RuntimeError:  # the old optimizer refuses rosters that can't fill every slot
        return {}
    return {int(k[1:]): float(val) for k, val in res["combined_scores"].items()}


def run_season(season: int, rows: pd.DataFrame, checkpoints=(4, 6, 8, 10), n_trades=80, sims=100,
               with_old=True, seed=0) -> Dict[str, pd.DataFrame]:
    s = v.load_season(season)
    p = params_excluding(season, rows)
    rng = np.random.default_rng(seed + season)
    tmp = OUT / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    team_out, trade_out = [], []
    tw = s.team_weekly
    for W in checkpoints:
        rosters = team_rosters(s, W)
        rostered = [i for ids in rosters.values() for i in ids]
        tbl = v.player_table(s, W, p, rostered=rostered)
        repl = v.waiver_replacement(tbl, rostered, depth=DEPTH, scale=p.waiver_scale if USE_WAIVER_SCALE else None)
        mult = REPL_MULT
        if REPL_MULT.get("auto"):
            st = pd.read_csv(OUT / "streamer_test.csv")
            st = st[st["season"] != season]
            g = st.groupby("position")
            mult = (g["stream2"].mean() / g["predicted"].mean()).to_dict()
        repl = {k: val * mult.get(k, 1.0) for k, val in repl.items()}
        reg = range(W, s.reg_weeks + 1)
        full = range(W, v.FANTASY_LAST_WEEK + 1)
        old = old_combined(s, W, tbl, rosters, tmp) if with_old else {}
        for t, ids in rosters.items():
            team_out.append({
                "season": season, "week": W, "team_id": t,
                "pred_points": v.team_value(ids, tbl, s.slots, reg, repl, sims=sims, seed=1),
                "old_combined": old.get(t),
                "ppg_so_far": tw[(tw["team_id"] == t) & (tw["week"] < W)]["points"].mean(),
                "actual_points": tw[(tw["team_id"] == t) & (tw["week"] >= W) & (tw["week"] <= s.reg_weeks)]["points"].sum(),
            })
        # hypothetical trades
        exp = v.expected_ros_points(tbl, W, v.FANTASY_LAST_WEEK)
        teams = list(rosters)
        base_pred = {t: v.team_value(rosters[t], tbl, s.slots, full, repl, sims=sims, seed=2) for t in teams}
        base_real = {t: realized_lineup_points(s, rosters[t], full, repl) for t in teams}
        for _ in range(n_trades):
            a, b = rng.choice(teams, 2, replace=False)
            pa = [i for i in rosters[a] if tbl.at[i, "position"] in SKILL] if True else []
            pb = [i for i in rosters[b] if i in tbl.index and tbl.at[i, "position"] in SKILL]
            pa = [i for i in pa if i in tbl.index]
            if len(pa) < 2 or len(pb) < 1:
                continue
            two_for_one = rng.random() < 0.4
            give_a = list(rng.choice(pa, 2 if two_for_one else 1, replace=False))
            give_b = list(rng.choice(pb, 1, replace=False))
            new_a = [i for i in rosters[a] if i not in give_a] + give_b
            new_b = [i for i in rosters[b] if i not in give_b] + give_a
            if two_for_one:  # b received an extra body: drop its lowest-forecast player
                drop = min((i for i in new_b if i in exp.index and i not in give_a), key=lambda i: exp[i])
                new_b.remove(drop)
            rec = {"season": season, "week": W, "a": a, "b": b, "shape": "2-for-1" if two_for_one else "1-for-1"}
            for side, new in (("a", new_a), ("b", new_b)):
                t = a if side == "a" else b
                rec[f"pred_{side}"] = v.team_value(new, tbl, s.slots, full, repl, sims=sims, seed=2) - base_pred[t]
                rec[f"real_{side}"] = realized_lineup_points(s, new, full, repl) - base_real[t]
            if with_old and len(trade_out) % 4 == 0:  # the old model is slow: score a subset
                post = dict(rosters)
                post[a], post[b] = new_a, new_b
                oc = old_combined(s, W, tbl, post, tmp)
                if oc and old:
                    rec["old_a"], rec["old_b"] = oc[a] - old[a], oc[b] - old[b]
            trade_out.append(rec)
    return {"teams": pd.DataFrame(team_out), "trades": pd.DataFrame(trade_out)}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("seasons", nargs="+", type=int)
    ap.add_argument("--target", default="hindsight", choices=["hindsight", "espn", "stream"])
    ap.add_argument("--repl-mult", default="", help="e.g. QB=1.3,RB=1.5 multiplies the replacement level (variant test)")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--no-waiver-scale", action="store_true")
    ap.add_argument("--no-old", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    TARGET, DEPTH, USE_WAIVER_SCALE = a.target, a.depth, not a.no_waiver_scale
    REPL_MULT = {"auto": 1.0} if a.repl_mult == "auto" else {k: float(x) for k, x in (kv.split("=") for kv in a.repl_mult.split(",") if kv)}
    rows = pd.read_csv(OUT / "player_rows.csv")
    for yr in a.seasons:
        res = run_season(yr, rows, with_old=not a.no_old, seed=a.seed)
        for k, df in res.items():
            df.to_csv(OUT / f"currency_{k}_{yr}{a.tag}.csv", index=False)
        print(yr, a.tag or "(main)", {k: len(df) for k, df in res.items()})
