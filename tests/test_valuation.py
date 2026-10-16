"""valuation.py: lineup optimizer, simulation invariants, forecaster shrinkage (offline)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import valuation as v  # noqa: E402

SLOTS = ["QB", "RB", "RB/WR", "WR", "WR/TE", "TE", "DST", "K", "FLEX"]


def test_best_lineup_fills_flex_optimally():
    pts = {"QB": [20], "RB": [15, 10], "WR": [18, 12, 11], "TE": [9], "K": [8], "DST": [7]}
    # 20 + 15 + 18 + 9 + 7 + 8 = 77 fixed; flex slots take 12, 11, 10
    assert v.best_lineup_points(pts, SLOTS) == 110


def test_best_lineup_handles_missing_positions():
    assert v.best_lineup_points({"QB": [20]}, SLOTS) == 20


def _tbl():
    return pd.DataFrame({
        "position": ["QB", "RB", "RB", "WR", "WR", "TE"],
        "est_ppg": [20.0, 15.0, 8.0, 14.0, 9.0, 7.0],
        "avail_now": [1.0, 0.5, 1.0, 1.0, 1.0, 1.0],
        "avail": [1.0, 0.5, 1.0, 1.0, 1.0, 1.0],
        "bye": [None, 5, None, None, None, None],
        "from_week": [4] * 6,
        "name": list("ABCDEF"),
    }, index=[1, 2, 3, 4, 5, 6])


def test_team_value_is_order_invariant_and_values_depth():
    t = _tbl()
    repl = {p: 0.0 for p in v.POSITIONS}
    a = v.team_value([1, 2, 3, 4, 5, 6], t, SLOTS, range(4, 8), repl, sims=200)
    b = v.team_value([6, 5, 4, 3, 2, 1], t, SLOTS, range(4, 8), repl, sims=200)
    assert a == b  # common random numbers keyed by player
    no_depth = v.team_value([1, 2, 4, 5, 6], t, SLOTS, range(4, 8), repl, sims=200)
    assert a > no_depth  # the backup RB covers the fragile starter and the bye


def test_bye_week_scores_zero():
    t = _tbl()
    repl = {p: 0.0 for p in v.POSITIONS}
    only_rb = v.team_value([2], t.assign(avail=1.0, avail_now=1.0), SLOTS, [5], repl, sims=10)
    assert only_rb == 0.0


def test_player_table_uses_only_past_weeks_and_shrinks():
    weekly = pd.DataFrame({
        "espn_id": [1, 1, 1, 2, 2, 2], "week": [1, 2, 3, 1, 2, 3],
        "points": [20.0, 20.0, 99.0, 0.0, 0.0, 0.0], "played": [True, True, True, True, False, False],
        "name": ["A"] * 3 + ["B"] * 3, "position": ["WR"] * 6,
    })
    rosters = pd.DataFrame({"week": [3, 3], "team_id": [1, 1], "espn_id": [1, 2], "lineup_slot": ["WR", "BE"],
                            "injury_status": ["ACTIVE", "OUT"]})
    s = v.Season(season=1999, weekly=weekly, prior_total={1: 170.0, 2: 170.0}, rosters=rosters,
                 team_weekly=pd.DataFrame(), settings={"lineup_slot_counts": {"WR": 1}}, byes={1: None, 2: None},
                 position={1: "WR", 2: "WR"}, name={1: "A", 2: "B"})
    p = v.ForecastParams(k={pos: 2.0 for pos in v.POSITIONS})
    t = v.player_table(s, 3, p)
    # week 3's 99 points must not leak into a week-3 forecast; prior 10/game, observed 20 over 2 games
    assert t.at[1, "n"] == 2 and t.at[1, "est_ppg"] == 15.0
    # B didn't play last week -> "missed_last"; the historical fit never sees ESPN's tag
    assert t.at[2, "status"] == "missed_last"
    live = v.player_table(s, 3, p, live_injuries=True)
    assert live.at[2, "avail_now"] == 0.0


def _write_season(tmp_path, season_proj, weekly_proj):
    import json

    d = tmp_path / "1999"
    d.mkdir()
    ids = list(range(1, 11))
    rows = [{"espn_id": i, "name": f"P{i}", "position": "WR", "pro_team_id": 1, "week": w, "points": 10.0,
             "played": True, "kind": "actual"} for i in ids for w in (1, 2, 3)]
    rows += [{"espn_id": i, "name": f"P{i}", "position": "WR", "pro_team_id": 1, "week": 0,
              "points": season_proj[i - 1], "played": None, "kind": "espn_season_proj"} for i in ids]
    pd.DataFrame(rows).to_csv(d / "players_weekly.csv", index=False)
    pd.DataFrame([{"week": w, "team_id": 1, "espn_id": i, "name": f"P{i}", "position": "WR", "lineup_slot": "WR",
                   "injury_status": None} for i in ids for w in (1, 2, 3)]).to_csv(d / "rosters_by_week.csv", index=False)
    pd.DataFrame([{"round": 1, "pick": i, "team_id": 1, "espn_id": i, "keeper": False} for i in ids]).to_csv(d / "draft.csv", index=False)
    pd.DataFrame(weekly_proj).to_csv(d / "espn_weekly_proj.csv", index=False)
    pd.DataFrame(columns=["week", "team_id", "points", "playoff"]).to_csv(d / "team_weekly.csv", index=False)
    (d / "settings.json").write_text(json.dumps({"lineup_slot_counts": {"WR": 1}, "regular_season_weeks": 14}))


def test_bad_season_projections_and_broken_week_are_dropped(tmp_path, monkeypatch):
    monkeypatch.setattr(v, "HIST", tmp_path)
    v.load_season.cache_clear()
    # 9 of 10 drafted players have a 0 season projection -> treat season projections as missing
    # week 1: every weekly projection is 0 (broken) -> dropped; weeks 2-3 fine
    weekly = [{"week": 1, "espn_id": i, "espn_proj": 0.0} for i in range(1, 11)]
    weekly += [{"week": w, "espn_id": i, "espn_proj": 12.0} for i in range(1, 11) for w in (2, 3)]
    _write_season(tmp_path, [0.0] * 9 + [300.0], weekly)
    s = v.load_season(1999)
    assert s.prior_total == {}
    assert 1 not in set(s.espn_weekly["week"])
    t = v.player_table(s, 3, v.ForecastParams())
    # fallback prior = mean of ESPN weekly projections published through week 3
    assert abs(t.at[1, "prior_ppg"] - 12.0) < 1e-9
    v.load_season.cache_clear()


def test_espn_zero_lowers_later_availability_more_for_free_agents():
    """An ESPN 0 in a game week cuts this week (espn_zero_now) and, with espn_zero_later
    fitted, every later week too -- harder for a player nobody rosters."""
    weekly = pd.DataFrame({
        "espn_id": [1, 1, 2, 2, 3, 3], "week": [1, 2] * 3, "points": [15.0] * 6, "played": [True] * 6,
        "name": list("AABBCC"), "position": ["WR"] * 6,
    })
    rosters = pd.DataFrame({"week": [3, 3], "team_id": [1, 1], "espn_id": [1, 2], "lineup_slot": ["WR", "BE"]})
    proj = pd.DataFrame({"week": [3, 3, 3], "espn_id": [1, 2, 3], "espn_proj": [14.0, 0.0, 0.0]})
    s = v.Season(season=1999, weekly=weekly, prior_total={i: 170.0 for i in (1, 2, 3)}, rosters=rosters,
                 team_weekly=pd.DataFrame(), settings={"lineup_slot_counts": {"WR": 1}}, byes={1: 9, 2: 9, 3: 9},
                 position={i: "WR" for i in (1, 2, 3)}, name={1: "A", 2: "B", 3: "C"}, espn_weekly=proj)
    p = v.ForecastParams(espn_zero_later={"WR": {"rostered": 0.55, "free_agent": 0.3}})
    t = v.player_table(s, 3, p, rostered={1, 2})
    base = p.availability["WR"]["later"]["active"]
    assert t.at[1, "avail"] == base                       # projected normally: untouched
    assert t.at[2, "avail"] == 0.55 and t.at[3, "avail"] == 0.3
    assert t.at[2, "avail_now"] == p.espn_zero_now["WR"]
    # rostered set unknown -> everyone treated as rostered
    assert v.player_table(s, 3, p).at[3, "avail"] == 0.55
    # params without the field keep the old behavior
    assert v.player_table(s, 3, v.ForecastParams(), rostered={1, 2}).at[3, "avail"] == base


def test_team_value_levels_reduce_to_point_model():
    """With level_sd = 0 (or tiny), the belief/truth lineup path matches the point model."""
    t = _tbl()
    slots = ["QB", "RB", "WR", "TE", "FLEX"]
    repl = {"QB": 10.0, "RB": 5.0, "WR": 5.0, "TE": 4.0}
    ids = list(t.index)
    base = v.team_value(ids, t, slots, range(4, 8), repl, sims=40, seed=1, levels=False)
    t2 = t.assign(level_sd=1e-9, learn_k=6.0)
    assert np.isclose(v.team_value(ids, t2, slots, range(4, 8), repl, sims=40, seed=1, levels=True), base, rtol=1e-6)
    # real uncertainty raises the value of a roster with bench depth (options), never below the point model by much
    t3 = t.assign(level_sd=0.6, learn_k=1.0)
    assert v.team_value(ids, t3, slots, range(4, 8), repl, sims=400, seed=1, levels=True) > base * 0.98


def test_best_lineup_bt_chooses_on_belief_scores_truth():
    pools = {"QB": [(20.0, 5.0), (10.0, 30.0)]}
    assert v.best_lineup_points_bt(pools, ["QB"]) == 5.0
