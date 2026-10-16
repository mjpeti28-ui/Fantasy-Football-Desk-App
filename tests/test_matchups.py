"""Matchups engine: shapes, shrinkage behaviour and the invariants the screen relies on. Needs the cached
nflverse tables (python -m app.engine.matchups_data 2023 2024 2025 <current>); skipped when they aren't built."""
from __future__ import annotations

import json
import math

import pytest

pytest.importorskip("pandas")
from app.engine import matchups as m, matchups_data as md  # noqa: E402

pytestmark = pytest.mark.skipif(
    not all((md.cache_dir() / f"{y}.csv.gz").exists() for y in (*md.HISTORY, m.current_season())),
    reason="matchup tables not built",
)


def test_board_has_32_defenses_and_valid_rates():
    b = m.board()
    assert len(b["teams"]) == 32
    for a, t in b["teams"].items():
        assert 0.0 <= t["man"] <= 1.0, a
        assert 0.0 <= t["blitz"] <= 1.0, a
    json.dumps(b, default=float)


def test_defense_shell_mix_sums_to_one():
    d = m.defense("DAL")
    for h in d["history"]:
        assert math.isclose(sum(h["shell"].values()), 1.0, abs_tol=1e-6)
    assert math.isclose(sum(d["expected"]["shell"].values()), 1.0, abs_tol=1e-6)
    # expected scheme moves toward the league only partway: it lies between last season and that season's league mean
    last, lg = d["history"][-1]["man"], d["league"]["base"]["man"]
    lo, hi = sorted((last, lg))
    assert lo - 1e-9 <= d["expected"]["man"] <= hi + 1e-9


def test_player_split_is_shrunk_toward_the_position_mean():
    ids = m.tables()["ids"]
    g = ids.index[ids["name"] == "CeeDee Lamb"][0]
    p = m.player(g, "PHI")
    h = p["hist"]
    assert 0.0 <= h["tr_rel"] <= 1.0
    # shrunk difference sits between the position mean and the raw difference
    mean_d = m.tables()["eb"]["WR"]["tr_mz"]["mean"]
    lo, hi = sorted((mean_d, h["tr_diff"]))
    assert lo - 1e-9 <= h["tr_diff_shrunk"] <= hi + 1e-9
    assert p["lens"]["opp"] == "PHI"


def test_efficiency_splits_are_labelled_unreliable():
    """The measured PPR/target man-vs-zone split carries almost no signal; the screen must not oversell it."""
    for pos in ("WR", "TE", "RB"):
        assert m.tables()["eb"][pos]["ppr_mz"]["tau2"] >= 0.0
    ids = m.tables()["ids"]
    g = ids.index[ids["name"] == "CeeDee Lamb"][0]
    assert m.player(g)["hist"]["ppr_rel"] < 0.35


def test_quarterback_and_missing_players():
    ids = m.tables()["ids"]
    g = ids.index[ids["name"] == "Josh Allen"][0]
    q = m.player(g, "MIA")
    assert q["pos"] == "QB" and "hist" in q
    assert m.player("00-0000000") is None
    assert m.defense("ZZZ") is None


def test_alias_and_schedule():
    assert m.norm("LAR") == "LA" and m.norm("wsh") == "WAS" and m.norm(None) is None
    sch = m.schedule(2026, 5)
    for a, g in sch.items():
        assert sch[g["opp"]]["opp"] == a  # opponents point at each other


def test_defenders_flag_departed_players():
    d = m.defenders("DAL")
    assert d["onoff"] and all("still" in r and "se" in r for r in d["onoff"])
    assert all((r["off_epa"] is None) == (r["se"] is None) for r in d["onoff"])  # no comparison without enough snaps off the field
    assert all(0.0 < r["pct"] <= 1.0 for r in d["onoff"])


def test_expected_man_rates_centre_on_the_league_baseline():
    """Teams are judged against their own season's league, so the 32 expectations average to the baseline (a pooled
    3-season mean made every team look light on man coverage)."""
    b = m.board()
    mean = sum(t["man"] for t in b["teams"].values()) / len(b["teams"])
    assert abs(mean - b["league"]["base"]["man"]) < 0.02
