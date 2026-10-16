"""Projection-accuracy tracking and input health checks (offline)."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import projection_tracking as pt  # noqa: E402


def test_week_calendar():
    # Snapshot on the Tuesday before week 3 projects from week 3.
    assert pt.first_week_after(datetime(2026, 9, 22, 21, tzinfo=timezone.utc)) == 3
    # Preseason snapshot projects from week 1.
    assert pt.first_week_after(datetime(2026, 8, 24, tzinfo=timezone.utc)) == 1
    # Friday of week 3 (after TNF) belongs to week 4.
    assert pt.first_week_after(datetime(2026, 9, 25, 12, tzinfo=timezone.utc)) == 4


def test_games_remaining_accounts_for_bye():
    assert pt.games_remaining(1, 7) == 17
    assert pt.games_remaining(3, 7) == 15
    assert pt.games_remaining(8, 7) == 11  # bye already passed


def test_score_end_to_end(tmp_path, monkeypatch):
    snap_dir = tmp_path / "snaps"
    snap_dir.mkdir()
    monkeypatch.setattr(pt, "SNAP_DIR", snap_dir)
    monkeypatch.setattr(pt, "LEGACY_PFF_DIR", tmp_path / "none")
    monkeypatch.setattr(pt, "PROJECTION_SOURCES", {"a": "a.csv", "b": "b.csv"})
    # Taken before week 1: full 17-game totals (bye week 10).
    rows = lambda pts: pd.DataFrame({
        "playerName": ["Good Back", "Bad Back", "Mid Back", "Kenneth Gainwell"],
        "teamName": ["DET", "DET", "DET", "TB"],
        "position": ["rb"] * 4,
        "byeWeek": [10] * 4,
        "fantasyPoints": pts,
    })
    rows([340, 170, 255, 170]).to_csv(snap_dir / "20260901T000000Z__a.csv", index=False)
    rows([170, 340, 255, 170]).to_csv(snap_dir / "20260901T000000Z__b.csv", index=False)
    actual = pd.DataFrame({
        "week": [1] * 5,
        "espn_id": [1, 2, 3, 4, 5],
        "name": ["Good Back", "Bad Back", "Mid Back", "Kenny Gainwell", "Hurt Guy"],
        "name_norm": ["good back", "bad back", "mid back", "kenny gainwell", "hurt guy"],
        "position": ["RB"] * 5,
        "fantasy_team": ["x"] * 5,
        "points": [20.0, 10.0, 15.0, 10.0, 0.0],
        "played": [True, True, True, True, False],
    })
    monkeypatch.setattr(pt, "load_actuals", lambda week, refresh=False: actual)
    result = pt.score(weeks=[1])
    s = result["summary"].set_index(["source", "position"])
    assert s.loc[("a", "RB"), "n"] == 4          # DNP excluded; Gainwell joined via last-name fallback
    assert s.loc[("a", "RB"), "mae"] == 0.0      # 340/17 = 20 etc.
    assert s.loc[("a", "RB"), "rank_corr"] > 0.9
    assert s.loc[("b", "RB"), "rank_corr"] < 0
    assert ("blend", "RB") in s.index


def test_snapshot_dedupes(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, "SNAP_DIR", tmp_path / "snaps")
    monkeypatch.setattr(pt, "LEGACY_PFF_DIR", tmp_path / "none")
    monkeypatch.setattr(pt, "ROOT", tmp_path)
    monkeypatch.setattr(pt, "PROJECTION_SOURCES", {"a": "a.csv"})
    (tmp_path / "a.csv").write_text("playerName,fantasyPoints\nX,1\n")
    assert len(pt.snapshot_projections(datetime(2026, 9, 1, tzinfo=timezone.utc))) == 1
    assert pt.snapshot_projections(datetime(2026, 9, 2, tzinfo=timezone.utc)) == []
    (tmp_path / "a.csv").write_text("playerName,fantasyPoints\nX,2\n")
    assert len(pt.snapshot_projections(datetime(2026, 9, 3, tzinfo=timezone.utc))) == 1


def test_health_flags_unmatched_and_imputed(tmp_path):
    import health

    rankings = pd.DataFrame({
        "Rank": [1, 2, 3], "Position": ["RB", "RB", "RB"],
        "Name": ["Known Back", "Unprojected Back", "Other Back"], "Team": ["DET"] * 3,
    })
    projections = pd.DataFrame({"playerName": ["Known Back", "Other Back"], "teamName": ["DET", "DET"],
                                "position": ["rb", "rb"], "fantasyPoints": [200.0, 100.0]})
    r, p = tmp_path / "r.csv", tmp_path / "p.csv"
    rankings.to_csv(r, index=False)
    projections.to_csv(p, index=False)
    rosters = {"Team_A": {"RB": ["Known Back", "Unprojected Back", "Nobody Atall"]}}
    findings = health.run_health_check(str(r), str(p), rosters=rosters, use_espn=False)
    by_check = {(f.level, f.check): f for f in findings}
    assert "Nobody Atall" in by_check[("ERROR", "matching")].players
    assert "Unprojected Back" in by_check[("WARN", "projections")].players


def test_espn_weekly_scored_by_espn_id(tmp_path, monkeypatch):
    snap_dir = tmp_path / "snaps"
    snap_dir.mkdir()
    monkeypatch.setattr(pt, "SNAP_DIR", snap_dir)
    monkeypatch.setattr(pt, "LEGACY_PFF_DIR", tmp_path / "none")
    monkeypatch.setattr(pt, "ACTUALS_DIR", tmp_path)
    monkeypatch.setattr(pt, "PROJECTION_SOURCES", {})
    pd.DataFrame({"week": [1, 1, 1, 2], "espn_id": [1, 2, 3, 1], "espn_proj": [20.0, 10.0, 15.0, 99.0]}).to_csv(tmp_path / "espn_weekly_proj.csv", index=False)
    actual = pd.DataFrame({"week": [1] * 3, "espn_id": [1, 2, 3], "name": ["A", "B", "C"], "name_norm": ["a", "b", "c"],
                           "position": ["RB"] * 3, "fantasy_team": ["x"] * 3, "points": [20.0, 10.0, 15.0], "played": [True] * 3})
    monkeypatch.setattr(pt, "load_actuals", lambda week, refresh=False: actual)
    s = pt.score(weeks=[1])["summary"].set_index(["source", "position"])
    assert s.loc[("espn_weekly", "RB"), "n"] == 3
    assert s.loc[("espn_weekly", "RB"), "mae"] == 0.0  # week-2 number must not leak into week 1
