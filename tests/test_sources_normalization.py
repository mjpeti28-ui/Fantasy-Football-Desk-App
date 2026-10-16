"""Cross-source normalization: team codes, D/ST names, FantasyPros payload -> CSV rows.

These cover the joins that silently broke when FantasyPros replaced PFF as
the rankings source (every roster D/ST unmatched; JAC vs JAX; ARZ/BLT/CLV/HST/LA
strength-of-schedule codes never matching rankings).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from teams import canonical_dst_name, canonical_team, looks_like_dst  # noqa: E402


@pytest.mark.parametrize(
    "code,expected",
    [("JAC", "JAX"), ("ARZ", "ARI"), ("BLT", "BAL"), ("CLV", "CLE"), ("HST", "HOU"), ("LA", "LAR"), ("WSH", "WAS"), ("sf", "SF"), ("FA", "FA")],
)
def test_canonical_team(code, expected):
    assert canonical_team(code) == expected


@pytest.mark.parametrize(
    "name,team,expected",
    [
        ("Houston Texans", "HOU", "Texans D/ST"),
        ("Los Angeles Rams", None, "Rams D/ST"),  # shared city; nickname disambiguates
        ("San Francisco 49ers", None, "49ers D/ST"),
        ("49ers D/ST", None, "49ers D/ST"),
        ("SEA DST", None, "Seahawks D/ST"),
        ("Seattle Defense", None, "Seahawks D/ST"),
        ("Jacksonville Jaguars", "JAC", "Jaguars D/ST"),
        ("Josh Allen", None, None),
    ],
)
def test_canonical_dst_name(name, team, expected):
    assert canonical_dst_name(name, team) == expected


def test_looks_like_dst_does_not_flag_players():
    assert looks_like_dst("Rams D/ST")
    assert looks_like_dst("Houston Texans")
    assert not looks_like_dst("Josh Allen")
    assert not looks_like_dst("Jaxon Smith-Njigba")


def _payload(position, players, scoring="PPR"):
    return {"ranking_type_name": "ros", "position_id": position, "scoring": scoring, "players": players}


def test_build_rows_rank_and_posrank_consistent_and_dst_named():
    import fetch_fantasypros as fp

    overall = _payload("ALL", [
        {"player_id": 1, "rank_ecr": 1},
        {"player_id": 3, "rank_ecr": 2},
        {"player_id": 2, "rank_ecr": 3},
        {"player_id": 10, "rank_ecr": 4},
    ])
    by_pos = {
        # Position board disagrees with the overall board on 2 vs 3 -- the overall order must win.
        "RB": _payload("RB", [
            {"player_id": 1, "player_name": "A", "player_team_id": "JAC", "pos_rank": "RB1"},
            {"player_id": 2, "player_name": "B", "player_team_id": "DET", "pos_rank": "RB2"},
            {"player_id": 3, "player_name": "C", "player_team_id": "DET", "pos_rank": "RB3"},
            {"player_id": 4, "player_name": "Deep", "player_team_id": "DET", "pos_rank": "RB4"},
        ]),
        "DST": _payload("DST", [
            {"player_id": 10, "player_name": "Houston Texans", "player_team_id": "HOU", "pos_rank": "DST1"},
        ], scoring="STD"),
    }
    rows = fp.build_rows(overall, by_pos)
    names = [r["Name"] for r in rows]
    assert names == ["A", "C", "B", "Texans D/ST", "Deep"]
    assert [r["Rank"] for r in rows] == [1, 2, 3, 4, 5]
    rb = [r["Position Rank"] for r in rows if r["Position"] == "RB"]
    assert rb == [1, 2, 3, 4]
    assert rows[0]["Team"] == "JAX"
    assert all(r["Projected Points"] is None for r in rows)


def test_check_payload_rejects_wrong_board():
    import fetch_fantasypros as fp

    with pytest.raises(RuntimeError):
        fp._check_payload(_payload("RB", [{"x": 1}], scoring="STD"), "RB")
    with pytest.raises(RuntimeError):
        fp._check_payload(_payload("WR", [{"x": 1}]), "RB")
    fp._check_payload(_payload("K", [{"x": 1}], scoring="STD"), "K")


def test_projection_fallback_and_imputation(tmp_path):
    import data

    rankings = pd.DataFrame({
        "Rank": [1, 2, 3, 4],
        "Position": ["RB", "RB", "RB", "DST"],
        "Name": ["Kenny Gainwell", "Known Back", "Unknown Back", "Tampa Bay Buccaneers"],
        "Team": ["TB", "DET", "DET", "TB"],
    })
    projections = pd.DataFrame({
        "playerName": ["Kenneth Gainwell", "Known Back", "Buccaneers D/ST"],
        "teamName": ["TB", "DET", "TB"],
        "position": ["rb", "rb", "dst"],
        "fantasyPoints": [200.0, 150.0, 100.0],
    })
    r_path, p_path = tmp_path / "r.csv", tmp_path / "p.csv"
    rankings.to_csv(r_path, index=False)
    projections.to_csv(p_path, index=False)
    df = data.load_rankings(str(r_path), str(p_path)).set_index("Name")
    assert df.loc["Kenny Gainwell", "ProjPoints"] == 200.0  # last-name+team+pos fallback
    assert df.loc["Buccaneers D/ST", "ProjPoints"] == 100.0  # DST canonicalized on both sides
    # Missing projection interpolated from the position curve (clamped past the last known rank).
    assert df.loc["Unknown Back", "ProjPoints"] == 150.0
