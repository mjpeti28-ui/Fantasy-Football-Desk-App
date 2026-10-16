"""PFF ROS board flattening, on the payload shape pff.com served on 2026-10-16."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("fetch_pff", Path(__file__).resolve().parents[1] / "tools" / "fetch_pff.py")
fetch_pff = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch_pff)


def _entry(pid, first, last, team, bye, current, pos_rank, points):
    mid, low, high = points if points else (None, None, None)
    return {"playerId": pid, "firstName": first, "lastName": last, "position": "RB", "teamAbbreviation": team,
            "byeWeek": bye, "rank": {"current": current, "position": pos_rank},
            "projection": {"points": {"mid": mid, "low": low, "high": high}}}


def test_build_ros_rows_keeps_ranks_and_labels_next_week_points():
    payload = {"currentGameWeek": 6, "rankings": [
        _entry(123126, "Bijan", "Robinson", "ATL", 11, 5, 3, (19.88, 13.84, 25.92)),
        _entry(97199, "Kenneth", "Walker III", "KC", 5, 1, 1, (20.26, 14.22, 26.3)),
        _entry(122474, "Jahmyr", "Gibbs", "DET", 6, 2, 2, None),  # on bye: no projection
    ]}
    rows = fetch_pff.build_ros_rows({"RB": payload})
    assert [r["Name"] for r in rows] == ["Kenneth Walker III", "Jahmyr Gibbs", "Bijan Robinson"]
    assert [r["Position Rank"] for r in rows] == [1, 2, 3]
    assert rows[2]["Rank"] == 5 and rows[2]["Next Week"] == 6 and rows[2]["Next Week Points"] == 19.88
    assert rows[1]["Next Week Points"] is None
    assert set(rows[0]) == set(fetch_pff.ROS_FIELDS)
