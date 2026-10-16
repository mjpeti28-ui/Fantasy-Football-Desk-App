"""Season analytics (app/engine/analytics.py) and the backtest registry it reads."""
import json

import numpy as np
import pandas as pd
import pytest

from app.engine import analytics as an


def test_tiers_scale_with_league_size():
    t = an.tiers(10)
    assert t["start"] == {"QB": 10, "TE": 10, "RB": 25, "WR": 25}
    assert t["smash"]["QB"] == 3 and t["smash"]["RB"] == 6
    assert t["dud"]["WR"] == 50
    assert an.tiers(12)["start"]["RB"] == 30


def test_summarize_rates_and_volatility():
    G = pd.DataFrame({"espn_id": [1, 1, 1, 1], "points": [10.0, 20.0, 30.0, 0.0], "xfp": [12.0] * 4, "fpoe": [-2.0, 8, 18, -12],
                      "opp": [8] * 4, "offense_pct": [.8] * 4, "target_share": [.2] * 4, "air_yards_share": [.3] * 4, "wopr": [.5] * 4,
                      "csh": [0.0] * 4, "smash": [0, 1, 1, 0.0], "start": [1, 1, 1, 0.0], "dud": [0, 0, 0, 1.0],
                      "resid": [0, 5, 10, -10.0], "beat": [0, 1, 1, 0.0]})
    s = an.summarize(G).loc[1]
    assert s["n"] == 4 and s["ppg"] == 15 and s["smash"] == 0.5 and s["dud"] == 0.25
    assert s["cv"] == pytest.approx(G["points"].std() / 15)


def test_registry_labels_are_known_and_complete():
    R = an.registry()
    assert R, "run backtest/analytics_metrics.py run"
    known = set(R["labels"])
    for k, m in R["players"].items():
        for pos, t in m["by_pos"].items():
            assert t["label"] in known, (k, pos)
    # nothing may claim more than its pooled stat (the position cells weren't tested out of sample)
    order = {"noise": 0, "weak": 0, "trait": 0, "priced": 0, "tilt": 1, "edge": 2, "adds": 3}
    for k, m in R["players"].items():
        top = m["by_pos"]["ALL"]["label"]
        for pos, t in m["by_pos"].items():
            if t["label"] in ("edge", "adds"):
                assert order[t["label"]] <= order[top], (k, pos)
    for k in an.READ:
        assert k in R["players"], k


def test_metrics_readings_fill_in():
    M = an.metrics()
    assert "{" not in "".join(m["reading"] for m in M.values())
    assert M["fpoe"]["sticks"] is not None and 0 <= M["fpoe"]["sticks"] < 0.5


@pytest.fixture(scope="module")
def players():
    return an.players(2026)


def test_players_payload(players):
    assert players["through_week"] >= 1
    rows = players["players"]
    assert rows and all(r["pos"] in an.SKILL for r in rows)
    r = max(rows, key=lambda r: r["games"])
    assert len(r["weekly"]) == players["through_week"]
    played = [w for w in r["weekly"] if w]
    assert len(played) == r["games"]
    assert all(w[7] in ("smash", "start", "mid", "dud") for w in played)
    assert r["starts"] <= r["games"] and r["duds"] <= r["games"]
    for r in rows:
        for v in r["pct"].values():
            assert v is None or 0 <= v <= 100


def test_flags_follow_their_thresholds(players):
    for r in players["players"]:
        for f in r["flags"]:
            if f["key"] in ("hot", "cold"):
                assert abs(r["fpoe"]) >= an.HOT and f["label"] == "noise"
                assert (f["key"] == "hot") == (r["fpoe"] > 0)
            if f["key"].startswith("snaps"):
                assert abs(r["snap_tr"]) >= an.SNAP_T and r["games"] >= 4
            if f["key"].startswith("value"):
                assert abs(r["est_tr"]) >= an.EST_T


def test_partial_weeks_stay_out(monkeypatch, tmp_path):
    s = pd.DataFrame({"season": 2026, "game_type": "REG", "week": [1, 1, 2, 2, 3], "home_score": [20, 17, 3, np.nan, np.nan]})
    f = tmp_path / "schedules.csv"
    s.to_csv(f, index=False)
    monkeypatch.setattr(an, "NFL", tmp_path)
    assert an.complete_week(2026, fallback=9) == 1


def test_league_table(players):
    L = an.league(2026)
    assert L["teams"] and L["cells"]
    for c in L["cells"]:
        assert c["opt"] + 1e-6 >= c["pts"], c      # the best lineup can't score less than the one used
        assert 1 <= c["rank"] <= L["n_teams"]
    tot = {t["id"]: t["pts"] for t in L["teams"]}
    for t, v in tot.items():
        assert v == pytest.approx(sum(c["pts"] for c in L["cells"] if c["team_id"] == t), abs=0.05)
    assert all(p["pos_pick"] >= 1 for p in L["draft"])


def test_endpoints():
    from fastapi.testclient import TestClient
    from app.engine import server
    c = TestClient(server.app)
    r = c.get("/api/analytics")
    assert r.status_code == 200 and json.loads(r.content)["players"]
    r = c.get("/api/analytics/league")
    assert r.status_code == 200 and json.loads(r.content)["teams"]


def test_schedule_ahead(players):
    weeks = players["sched_weeks"]
    assert weeks and weeks[0] == players["through_week"] + 1 and weeks[-1] == an.LAST
    S = players["schedule"]
    for team, games in S.items():
        for w, g in games.items():
            assert int(w) in weeks
            back = S[g["opp"]][w]                      # every game appears from both sides
            assert back["opp"] == team and back["home"] != g["home"]
    ev = players["sched_ev"]
    assert ev["label"] in an.registry()["labels"]
