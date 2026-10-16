"""FantasyPros API client: batching, per-day cache, quota ledger (no network)."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fantasypros_api as fpa  # noqa: E402


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


@pytest.fixture
def fake_http(monkeypatch):
    calls = []

    def fake_get(url, headers, params, timeout):
        calls.append(params)
        ids = [int(i) for i in params["players"].split(":")][:10]
        return _Resp({"ros_projections": True, "players": [
            {"fpid": i, "name": f"P{i}", "team_id": "JAC", "stats": {"points_ppr": float(i)}} for i in ids
        ]})

    monkeypatch.setattr(fpa.httpx, "get", fake_get)
    monkeypatch.setattr(fpa, "MIN_INTERVAL_S", 0.0)
    return calls


def _client(tmp_path):
    return fpa.FantasyProsClient(api_key="test", cache_dir=tmp_path, today=date(2026, 9, 22))


def test_batches_of_ten_and_same_day_cache(tmp_path, fake_http):
    client = _client(tmp_path)
    players = client.ros_projections(2026, "RB", range(1, 26))
    assert len(players) == 25
    assert len(fake_http) == 3  # 10 + 10 + 5
    assert client.calls_used_today() == 3
    client.ros_projections(2026, "RB", range(1, 26))
    assert len(fake_http) == 3  # served from cache, no new calls
    assert _client(tmp_path).calls_used_today() == 3  # ledger persists across instances


def test_shifted_player_set_only_fetches_new_players(tmp_path, fake_http):
    client = _client(tmp_path)
    client.ros_projections(2026, "WR", range(1, 21))  # 2 calls
    players = client.ros_projections(2026, "WR", range(3, 24))  # 3..20 cached; 21..23 new
    assert sorted(p["fpid"] for p in players) == list(range(3, 24))
    assert len(fake_http) == 3
    assert fake_http[-1]["players"] == "21:22:23"


def test_quota_guard(tmp_path, fake_http):
    client = _client(tmp_path)
    client._ledger_path.write_text('{"calls": %d}' % (fpa.DAILY_LIMIT - fpa.DAILY_SAFETY_MARGIN))
    with pytest.raises(fpa.QuotaExceeded):
        client.ros_projections(2026, "WR", [1, 2])
    assert fake_http == []


def test_ppr_points_accepts_object_or_array():
    assert fpa.ppr_points({"stats": {"points_ppr": "12.5"}}) == 12.5
    assert fpa.ppr_points({"stats": [{"points_ppr": 3}]}) == 3.0
    assert fpa.ppr_points({"stats": {}}) is None


def test_calls_needed():
    assert _client_calls({"QB": 30, "RB": 61, "K": 0}) == 3 + 7


def _client_calls(counts):
    return fpa.FantasyProsClient.calls_needed(None, counts)
