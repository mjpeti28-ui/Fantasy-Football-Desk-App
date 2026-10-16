from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest

# fastapi is only needed by the API tests; keep this conftest importable without
# it so non-API tests (data normalization, team/DST mapping) run anywhere.
try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - environment-dependent
    TestClient = None  # type: ignore[assignment,misc]

SCORABLE_GROUPS = {"QB", "RB", "WR", "TE"}


@pytest.fixture(scope="module")
def client() -> TestClient:
    if TestClient is None:
        pytest.skip("fastapi not installed")
    from api.main import app

    return TestClient(app)


@pytest.fixture(scope="module")
def league_teams(client: TestClient) -> List[str]:
    """Team slugs from whatever league is currently configured.

    Team names change between seasons (and mid-season, when managers rename),
    so tests resolve them at runtime rather than pinning last year's roster.
    """

    resp = client.get("/teams")
    assert resp.status_code == 200, resp.text
    teams = resp.json()
    assert isinstance(teams, list) and len(teams) >= 2, f"need at least two teams, got {teams!r}"
    return teams


@pytest.fixture(scope="module")
def trade_pair(client: TestClient, league_teams: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Two different teams, each with one scorable starter to offer in a trade."""

    found: List[Dict[str, Any]] = []
    for team in league_teams:
        resp = client.get(f"/teams/{team}")
        if resp.status_code != 200:
            continue
        for entry in resp.json().get("starters") or []:
            group = entry.get("position")
            name = entry.get("name")
            if group in SCORABLE_GROUPS and name:
                found.append({"team": team, "group": group, "name": name})
                break
        if len(found) == 2:
            break
    if len(found) < 2:
        pytest.skip("league data does not expose two teams with scorable starters")
    return found[0], found[1]
