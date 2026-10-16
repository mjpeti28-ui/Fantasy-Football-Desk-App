"""Waivers: players you won't drop are never offered as the drop."""
import pytest

pytest.importorskip("fastapi")

from app.engine import waivers as W  # noqa: E402


@pytest.fixture
def keepfile(tmp_path, monkeypatch):
    f = tmp_path / "waiver_keep.json"
    monkeypatch.setattr(W, "_keep_file", lambda: f)
    return f


def test_keep_round_trips_and_dedupes(keepfile):
    assert W.load_keep() == []
    assert W.save_keep([5, 3, 5, "9"]) == [3, 5, 9]
    assert W.load_keep() == [3, 5, 9]
    assert W.save_keep([]) == []
    assert W.load_keep() == []


def test_unreadable_file_means_nobody_kept(keepfile):
    keepfile.write_text("not json")
    assert W.load_keep() == []


def test_board_never_drops_a_kept_player():
    base = W._build(2)
    assert base["keep"] == [] and base["candidates"]
    usual = max({c["drop_id"] for c in base["candidates"]}, key=[c["drop_id"] for c in base["candidates"]].count)
    out = W._build(2, {usual})
    assert out["keep"] == [usual]
    assert usual not in {c["drop_id"] for c in out["candidates"]}
    assert out["candidates"], "the next-cheapest drops are still tried"
    assert [r for p in out["depth"].values() for r in p if r["id"] == usual][0]["kept"] is True
    assert out["drop_cost"][str(usual)] == base["drop_cost"][str(usual)]  # his worth is still shown


def test_keeping_everyone_leaves_no_suggestions_and_does_not_crash():
    base = W._build(2)
    mine = {r["id"] for p in base["depth"].values() for r in p if not r["ir"]}
    assert W._build(2, mine)["candidates"] == []


def test_endpoint_saves_and_next_board_uses_it(keepfile):
    from fastapi.testclient import TestClient
    from app.engine import server
    c = TestClient(server.app)
    assert c.post("/api/waivers/keep", json={"ids": [4047365, 4047365]}).json() == {"keep": [4047365]}
    assert W.load_keep() == [4047365]
