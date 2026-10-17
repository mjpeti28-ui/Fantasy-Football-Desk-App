"""PFF alignment for WR/TE role styles: optional data, clean fallback, nothing per-player in the shipped model."""
import json
import urllib.error

import numpy as np
import pandas as pd

from app.engine import roles as R
from app.engine import roles_pff as rp

MODEL = R.load_model()
WR_ROW = {"adot": 12.0, "behind_r": 0.07, "deep_r": 0.2}


def test_wr_te_styles_use_pff_with_free_fallback():
    for pos in ("WR", "TE"):
        assert any(c in R.PFF_STYLE[pos] for c in MODEL["style"][pos]["features"])
        assert not any(c in R.PFF_STYLE[pos] for c in MODEL["style_free"][pos]["features"])
    for pos in ("RB", "QB"):
        assert pos not in MODEL.get("style_free", {})


def test_classify_falls_back_without_pff():
    st = R.style_of(MODEL, "WR", WR_ROW, None, 6)
    assert st["pff"] is False and st["name"] in MODEL["style_free"]["WR"]["names"]
    st = R.style_of(MODEL, "WR", {**WR_ROW, "slot_rate": 60.0}, None, 6)
    assert st["pff"] is True and st["name"] == "Slot receiver"
    st = R.style_of(MODEL, "WR", {**WR_ROW, "slot_rate": 15.0}, None, 6)
    assert st["name"] == "Outside receiver"
    assert R.style_of(MODEL, "WR", {**WR_ROW, "slot_rate": np.nan}, None, 6)["pff"] is False
    te = {"tgt_per_snap": 0.06, "adot": 6.0, "behind_r": 0.2, "inline_rate": 70.0, "slot_rate": 15.0, "route_rate": 75.0}
    assert R.style_of(MODEL, "TE", te, None, 6)["name"] == "Inline TE"


def test_model_stores_no_player_rows():
    txt = (R.HERE / "roles_model.json").read_text()
    assert "00-00" not in txt and "player_id" not in txt
    for pos in ("WR", "TE"):
        m = MODEL["style"][pos]
        assert set(m) <= {"features", "mean", "scale", "weights", "means", "covs", "names", "n", "n_fit", "seasons", "source",
                          "centroids", "yoy_same", "kappa", "transitions", "meaning", "early_acc", "check_free_same_rows"}


def test_cli_without_key_or_network_exits_cleanly(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rp, "CACHE", tmp_path)
    monkeypatch.setattr(rp, "ROOT", tmp_path)  # no .env.local here
    monkeypatch.delenv("PFF_API_KEY", raising=False)
    (tmp_path / "app" / "cache.nosync" / "roles" / "2026").mkdir(parents=True)
    pd.DataFrame({"week": [1, 2, 3]}).to_csv(tmp_path / "app" / "cache.nosync" / "roles" / "2026" / "roles_weekly.csv", index=False)
    assert rp.main(["2026"]) == 0
    assert "not set" in capsys.readouterr().out
    monkeypatch.setenv("PFF_API_KEY", "test-not-a-key")

    def down(*a, **k):
        raise urllib.error.URLError("offline")
    monkeypatch.setattr(rp.urllib.request, "urlopen", down)
    assert rp.main(["2026"]) == 0
    out = capsys.readouterr().out
    assert "unavailable" in out and "test-not-a-key" not in out
    assert not list(tmp_path.rglob("*.json"))
    assert rp.season_features(2026, pd.DataFrame({"gsis": ["x"], "pos": ["WR"]})).empty


def test_season_features_and_match(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "CACHE", tmp_path)
    rows = [{"player_id": 1, "player": "Amon-Ra St. Brown", "position": "WR", "team": "DET", "player_game_count": 5, "slot_snaps": 120,
             "inline_snaps": 0, "wide_snaps": 60, "routes": 170, "pass_plays": 180, "targets": 50},
            {"player_id": 2, "player": "Mike Williams", "position": "WR", "team": "NYJ", "player_game_count": 5, "slot_snaps": 10,
             "inline_snaps": 0, "wide_snaps": 150, "routes": 150, "pass_plays": 160, "targets": 30}]
    p = rp.window_path(2026, 1, 5)
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"receiving_summary": rows}))
    weekly = pd.DataFrame({"gsis": ["g1", "g2", "g3"], "pos": ["WR"] * 3, "pname": ["Amon-Ra St. Brown", "Mike Williams", "Mike Williams"]})
    f = rp.season_features(2026, weekly)
    assert list(f.index) == ["g1"]  # two Mike Williams at WR: dropped, not guessed
    assert abs(f.loc["g1", "slot_rate"] - 100 * 120 / 180) < 1e-9 and f.loc["g1", "pff_through"] == 5


def test_role_map_uses_free_axes_for_unmatched_player():
    from app.engine import dossier
    cols = {"pos": "WR", "games": 6.0, "tgt_sh": 0.2, "adot": 11.0, "behind_r": 0.08, "deep_r": 0.15}
    cur = pd.DataFrame([{**cols, "slot_rate": 55.0}, {**cols, "slot_rate": np.nan}], index=["matched", "unmatched"])
    for me, axes in (("matched", MODEL["axes_pff"]["WR"]), ("unmatched", MODEL["axes"]["WR"])):
        m = dossier._map(MODEL, "WR", cur, pd.DataFrame(), None, me, {}, {})
        assert [m["x"], m["y"]] == list(axes)
        assert [g["name"] for g in m["groups"]] == (MODEL["style"] if me == "matched" else MODEL["style_free"])["WR"]["names"]
