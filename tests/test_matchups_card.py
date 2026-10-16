"""Matchups card: the defense rating, the grades, and the card's bottom line. The card must never adjust ESPN's
projection, must say how much of it is the matchup, and must tag every row. Needs the cached nflverse tables."""
from __future__ import annotations

import json
import math

import pandas as pd
import pytest

pytest.importorskip("pandas")
from app.engine import matchups as m, matchups_card as mc, matchups_data as md, matchups_rating as mr  # noqa: E402

needs_tables = pytest.mark.skipif(
    not all((md.cache_dir() / f"{y}.csv.gz").exists() for y in (*md.HISTORY, m.current_season())),
    reason="matchup tables not built",
)


def _tbl(**kw):
    return pd.DataFrame({"epa": pd.Series({k: v[0] for k, v in kw.items()}), "n": pd.Series({k: v[1] for k, v in kw.items()})})


def test_rate_formula_and_direction():
    prior = _tbl(A=(100.0, 1000), B=(-100.0, 1000))           # A allowed +0.1 EPA/play last year, B -0.1
    cur = _tbl(A=(0.0, 250), B=(0.0, 250))                    # this season so far both exactly at the league mean (0)
    r = mr.rate(prior, cur)
    # prior_i = lgp + rho * (pv_i - lgp), lgp = 0; A: 0.35*0.1 = 0.035 ; weight 250/500 on the prior
    assert math.isclose(r["A"], 0.5 * 0.035, abs_tol=1e-9)
    assert math.isclose(r["B"], -0.5 * 0.035, abs_tol=1e-9)
    assert r["A"] > r["B"]                                      # softer defense has the higher rating


def test_rate_edge_cases():
    prior = _tbl(A=(50.0, 500), B=(-50.0, 500))
    empty = _tbl()
    empty = empty.iloc[0:0]
    r = mr.rate(prior, empty)                                  # week 1: only the prior
    assert math.isclose(r["A"], 0.35 * 0.1, abs_tol=1e-9) and math.isclose(r["B"], -0.35 * 0.1, abs_tol=1e-9)
    r2 = mr.rate(empty, _tbl(A=(10.0, 100), B=(-10.0, 100)))   # first season: no prior at all
    assert r2["A"] > 0 > r2["B"]


def test_credit_file_is_consistent():
    c = mc.credit_table()
    assert c, "run backtest/matchup_mechanisms.py to write matchups_credit.json"
    for pos, spec in mc.SPEC.items():
        for k in spec:
            x = c["credit"][pos][k]
            assert math.isclose(x["per_unit"] * c["sd_ref"][mc.KIND[k]], x["per_sd"], rel_tol=1e-6)
            assert x["ci_per_sd"][0] <= x["per_sd"] <= x["ci_per_sd"][1]
            lo = c["left_over"][pos][k]
            assert lo["ci_per_sd"][0] <= lo["per_sd"] <= lo["ci_per_sd"][1]
    # the finding the card is built on: after ESPN's number, no position keeps a usable defense effect
    for pos, spec in mc.SPEC.items():
        for k in spec:
            lo, hi = c["left_over"][pos][k]["ci_per_sd"]
            assert lo <= 0.0 <= hi or abs(lo) < 1.0, (pos, k)


def test_credit_file_has_the_share_of_projection_fit():
    c = mc.credit_table()
    for pos, spec in mc.SPEC.items():
        assert c["credit"][pos]["mean_proj"] > 5
        for k in spec:
            x = c["credit"][pos][k]
            assert math.isclose(x["pct_per_unit"] * c["sd_ref"][mc.KIND[k]], x["pct_per_sd"], rel_tol=1e-6)
            assert x["ci_pct_per_sd"][0] <= x["pct_per_sd"] <= x["ci_pct_per_sd"][1]
            # the share model and the flat model agree at a typical projection for the position
            flat = x["per_sd"]
            at_typical = mc.scaled(c, pos, k, c["credit"][pos]["mean_proj"])["per_sd"]
            assert abs(at_typical - flat) < 0.35 * max(abs(flat), 0.3), (pos, k, flat, at_typical)


def test_scaled_credit_grows_with_the_projection():
    c = mc.credit_table()
    small, big = mc.scaled(c, "RB", "pass_r", 6.0), mc.scaled(c, "RB", "pass_r", 18.0)
    assert math.isclose(big["per_unit"] / small["per_unit"], 3.0, rel_tol=1e-9)
    assert small["ci"][0] <= small["per_sd"] <= small["ci"][1]
    fallback = mc.scaled(c, "RB", "pass_r", None)                      # no projection: the position's typical one, never a crash
    assert fallback["base"] == c["credit"]["RB"]["mean_proj"]
    old = {"credit": {"RB": {"pass_r": {"per_unit": 6.0, "per_sd": 0.4, "ci_per_sd": [0.2, 0.6]}}}}
    assert mc.scaled(old, "RB", "pass_r", 15.0)["per_unit"] == 6.0       # an older credit file still works


@needs_tables
def test_grades_cover_the_league():
    g = mc.grades()
    assert len(g) == 32
    for kind in ("pass", "run"):
        ranks = sorted(v[kind]["rank"] for v in g.values())
        assert ranks == list(range(1, 33))
        labels = [v[kind]["label"] for v in g.values()]
        assert labels.count("Tough") == 11 and labels.count("Soft") == 11
        assert abs(sum(v[kind]["r"] for v in g.values())) < 1e-6          # centred on the league
        assert all(0.0 <= v[kind]["w_now"] < 1.0 for v in g.values())
    json.dumps(g)


def _snap(espn_id: int, proj: float, injury: str = "ACTIVE", bye: bool = False):
    return {"season": m.current_season(), "week": 5,
            "team_week": {"players": [{"id": espn_id, "espn_week": proj, "est_ppg": proj - 1.0, "injury": injury, "game": {"bye": bye}}], "matchup": {"opp_starters": []}}}


def _espn(name: str) -> int:
    ids = m.tables()["ids"]
    return int(ids.loc[ids["name"] == name, "espn_id"].dropna().iloc[0])


def _card(monkeypatch, name: str, opp: str, proj: float, sched: str | None = None, **kw):
    """Build a card as if `opp` were on his schedule this week (or `sched` is, when looking through)."""
    eid = _espn(name)
    pv = m.player_view(eid, opp)
    real = sched if sched is not None else opp
    monkeypatch.setattr(m, "schedule", lambda season, week: {pv["team"]: {"opp": real, "home": True, "date": "", "time": None}} if real else {})
    return eid, pv, mc.build(pv, _snap(eid, proj, **kw))


@needs_tables
def test_card_for_a_receiver_never_changes_the_projection(monkeypatch):
    eid, pv, c = _card(monkeypatch, "CeeDee Lamb", "PHI", 17.0)
    assert c["bottom"]["tone"] == "go" and "17.0" in c["bottom"]["headline"] and not c["through"]
    assert c["proj"]["espn"] == 17.0                                   # untouched
    keys = [r["key"] for r in c["rows"]]
    assert "pass" in keys and "lens" in keys
    assert all(r["tag"] in {"priced", "small", "context", "role"} for r in c["rows"])
    r = {r["key"]: r for r in c["rows"]}
    assert r["lens"]["tag"] == "small"                                  # the coverage lens is never presented as an input
    cj = mc.credit_table()
    g = mc.grades()["PHI"]["pass"]
    assert math.isclose(c["priced"]["pts"], mc.scaled(cj, "WR", "pass_r", 17.0)["per_unit"] * g["r"], rel_tol=1e-9)
    assert abs(c["priced"]["pts"]) < 1.0                                 # a receiver's matchup is worth well under a point


@needs_tables
def test_card_for_a_quarterback_has_pass_rush_and_no_run_row(monkeypatch):
    _, _, c = _card(monkeypatch, "Josh Allen", "LA", 20.0)
    keys = [r["key"] for r in c["rows"]]
    assert "pass" in keys and "run" not in keys and "lens" not in keys
    assert c["bottom"]["tone"] == "go"


@needs_tables
def test_out_and_bye_players_get_no_matchup_read(monkeypatch):
    _, pv, out = _card(monkeypatch, "CeeDee Lamb", "PHI", 0.0, injury="OUT")
    assert out["dead"] and out["bottom"]["tone"] == "out"
    eid = _espn("CeeDee Lamb")
    pv = dict(pv); pv["opp"] = None
    assert mc.build(pv, _snap(eid, 17.0, bye=True))["bottom"]["tone"] == "bye"


@needs_tables
def test_looking_through_another_defense_is_marked_and_not_a_projection(monkeypatch):
    _, _, c = _card(monkeypatch, "CeeDee Lamb", "PHI", 17.0, sched="WAS")
    assert c["through"] and c["sched"] == "WAS" and c["opp"] == "PHI"
    assert c["bottom"]["tone"] == "whatif" and "PHI" in c["bottom"]["headline"]
    assert c["proj"]["espn"] == 17.0                                    # still his real game's number, untouched
    assert c["priced"] and any("WAS" in l for l in c["bottom"]["lines"])
    # a bye-week player looked through a defense still gets the what-if card, and nothing says "bye"
    _, _, b = _card(monkeypatch, "CeeDee Lamb", "PHI", 17.0, sched="", bye=True)
    assert b["through"] and b["bottom"]["tone"] == "whatif" and any("no game this week" in l for l in b["bottom"]["lines"])


@needs_tables
def test_worth_scales_with_the_players_projection(monkeypatch):
    _, _, lo = _card(monkeypatch, "Josh Allen", "LA", 10.0)
    _, _, hi = _card(monkeypatch, "Josh Allen", "LA", 20.0)
    assert lo["priced"]["pts"] != 0 and math.isclose(hi["priced"]["pts"] / lo["priced"]["pts"], 2.0, rel_tol=1e-9)
    assert lo["proj"]["espn"] == 10.0 and hi["proj"]["espn"] == 20.0          # the number itself is never touched
    assert hi["priced"]["parts"][0]["pct"] is not None


@needs_tables
def test_early_down_runner_gets_a_context_row_and_nothing_else_changes(monkeypatch):
    base_style = {"tier": "Lead back", "style": "All-purpose back", "trend": None, "trend_text": None}
    monkeypatch.setattr(mc, "_role", lambda eid: dict(base_style))
    _, _, plain = _card(monkeypatch, "James Cook", "LA", 15.0)
    assert "style_run" not in [r["key"] for r in plain["rows"]]
    monkeypatch.setattr(mc, "_role", lambda eid: dict(base_style, style="Early-down runner"))
    _, _, ed = _card(monkeypatch, "James Cook", "LA", 15.0)
    rows = {r["key"]: r for r in ed["rows"]}
    assert rows["style_run"]["tag"] == "context" and rows["style_run"]["effect"] is None
    assert "ESPN prices" in rows["style_run"]["tip"] and "2016" in rows["style_run"]["tip"]
    assert ed["priced"]["pts"] == plain["priced"]["pts"] and ed["proj"]["espn"] == 15.0   # explains the number, never edits it
    assert ed["bottom"]["headline"] == plain["bottom"]["headline"]
    # a receiver with the same label gets no such row (the finding is about running backs)
    monkeypatch.setattr(mc, "_role", lambda eid: dict(base_style, style="Early-down runner"))
    _, _, wr = _card(monkeypatch, "CeeDee Lamb", "PHI", 17.0)
    assert "style_run" not in [r["key"] for r in wr["rows"]]
