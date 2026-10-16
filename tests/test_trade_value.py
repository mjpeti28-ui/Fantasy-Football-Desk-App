from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import trade_value as tv  # noqa: E402
import valuation as v  # noqa: E402


def _ctx(monkeypatch, players, ir=()):
    tbl = pd.DataFrame({"position": {i: p for i, (p, _) in players.items()}})
    exp = pd.Series({i: e for i, (_, e) in players.items()}, dtype=float)
    monkeypatch.setattr(v, "expected_ros_points", lambda t, *a, **k: exp.loc[t.index])
    return tv.Context(None, 3, None, tbl, {}, {}, {}, frozenset(ir))


def test_after_drops_backup_kicker_before_skill_players(monkeypatch):
    ctx = _ctx(monkeypatch, {1: ("K", 117), 2: ("K", 113), 3: ("WR", 90), 4: ("RB", 150), 5: ("RB", 200)})
    assert sorted(tv._after(ctx, [1, 2, 3, 4], [4], [5])) == [1, 2, 3, 5]  # 1-for-1: no drop
    ctx = _ctx(monkeypatch, {1: ("K", 117), 2: ("K", 113), 3: ("WR", 90), 4: ("RB", 150), 5: ("RB", 200), 6: ("WR", 60)})
    after = tv._after(ctx, [1, 2, 3, 4], [4], [5, 6])  # gets two, gives one: one must go
    assert 2 not in after and 1 in after and {3, 5, 6} <= set(after)


def test_after_never_drops_or_counts_ir_slot_players(monkeypatch):
    ctx = _ctx(monkeypatch, {1: ("WR", 10), 2: ("WR", 100), 3: ("RB", 120), 4: ("RB", 200), 5: ("RB", 180), 6: ("K", 100)}, ir=[1])
    after = tv._after(ctx, [1, 2, 3, 6], [3], [4, 5])
    assert 1 in after and 6 in after and 2 not in after  # lowest non-IR, non-only-kicker goes


def test_after_keeps_only_kicker(monkeypatch):
    ctx = _ctx(monkeypatch, {1: ("K", 5), 2: ("WR", 100), 3: ("RB", 120), 4: ("RB", 200), 5: ("RB", 180)})
    after = tv._after(ctx, [1, 2, 3], [3], [4, 5])
    assert 1 in after and 2 not in after
