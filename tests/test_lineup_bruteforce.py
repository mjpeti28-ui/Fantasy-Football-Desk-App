"""Randomized check: greedy lineup solvers vs exhaustive assignment."""
from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import valuation as v  # noqa: E402
from backtest.start_sit import choose  # noqa: E402

SLOT_SETS = [
    ["QB", "RB", "RB/WR", "WR", "WR/TE", "TE", "DST", "K", "FLEX"],   # 2022+
    ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "DST", "K"],         # 2021
]


def brute(players, slots):
    """players: list of (pos, pts). Exhaustive best assignment (small rosters only)."""
    best = 0.0
    idx = list(range(len(players)))
    def rec(si, used, tot):
        nonlocal best
        if si == len(slots):
            best = max(best, tot)
            return
        elig = v.SLOT_ELIGIBILITY[slots[si]]
        rec(si + 1, used, tot)  # leave slot empty
        for i in idx:
            if i not in used and players[i][0] in elig:
                rec(si + 1, used | {i}, tot + players[i][1])
    rec(0, frozenset(), 0.0)
    return best


def test_greedy_matches_bruteforce():
    rng = random.Random(0)
    for trial in range(300):
        slots = SLOT_SETS[trial % 2]
        players = [(rng.choice(["QB", "RB", "RB", "WR", "WR", "WR", "TE", "TE", "K", "DST"]), round(rng.uniform(0, 30), 1))
                   for _ in range(rng.randint(8, 12))]
        pts = {}
        for pos, p in players:
            pts.setdefault(pos, []).append(p)
        for p in pts:
            pts[p].sort(reverse=True)
        exact = brute(players, slots)
        assert abs(v.best_lineup_points(pts, slots) - exact) < 1e-6, (players, slots)
        picks = choose([(i, pos, p) for i, (pos, p) in enumerate(players)], slots)
        assert abs(sum(players[i][1] for i in picks) - exact) < 1e-6
        assert len(set(picks)) == len(picks)
