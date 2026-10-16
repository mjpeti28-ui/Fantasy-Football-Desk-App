"""Availability model: P(plays) for the next game and each later game.

Logistic regression on what's knowable by midweek (when waivers run): status bucket,
consecutive games missed, share of games played so far, rostered or not, last week's
official injury report, depth-chart rank, and whether ESPN projects him at 0 this week.
Later games decay with horizon h (games ahead): p(h) = sigmoid(a + b*log(1+h)).

Backtest (backtest/availability.py, leave-one-season-out 2021-2025, players with a
forecast >= 3 ppg): Brier .1539 -> .141, rest-of-season points rank corr .678 -> .690,
points MAE 30.3 -> 29.0, waiver replay 80.0 -> 82.5 pts/pickup.

The biggest single piece of information: when ESPN projects a 0, a player who is on the
weekly injury report (Out/Doubtful/Questionable) is a short-term absence (~.65-.70 of later
games played), while one who isn't on it (IR, suspended, released) mostly doesn't come
back (~.26 free agents, ~.42 rostered).

Needs the nflverse cache (history/nflverse/<season>/injuries.csv, depth.csv; ids.csv).
Without it, apply() leaves the v2 bucket availability untouched.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
NFLV = ROOT / "history" / "nflverse"
SKILL = ("QB", "RB", "WR", "TE")
FEATURES = ["st_missed_last", "st_ir", "streak1", "streak2_3", "streak4p", "frac_played", "no_games_yet",
            "rostered", "prev_out", "prev_q", "depth1", "depth2", "depth3p", "off_chart", "espn0"]
POS = ["pos_QB", "pos_RB", "pos_TE"]


@lru_cache(maxsize=16)
def _season_cache(season: int):
    d = NFLV / str(season)
    if not (d / "injuries.csv").exists() or not (NFLV / "ids.csv").exists():
        return None
    ids = pd.read_csv(NFLV / "ids.csv").dropna(subset=["gsis_id", "espn_id"]).drop_duplicates("gsis_id")
    inj = pd.read_csv(d / "injuries.csv")
    inj["espn_id"] = inj["gsis_id"].map(ids.set_index("gsis_id")["espn_id"])
    inj = inj.dropna(subset=["espn_id"]).drop_duplicates(["espn_id", "week"], keep="last")
    inj_status = {(int(a), int(b)): s for a, b, s in zip(inj["espn_id"], inj["week"], inj["report_status"])}
    dmin = {}
    if (d / "depth.csv").exists():
        dp = pd.read_csv(d / "depth.csv").dropna(subset=["espn_id"])
        dmin = {(int(a), int(b)): v for (a, b), v in dp.groupby(["espn_id", "week"])["depth"].min().items()}
    return inj_status, dmin


def raw_features(season: int, week: int, players: pd.DataFrame, played: set,
                 rostered: Optional[Iterable[int]]) -> Optional[pd.DataFrame]:
    """players: columns espn_id, position, status, espn_now, bye. ``played``: set of
    (espn_id, week) with a game played. Returns FEATURES + POS per player, or None
    when the nflverse cache for the season is missing."""
    cache = _season_cache(season)
    if cache is None:
        return None
    inj_status, dmin = cache
    rset = None if rostered is None else {int(i) for i in rostered}
    out = []
    for r in players.itertuples():
        pid, pos = int(r.espn_id), r.position
        bye = int(r.bye) if pd.notna(r.bye) else -1
        past = [w for w in range(1, week) if w != bye]
        streak = 0
        for w in reversed(past):
            if (pid, w) in played:
                break
            streak += 1
        frac = (sum((pid, w) in played for w in past) / len(past)) if past else 1.0
        pw = week - 1 if week - 1 != bye else week - 2
        prev = inj_status.get((pid, pw))
        dep = dmin.get((pid, week), dmin.get((pid, week - 1)))
        out.append({
            "espn_id": pid,
            "st_missed_last": float(r.status == "missed_last"), "st_ir": float(r.status == "ir"),
            "streak1": float(streak == 1), "streak2_3": float(2 <= streak <= 3), "streak4p": float(streak >= 4),
            "frac_played": frac, "no_games_yet": float(bool(past) and frac == 0),
            "rostered": float(rset is None or pid in rset),
            "prev_out": float(prev in ("Out", "Doubtful")), "prev_q": float(prev == "Questionable"),
            "depth1": float(dep == 1), "depth2": float(dep == 2),
            "depth3p": float(dep is not None and not pd.isna(dep) and dep >= 3), "off_chart": float(dep is None),
            "espn0": float(r.espn_now == 0 and bye != week),
            "pos_QB": float(pos == "QB"), "pos_RB": float(pos == "RB"), "pos_TE": float(pos == "TE"),
        })
    return pd.DataFrame(out)


def design(Fe: pd.DataFrame, h: np.ndarray) -> np.ndarray:
    base = Fe[FEATURES + POS].to_numpy(float)
    first = (h == 0).astype(float)[:, None]
    logh = np.log1p(h)[:, None]
    return np.column_stack([base * first, base * (1 - first), first, logh, Fe[FEATURES].to_numpy(float) * logh])


def fit(Fe: pd.DataFrame, h: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 40) -> Dict:
    X = np.column_stack([np.ones(len(y)), design(Fe, h)])
    b = np.zeros(X.shape[1])
    reg = l2 * np.r_[0, np.ones(len(b) - 1)]
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        g = X.T @ (p - y) + reg * b
        H = (X * (p * (1 - p))[:, None]).T @ X + np.diag(reg)
        step = np.linalg.solve(H, g)
        b -= step
        if np.abs(step).max() < 1e-7:
            break
    return {"features": FEATURES, "pos": POS, "coef": [round(float(x), 6) for x in b]}


def curve_params(model: Dict, Fe: pd.DataFrame) -> pd.DataFrame:
    """Per player: p_next, and (a, b) for later games p(h) = sigmoid(a + b*log1p(h))."""
    c = np.asarray(model["coef"])
    nb = len(FEATURES) + len(POS)
    c0, bn, bl, bf, bh, bx = c[0], c[1:1 + nb], c[1 + nb:1 + 2 * nb], c[1 + 2 * nb], c[2 + 2 * nb], c[3 + 2 * nb:]
    base = Fe[FEATURES + POS].to_numpy(float)
    lin_next = c0 + base @ bn + bf
    a = c0 + base @ bl
    b = bh + Fe[FEATURES].to_numpy(float) @ bx
    return pd.DataFrame({"espn_id": Fe["espn_id"].to_numpy(), "p_next": 1 / (1 + np.exp(-lin_next)), "av_a": a, "av_b": b})


def later_prob(a, b, h):
    return 1 / (1 + np.exp(-(a + b * np.log1p(h))))


def fit_from_backtest_cache(seasons, relevant_min: float = 3.0) -> Dict:
    """Fit on backtest/availability.py's cached features (history/backtest/multi)."""
    out = ROOT / "history" / "backtest" / "multi"
    F = pd.read_csv(out / "avail_features.csv")
    G = pd.read_csv(out / "avail_games.csv")
    D = G.merge(F, on=["season", "week", "espn_id"])
    D = D[D["season"].isin(list(seasons)) & (D["est_ppg"] >= relevant_min) & D["position"].isin(SKILL)]
    for p in ("QB", "RB", "TE"):
        D["pos_" + p] = (D["position"] == p).astype(float)
    return fit(D, D["h"].to_numpy(float), D["played"].to_numpy(float))


def save(model: Dict, path: Path = ROOT / "availability_model.json") -> None:
    path.write_text(json.dumps(model, indent=1))
