"""What our rest-of-season number is made of, for the player file.

Two views, both descriptive:
- the blend: ESPN's half (smoothed weekly projection), our own average's half, the FantasyPros
  pull and (WRs) the PFF targets-per-route nudge, week by week from the ledger snapshots (they add up to est_ppg; snapshots written before the
  parts were stored show the total only);
- the box-score build: fantasy points per game = team plays x his share x expected points per
  opportunity x finishing (points / expected), season vs his last 3 games, from nflverse. Share settles
  in about a game; finishing is mostly luck within a season (metrics/own-projection), so the factor
  that moved tells you whether a hot or cold stretch is role or luck.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import data as d


def _f(x, nd: int = 2) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if v != v else round(v, nd)


@lru_cache(maxsize=64)
def _snap(path: str, mtime: float) -> pd.DataFrame:
    cols = ["espn_id", "est_ppg", "c_espn", "c_ours", "c_fp", "c_pff", "tprr", "espn_smooth", "ours_ppg", "n", "ecr_rank"]
    x = pd.read_csv(path)
    return x[[c for c in cols if c in x.columns]].set_index("espn_id")


def history(season: int, espn_id: int) -> List[dict]:
    """One row per ledger week: est and, where stored, its three parts and their inputs."""
    out = []
    for p in sorted((d.LEDGER / str(season)).glob("wk*_players.csv")):
        m = re.match(r"wk(\d+)_players", p.stem)
        if not m:
            continue
        x = _snap(str(p), p.stat().st_mtime)
        if espn_id not in x.index:
            continue
        r = x.loc[espn_id]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        out.append({"week": int(m.group(1)), "est": _f(r.get("est_ppg")), "pe": _f(r.get("c_espn")), "po": _f(r.get("c_ours")),
                    "pf": _f(r.get("c_fp")), "pp": _f(r.get("c_pff")), "tprr": _f(r.get("tprr"), 3), "ew": _f(r.get("espn_smooth"), 1), "ou": _f(r.get("ours_ppg"), 1),
                    "n": _f(r.get("n"), 0), "r": _f(r.get("ecr_rank"), 0)})
    return sorted(out, key=lambda r: r["week"])


@lru_cache(maxsize=4)
def _opps(season: int, mtime: float) -> pd.DataFrame:
    o = pd.read_csv(d.NFLVERSE / str(season) / "opportunity.csv")
    ids = pd.read_csv(d.NFLVERSE / "ids.csv").dropna(subset=["gsis_id", "espn_id"]).drop_duplicates("gsis_id")
    o = o[o["position"].isin(["QB", "RB", "WR", "TE"])].copy()
    o["opps"] = (o["pass_attempt"].where(o["position"] == "QB", o["rec_attempt"]) + o["rush_attempt"]).fillna(0)
    t = o.groupby(["posteam", "week"])[["pass_attempt", "rush_attempt"]].sum().sum(axis=1).rename("plays")
    o = o.join(t, on=["posteam", "week"])
    o["espn_id"] = o["player_id"].map(dict(zip(ids["gsis_id"], ids["espn_id"].astype(int))))
    return o.dropna(subset=["espn_id"])


def _window(g: pd.DataFrame) -> Optional[dict]:
    plays, opps, xfp, fp = (float(g[c].sum()) for c in ("plays", "opps", "total_fantasy_points_exp", "total_fantasy_points"))
    n = len(g)
    if not n or plays <= 0 or opps <= 0 or xfp <= 0:
        return None
    return {"games": n, "plays": round(plays / n, 1), "share": round(opps / plays, 3), "xpo": round(xfp / opps, 3),
            "fin": round(fp / xfp, 3), "ppg": round(fp / n, 1), "xppg": round(xfp / n, 1)}


def build_up(season: int, espn_id: int, last: int = 3) -> Optional[dict]:
    """Season vs last-``last``-games build of points per game, and which factor moved most (log share)."""
    p = d.NFLVERSE / str(season) / "opportunity.csv"
    try:
        o = _opps(season, p.stat().st_mtime)
    except (OSError, ValueError, KeyError):
        return None
    g = o[o["espn_id"] == espn_id].sort_values("week")
    g = g[g["plays"] > 0]
    s, r = _window(g), _window(g.tail(last))
    if s is None or r is None or len(g) < last + 1:
        return {"season": s, "recent": None, "moved": None} if s else None
    moves = {k: float(np.log(r[k] / s[k])) for k in ("plays", "share", "xpo", "fin") if s[k] > 0 and r[k] > 0}
    total = sum(moves.values())
    moved = max(moves, key=lambda k: abs(moves[k])) if moves else None
    return {"season": s, "recent": r, "moved": moved, "log": {k: round(v, 3) for k, v in moves.items()}, "log_total": round(total, 3)}


def build(season: int, espn_id: int) -> dict:
    return {"history": history(season, espn_id), "build": build_up(season, espn_id)}
