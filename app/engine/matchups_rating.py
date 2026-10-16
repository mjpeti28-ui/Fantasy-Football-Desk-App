"""Defense ratings from every play: trailing EPA per dropback (or per rush) allowed, shrunk toward a prior built
from last season. Higher = softer defense.

Shared by the Matchups card and backtest/matchup_mechanisms.py, so the screen shows exactly what was tested.
Rating(team) = (EPA allowed so far + K * prior) / (plays so far + K) - league mean to date,
prior = last season's league mean + RHO * (the team's last-season value - last season's league mean).
"""
from __future__ import annotations

import pandas as pd

K_PLAYS = 250.0  # plays of prior (about 7 games of dropbacks, 10 of rushes)
RHO = 0.35       # share of last season's distance from the league that carries over


def team_table(plays: pd.DataFrame, kind: str) -> pd.DataFrame:
    """One row per defense: summed EPA and play count. kind: 'pass' (dropbacks) or 'run' (designed runs)."""
    flag = "db" if kind == "pass" else "run"
    p = plays[plays[flag] == 1]
    g = p.groupby("defteam")["epa"].agg(epa="sum", n="count")
    return g


def rate(prior: pd.DataFrame, cur: pd.DataFrame, k: float = K_PLAYS, rho: float = RHO) -> pd.Series:
    """prior, cur: frames indexed by defense with columns epa (sum) and n (plays). Either may be empty."""
    cn = float(cur["n"].sum()) if len(cur) else 0.0
    pn = float(prior["n"].sum()) if len(prior) else 0.0
    lgc = float(cur["epa"].sum()) / cn if cn else None
    lgp = float(prior["epa"].sum()) / pn if pn else None
    if lgp is None:
        lgp = lgc if lgc is not None else 0.0
    if lgc is None:
        lgc = lgp
    teams = prior.index.union(cur.index)
    pv = (prior["epa"] / prior["n"]).reindex(teams) if pn else pd.Series(index=teams, dtype=float)
    pri = lgp + rho * (pv.fillna(lgp) - lgp)
    e = cur["epa"].reindex(teams).fillna(0.0) if len(cur) else pd.Series(0.0, index=teams)
    n = cur["n"].reindex(teams).fillna(0.0) if len(cur) else pd.Series(0.0, index=teams)
    return (e + k * pri) / (n + k) - lgc
