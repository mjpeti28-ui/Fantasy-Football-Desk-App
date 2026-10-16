"""Similar players: the nearest historical player-seasons (2016-25) at the same week of the season,
and what happened to them afterward. Backtested leave-one-season-out in roles_fit.py: the
10th-90th percentile range of comps' rest-of-season PPR/game held the actual outcome ~78-81% of
the time; as a point estimate the comp median is about as good as a simple regression on points
so far and last season (it is not an edge over the model)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import roles as R
except ImportError:
    import roles as R

HERE = Path(__file__).resolve().parent
COMP_W = {"ppr_g": 1, "prev_ppg_i": 1, "share": .7, "snap": .5, "STY": .35, "age_i": .4, "exp_i": .2, "has_prev": .5, "trend": .3}
COMP_K = 40
Q = [.1, .25, .5, .75, .9]


def comp_matrix(d: pd.DataFrame, pos: str, fill: dict | None = None):
    d = d.copy()
    d["has_prev"] = (d.prev_games.fillna(0) >= 4).astype(float)
    d["prev_ppg_i"] = np.where(d.has_prev == 1, d.prev_ppg, d.ppr_g * 0.8)
    d["age_i"] = d.age.fillna(26.0); d["exp_i"] = d.exp.fillna(3).clip(0, 12)
    cols, wts = [], []
    for c, v in COMP_W.items():
        if c == "STY":
            cols += list(R.STYLE[pos]); wts += [v] * len(R.STYLE[pos])
        else:
            cols.append(c); wts.append(v)
    for c in cols:
        d[c] = d[c].astype(float)
        d[c] = d[c].fillna(fill[c] if fill else d[c].median())
    return d, cols, np.array(wts)


@lru_cache(maxsize=1)
def library() -> pd.DataFrame:
    return pd.read_csv(HERE / "roles_library.csv.gz")


def find(pos: str, row: dict, W: int, k: int = COMP_K, n_show: int = 12) -> dict | None:
    """row: the live player's checkpoint features (same columns as the library)."""
    L = library()
    W = int(min(max(W, 2), 14))
    pool = L[(L.pos == pos) & (L.W == W) & (L.ros_team_games >= 3) & (L.ros_games >= 3)].reset_index(drop=True)
    if len(pool) < k:
        return None
    pool, cols, wts = comp_matrix(pool, pos)
    fill = {c: float(pool[c].median()) for c in cols}
    me, _, _ = comp_matrix(pd.DataFrame([row]), pos, fill)
    X = pool[cols].values; mu, sd = X.mean(0), X.std(0).clip(1e-6)
    Z = (X - mu) / sd * wts; z = (me[cols].values[0] - mu) / sd * wts
    D = np.sqrt(((Z - z) ** 2).sum(1)); nn = np.argsort(D)[:k]
    c = pool.iloc[nn].copy(); c["dist"] = D[nn]
    y = c.ros_ppg.values
    weeks = [float(v) for t in c.traj.fillna("") for v in str(t).split("|") if v != ""]
    weeks = np.array(weeks) if weeks else np.array([0.0])
    contrib = {col: float(abs((me[col].values[0] - c[col].mean()) / sd[i] * wts[i])) for i, col in enumerate(cols)}
    out = {
        "W": W, "k": k, "pool": int(len(pool)),
        "q": {f"p{int(q * 100)}": round(float(np.quantile(y, q)), 1) for q in Q},
        "avail": round(float((c.ros_games / c.ros_team_games).mean()), 3),
        "kept_role": round(float((c.ros_snap >= c.snap - 10).mean()), 3),
        "beat_now": round(float((c.ros_ppg >= c.ppr_g).mean()), 3),
        "weeks": {"n": int(len(weeks)), "q": [round(float(np.quantile(weeks, q)), 1) for q in (.1, .25, .5, .75, .9)],
                  "ge15": round(float((weeks >= 15).mean()), 3), "lt5": round(float((weeks < 5).mean()), 3),
                  "hist": np.histogram(weeks.clip(0, 40), bins=np.arange(0, 42, 2))[0].tolist()},
        "dist_ppg": np.histogram(y.clip(0, 30), bins=np.arange(0, 31, 1))[0].tolist(),
        "list": [],
        "all": [{"name": r["name"] if isinstance(r["name"], str) else "Unknown", "season": int(r.season), "ros_ppg": round(float(r.ros_ppg), 1),
                 "ppg_then": round(float(r.ppr_g), 1)} for _, r in c.iterrows()],
    }
    dmax = float(D[nn].max()) or 1.0
    for _, r in c.head(n_show).iterrows():
        traj = [None if v == "" else float(v) for v in str(r.traj if isinstance(r.traj, str) else "").split("|")] if isinstance(r.traj, str) else []
        out["list"].append({
            "name": r["name"] if isinstance(r["name"], str) else "Unknown", "season": int(r.season), "team": r.posteam,
            "sim": round(100 * (1 - float(r.dist) / (dmax * 1.15)), 0),
            "ppg_then": round(float(r.ppr_g), 1), "snap_then": round(float(r.snap), 0), "share_then": None if pd.isna(r.share) else round(float(r.share) * 100, 0),
            "age": None if pd.isna(r.age) else round(float(r.age), 1), "prev_ppg": None if pd.isna(r.prev_ppg) else round(float(r.prev_ppg), 1),
            "ros_ppg": round(float(r.ros_ppg), 1), "ros_games": int(r.ros_games), "ros_team_games": int(r.ros_team_games),
            "ros_snap": None if pd.isna(r.ros_snap) else round(float(r.ros_snap), 0), "traj": traj,
        })
    return out
