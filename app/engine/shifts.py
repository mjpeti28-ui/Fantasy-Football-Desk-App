"""League-wide usage shifts: every QB/RB/WR/TE's snap and share change, two ways —
the last two games vs his earlier games, and this season vs last — with who rosters him, his
injury status, the model's rest-of-season value, and what changes like his have historically
meant (hold rates and the next-week check against ESPN's projection, from roles_fit.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from . import roles as R
except ImportError:
    import roles as R


def _f(x, nd=1):
    try:
        x = float(x)
        return None if np.isnan(x) else round(x, nd)
    except (TypeError, ValueError):
        return None


def zone(pos: str, ds: float | None, dh: float | None) -> str | None:
    """Same thresholds as the trend tags (roles.trends)."""
    if ds is None:
        return None
    if ds >= R.SNAP_T:
        return "snap_up"
    if ds <= -R.SNAP_T:
        return "snap_down"
    if dh is None or np.isnan(dh):
        return None
    if dh >= R.SHARE_T:
        return "share_up_with_snaps" if ds >= 10 else "share_spike"
    if dh <= -R.SHARE_T and ds > -10:
        return "share_drop"
    return None


def season_zone(ds, dh):
    if ds is None:
        return None
    if ds >= R.SNAP_T:
        return "snap_up"
    if ds <= -R.SNAP_T:
        return "snap_down"
    if dh is not None and dh >= R.SHARE_T:
        return "share_up"
    if dh is not None and dh <= -R.SHARE_T:
        return "share_down"
    return None


ROUTE_T, SLOT_T = 15, 20  # role-mover thresholds, points of route share / slot rate (descriptive; not backtested as signals)


def role_moves(split: pd.DataFrame, weekly: pd.DataFrame) -> dict:
    """WR/TE route share (PFF routes / team dropbacks) and slot rate, last two weeks vs the weeks before,
    from roles_pff.recent_split. Needs a game in the recent weeks (one, while the newest week is under way) and 2+
    before them. gsis -> dict."""
    if split is None or not len(split):
        return {}
    lo, hi = int(split["lo"].iloc[0]), int(split["hi"].iloc[0])
    w = weekly[weekly["pos"].isin(["WR", "TE"]) & (weekly["week"] <= hi)]
    rec, ear = w[w["week"] >= lo], w[w["week"] < lo]
    db_r, db_e = rec.groupby("gsis")["t_db"].sum(), ear.groupby("gsis")["t_db"].sum()
    n_r, n_e = rec.groupby("gsis").size(), ear.groupby("gsis").size()
    out = {}
    for g, x in split.iterrows():
        if n_r.get(g, 0) < 1 or n_e.get(g, 0) < 2 or not db_r.get(g) or not db_e.get(g) or not x["r_pass_plays"] or not x["e_pass_plays"]:
            continue
        fr, to = min(100.0, 100 * x["e_routes"] / db_e[g]), min(100.0, 100 * x["r_routes"] / db_r[g])
        fs, ts = min(100.0, 100 * x["e_slot_snaps"] / x["e_pass_plays"]), min(100.0, 100 * x["r_slot_snaps"] / x["r_pass_plays"])
        out[g] = {"label": f"weeks {lo}–{hi} vs 1–{lo - 1}", "from_route": _f(fr, 0), "to_route": _f(to, 0), "d_route": _f(to - fr, 0),
                  "from_slot": _f(fs, 0), "to_slot": _f(ts, 0), "d_slot": _f(ts - fs, 0),
                  "games": int(n_r[g]), "move": bool(abs(to - fr) >= ROUTE_T or abs(ts - fs) >= SLOT_T)}
    return out


def build(board: dict, prev_agg: pd.DataFrame, e2g: dict, owners: dict, ledger: pd.DataFrame | None,
          proj: dict, model: dict, week: int, pro_team: dict, headshot, roles_pff: dict | None = None) -> dict:
    rows = []
    for pid, p in board["players"].items():
        pos = p["pos"]; s = p.get("series")
        if pos not in ("QB", "RB", "WR", "TE") or not s or len(s["weeks"]) < 2:
            continue
        sn = np.array(s["snap"], float); sh = np.array([np.nan if v is None else v for v in s["share"]], float)
        if pos == "QB":
            sh = np.full(len(sn), np.nan)
        recent = None
        if len(sn) >= 4:
            recent = {"from_snap": _f(sn[:-2].mean()), "to_snap": _f(sn[-2:].mean()),
                      "from_share": _f(np.nanmean(sh[:-2])) if pos != "QB" else None, "to_share": _f(np.nanmean(sh[-2:])) if pos != "QB" else None,
                      "label": f"weeks {s['weeks'][-2]}–{s['weeks'][-1]} vs {s['weeks'][0]}–{s['weeks'][-3]}"}
            recent["d_snap"] = _f(recent["to_snap"] - recent["from_snap"])
            recent["d_share"] = _f(recent["to_share"] - recent["from_share"]) if recent["to_share"] is not None and recent["from_share"] is not None else None
            recent["zone"] = zone(pos, recent["d_snap"], recent["d_share"])
            if recent["zone"] and pos in model["trend"]:
                key = recent["zone"]
                cause = next((t.get("cause") for t in p["trends"] if t["kind"] == "snap_up" and t.get("cause")), None)
                if key == "snap_up" and cause:
                    key = "snap_up_out"; recent["cause"] = cause
                    recent["hold_if_still_out"] = model["trend"][pos].get("snap_up_still_out", {}).get("hold")
                recent["hold"] = model["trend"][pos].get(key, {}).get("hold")
                e = model.get("trend_pts", {}).get(pos, {}).get(key, {})
                recent["espn"] = {k: e.get(k) for k in ("rel", "lo", "hi")} if e else None
        season = None
        g = e2g.get(int(pid))
        if g is not None and g in prev_agg.index and prev_agg.loc[g]["games"] >= 6 and len(sn) >= 2:
            pr = prev_agg.loc[g]
            key = "db_sh" if pos == "QB" else "opp_sh" if pos == "RB" else "tgt_sh"
            ps = _f(float(pr[key]) * 100) if not pd.isna(pr.get(key)) else None
            cs = _f(np.nanmean(sh)) if pos != "QB" else None
            season = {"from_snap": _f(pr["snap"]), "to_snap": _f(sn.mean()), "from_share": ps if pos != "QB" else None, "to_share": cs,
                      "label": f"{board['season']} so far vs {board['season'] - 1}"}
            season["d_snap"] = _f(season["to_snap"] - season["from_snap"])
            season["d_share"] = _f(season["to_share"] - season["from_share"]) if season["to_share"] is not None and season["from_share"] is not None else None
            season["zone"] = season_zone(season["d_snap"], season["d_share"])
            if season["zone"] and pos in model.get("season_shift", {}):
                season["hold"] = model["season_shift"][pos].get(season["zone"], {}).get("hold")
        L = None
        if ledger is not None and int(pid) in ledger.index:
            x = ledger.loc[int(pid)]
            L = {"est_ppg": _f(x.get("est_ppg")), "avail": _f(x.get("avail"), 3), "status": x.get("status") if isinstance(x.get("status"), str) else None}
        own = owners.get(int(pid))
        ab = pro_team.get(int(pid)) or p.get("team")
        rows.append({"id": int(pid), "name": p["name"], "pos": pos, "team": ab, "headshot": headshot(int(pid), pos, ab),
                     "owner": own, "games": p["games"], "tier": p["tier"], "style": (p.get("style") or {}).get("name"),
                     "series": s, "recent": recent, "season": season, "pff": (roles_pff or {}).get(g), "ledger": L, "espn_week": _f(proj.get(int(pid))),
                     "injury": (own or {}).get("injury") or (L or {}).get("status")})
    zones = {}
    for pos in ("RB", "WR", "TE"):
        t, e = model["trend"].get(pos, {}), model.get("trend_pts", {}).get(pos, {})
        zones[pos] = {k: {"hold": t.get(k, {}).get("hold"), "n": t.get(k, {}).get("n"), "espn": e.get(k)} for k in
                      ("snap_up", "snap_up_out", "snap_up_still_out", "share_up_with_snaps", "share_spike", "snap_down", "share_drop")}
        zones[pos]["season"] = model.get("season_shift", {}).get(pos, {})
    return {"season": board["season"], "through_week": board["through_week"], "partial": board.get("partial", False), "week": week, "rows": rows, "zones": zones,
            "thresholds": {"snap": R.SNAP_T, "share": R.SHARE_T, "snap_with": 10, "route": ROUTE_T, "slot": SLOT_T}}


def engine_build() -> dict:
    from . import data as d, views
    season, cur, prev, ids = R.tables()
    b = R.board()
    inp = d.load()
    prev_agg = R.aggregate(prev, ("gsis", "pos")).set_index("gsis") if prev is not None else pd.DataFrame()
    m = ids.dropna(subset=["gsis_id", "espn_id"]).drop_duplicates("espn_id")
    e2g = dict(zip(m["espn_id"].astype(int), m["gsis_id"]))
    wk = inp.rosters["week"].max(); ro = inp.rosters[inp.rosters["week"] == wk]
    owners = {int(r.espn_id): {"team_id": int(r.team_id), "name": inp.team_name(int(r.team_id)),
                               "injury": r.injury_status if isinstance(getattr(r, "injury_status", None), str) else None} for r in ro.itertuples()}
    led = inp.ledger_players.set_index("espn_id") if len(inp.ledger_players) else None
    pr = inp.espn_proj[inp.espn_proj["week"] == inp.week]
    proj = dict(zip(pr["espn_id"].astype(int), pr["espn_proj"]))
    from . import roles_pff as rp
    moves = role_moves(rp.recent_split(season, cur, ids), cur)
    return build(b, prev_agg, e2g, owners, led, proj, R.load_model(), inp.week, views._pro_team(inp), views.headshot, moves)
