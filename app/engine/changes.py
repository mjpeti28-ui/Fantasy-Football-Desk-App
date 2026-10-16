"""What changed since you last looked: injury tags, projections and roster moves across pulls.

Each time the inputs change (a refresh, or the weekly run), the engine saves a compact state of
the league to app/cache.nosync/states/<id>.json: who owns whom, injury tags, this week's ESPN
projection and our rest-of-season number. The digest compares the state you last saw with the
newest one. The very first time, the baseline is the weekly run's latest espn_live.csv snapshot,
so there's something to compare against straight away.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from . import data as d, store

ME = 3
KEEP = 40
BAD = {"OUT": 3, "INJURY_RESERVE": 4, "SUSPENSION": 4, "DOUBTFUL": 2, "QUESTIONABLE": 1, "DAY_TO_DAY": 1}


def _dir() -> Path:
    return store.local_dir("states")


def _norm(s) -> str:
    s = str(s or "").strip().upper()
    return "" if s in ("", "NAN", "NONE", "ACTIVE", "NORMAL") else s


def _f(x) -> Optional[float]:
    try:
        v = float(x)
        return None if v != v else round(v, 2)
    except (TypeError, ValueError):
        return None


def _state_now() -> dict:
    inp = d.load()
    wk = int(inp.rosters["week"].max())
    ro = inp.rosters[inp.rosters["week"] == wk]
    proj = inp.espn_proj[inp.espn_proj["week"] == inp.week].set_index("espn_id")["espn_proj"].to_dict()
    lp = inp.ledger_players.set_index("espn_id")
    players: Dict[str, dict] = {}
    for r in ro.itertuples():
        pid = int(r.espn_id)
        players[str(pid)] = {"n": str(r.name), "p": str(r.position).replace("D/ST", "DST"), "o": int(r.team_id),
                             "s": _norm(getattr(r, "injury_status", "")), "w": _f(proj.get(pid)),
                             "e": _f(lp.at[pid, "est_ppg"]) if pid in lp.index else None}
    # free agents worth tracking: anyone our model has at 6+ ppg, so adds/drops and their tags show
    for pid, r in lp.iterrows():
        if str(int(pid)) in players or (_f(r.get("est_ppg")) or 0) < 6:
            continue
        players[str(int(pid))] = {"n": str(r.get("name")), "p": str(r.get("position")), "o": None, "s": _norm(r.get("status")),
                                  "w": _f(proj.get(int(pid))), "e": _f(r.get("est_ppg"))}
    t = max(v for v in inp.ages.values() if v)
    return {"id": int(t), "at": t, "week": inp.week, "source": "Desk refresh", "players": players,
            "teams": {str(k): inp.team_name(int(k)) for k in sorted(ro["team_id"].unique())}}


def _baseline() -> Optional[dict]:
    """The weekly run's newest espn_live.csv, as a first state to compare against."""
    snaps = sorted((d.ROOT / "history" / "snapshots" / str(d.load().season)).glob("*/espn_live.csv"))
    if not snaps:
        return None
    try:
        df = pd.read_csv(snaps[-1])
    except Exception:
        return None
    inp = d.load()
    players = {}
    for r in df.itertuples():
        owner = int(r.on_team_id) if pd.notna(r.on_team_id) and int(r.on_team_id) > 0 else None
        players[str(int(r.espn_id))] = {"n": str(r.name), "p": str(r.position).replace("D/ST", "DST"), "o": owner,
                                        "s": _norm(r.injury_status), "w": _f(r.espn_proj) if int(r.week) == inp.week else None, "e": None}
    t = snaps[-1].stat().st_mtime
    return {"id": int(t), "at": t, "week": int(df["week"].max()), "source": "weekly run", "players": players, "teams": {}}


def capture() -> dict:
    """Save the current state if it's new; returns it."""
    now = _state_now()
    p = _dir() / f"{now['id']}.json"
    if not p.exists():
        if not list(_dir().glob("*.json")):
            b = _baseline()
            if b and b["id"] < now["id"]:
                (_dir() / f"{b['id']}.json").write_text(json.dumps(b))
        tmp = p.with_suffix(".tmpw")
        tmp.write_text(json.dumps(now))
        tmp.replace(p)
        for old in sorted(_dir().glob("*.json"))[:-KEEP]:
            try:
                old.unlink()
            except OSError:
                pass
    return now


def _states() -> List[Path]:
    return sorted(_dir().glob("*.json"), key=lambda q: int(q.stem))


def digest(since: Optional[int] = None) -> dict:
    """Changes between the state at `since` (or the one before the newest) and the newest."""
    capture()
    ps = _states()
    if len(ps) < 2:
        return {"from": None, "to": int(ps[-1].stem) if ps else None, "items": [], "note": "Nothing to compare yet: this is the first pull Desk has seen."}
    cur = json.loads(ps[-1].read_text())
    older = [q for q in ps[:-1] if since is None or int(q.stem) <= since]
    if since is not None and since >= int(ps[-1].stem):
        return {"from": since, "to": cur["id"], "at_from": None, "at_to": cur["at"], "items": [], "note": "Nothing new since you last looked."}
    base = json.loads((older[-1] if older else ps[0]).read_text())
    inp = d.load()
    names = {str(t): inp.team_name(int(t)) for t in inp.settings.get("team_names", {})}
    from .views import _opponent
    opp = _opponent(inp, ME, inp.week)
    items: List[dict] = []
    B, C = base["players"], cur["players"]

    def add(kind, pid, text, tone="", team=None, rank=0):
        p = C.get(pid) or B.get(pid) or {}
        items.append({"kind": kind, "id": int(pid), "name": p.get("n"), "pos": p.get("p"), "team": team, "text": text, "tone": tone, "rank": rank})

    for pid, c in C.items():
        b = B.get(pid)
        mine, theirs = c.get("o") == ME, opp is not None and c.get("o") == opp
        # roster moves
        bo = b.get("o") if b else None
        if b is not None and bo != c.get("o"):
            if bo is None and c.get("o") is not None:
                add("move", pid, f"{names.get(str(c['o']), 'Team')} added him", "", c["o"], 2 if not mine else 5)
            elif bo is not None and c.get("o") is None:
                add("move", pid, f"{names.get(str(bo), 'Team')} dropped him" + (f" (our model: {c['e']:.1f} ppg)" if c.get("e") else ""), "up" if (c.get("e") or 0) >= 8 else "", bo, 3)
            else:
                add("move", pid, f"moved from {names.get(str(bo), 'a team')} to {names.get(str(c['o']), 'a team')} (trade)", "", c.get("o"), 4)
        if not b or not (mine or theirs):
            continue
        who = "mine" if mine else "opp"
        # injury tags
        if b.get("s") != c.get("s"):
            worse = BAD.get(c.get("s"), 0) > BAD.get(b.get("s"), 0)
            text = f"{(b.get('s') or 'active').replace('_', ' ').lower()} → {(c.get('s') or 'active').replace('_', ' ').lower()}"
            add(who, pid, text, "down" if worse == mine else "up", c.get("o"), 10 if mine else 8)
        # this week's projection
        if cur["week"] == base["week"] and b.get("w") is not None and c.get("w") is not None and abs(c["w"] - b["w"]) >= 1.5:
            up = c["w"] > b["w"]
            add(who, pid, f"ESPN this week {b['w']:.1f} → {c['w']:.1f}", "up" if up == mine else "down", c.get("o"), 6 if mine else 4)
        # our rest-of-season number
        if mine and b.get("e") is not None and c.get("e") is not None and abs(c["e"] - b["e"]) >= 0.75:
            add("mine", pid, f"our rest-of-season number {b['e']:.1f} → {c['e']:.1f} ppg", "up" if c["e"] > b["e"] else "down", c.get("o"), 5)
    for pid, b in B.items():  # rostered before, now a free agent we don't track (model under 6 ppg)
        if pid not in C and b.get("o") is not None:
            p = b
            items.append({"kind": "move", "id": int(pid), "name": p.get("n"), "pos": p.get("p"), "team": b["o"],
                          "text": f"{names.get(str(b['o']), 'Team')} dropped him", "tone": "", "rank": 3})
    items.sort(key=lambda x: -x["rank"])
    return {"from": base["id"], "to": cur["id"], "at_from": base["at"], "at_to": cur["at"], "from_source": base.get("source"),
            "week": cur["week"], "opp": opp, "opp_name": names.get(str(opp)) if opp else None, "items": items}
