"""Local HTTP engine for the Desk app (read-only toward ESPN).

    python3 -m uvicorn app.engine.server:app --port 8765      (from the repo root)

Binds to 127.0.0.1 only. Endpoints:
    GET  /api/health
    GET  /api/warm                   what the start-up warm-up has precomputed so far
    GET  /api/snapshot?team=3        every view for the app, from cached files (fast)
    POST /api/refresh                pull ESPN rosters + this week's projections in the background (?full=true: weekly_run)
    GET  /api/refresh                refresh status + FantasyPros API calls used today
    GET  /api/live                   live games folded into this week's matchup (ESPN public feeds)
    GET  /api/matchups               defenses in brief + your players / opponent's starters against theirs
    GET  /api/matchups/defense/DAL   one defense: scheme, allowed, who hurt it, defenders
    GET  /api/matchups/player/ID     one player vs coverage + his start/sit card (?opp=PHI to look through a different defense)
    GET  /api/analytics              season analytics for every relevant QB/RB/WR/TE, each stat labelled by the 2021-25 backtest
    GET  /api/analytics/league       team-by-week table: points, best possible lineup, luck, drafted vs added, draft board
    POST /api/trade/evaluate         {"team_a":3,"give_a":[ids],"team_b":7,"give_b":[ids],"espn_view":false}
"""
from __future__ import annotations

import subprocess
import sys
import threading
from datetime import datetime, timezone
import time
from typing import List, Optional

import json
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import analytics, changes, data as d, dossier, journal, league, matchups, matchups_card, matchups_pff, roles, shifts, snapshot, trades, waivers

# ---------- warm-up: precompute the slow views in the background as soon as the engine starts ----------
_warm: dict = {"state": "idle", "started": None, "finished": None, "steps": {}, "again": False}
_files: dict = {}


def _player(pid: int):
    """Player file, cached per (player, role tables, input files)."""
    roles.tables()
    k = (int(pid), repr(roles._CACHE.get("tkey")), d.ages_key())
    if k not in _files:
        if len(_files) > 400:
            _files.clear()
        _files[k] = dossier.engine_build(int(pid))
    return _files[k]


def _my_files() -> None:
    inp = d.load()
    ro = inp.rosters[inp.rosters["week"] == inp.rosters["week"].max()]
    for pid in ro[ro["team_id"] == 3]["espn_id"].astype(int):
        try:
            _player(pid)
        except Exception:
            pass


WARM_STEPS = [
    ("inputs", "Reading data", lambda: (d.load(), changes.capture())),
    ("model", "Loading the model", lambda: league._ctx()),
    ("waivers", "Waiver board", lambda: waivers.board(6)),
    ("roles", "Player roles", lambda: roles.board()),
    ("shifts", "Usage shifts", lambda: shifts.engine_build()),
    ("league", "League odds", lambda: league.analysis(2000)),
    ("files", "Your player files", _my_files),
    ("matchups", "Matchup tables", lambda: (matchups.warm(), matchups_card.grades())),
    ("analytics", "Season analytics", lambda: (analytics.players(), analytics.league())),
]


def _warm_up() -> None:
    while True:
        _warm.update(state="running", started=time.time(), finished=None, again=False,
                     steps={k: {"label": label, "state": "waiting"} for k, label, _ in WARM_STEPS})
        for k, _label, fn in WARM_STEPS:
            t0 = time.time()
            _warm["steps"][k]["state"] = "running"
            try:
                fn()
                _warm["steps"][k].update(state="ready", seconds=round(time.time() - t0, 1))
            except Exception as e:  # a failed step shouldn't stop the others; the endpoint will retry on demand
                _warm["steps"][k].update(state="failed", error=f"{type(e).__name__}: {e}"[:300], seconds=round(time.time() - t0, 1))
        _warm.update(state="done", finished=time.time())
        if not _warm.get("again"):
            return


def start_warm() -> None:
    if _warm["state"] == "running":
        _warm["again"] = True
        return
    threading.Thread(target=_warm_up, daemon=True, name="desk-warm").start()


@asynccontextmanager
async def lifespan(_app):
    start_warm()
    yield


app = FastAPI(title="Desk engine", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_refresh = {"state": "idle", "started": None, "finished": None, "log": []}
_ctx_cache: dict = {}


def _np(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.ndarray, set, frozenset)):
        return list(o)
    return str(o)


def _json(obj) -> JSONResponse:
    return JSONResponse(json.loads(json.dumps(obj, default=_np)))


def _clear_caches() -> None:
    _live_snap["v"] = None
    d.load.cache_clear()
    _ctx_cache.clear()
    _files.clear()


# ---------- live games (app/engine/live.py, reloaded whenever that file changes, so it can be
# improved while Desk runs without restarting the engine)
_live_mod: dict = {"m": None, "mtime": 0.0}
_live_snap: dict = {"t": 0.0, "v": None}
_live_lock = threading.Lock()


def _live():
    import importlib
    from pathlib import Path
    mt = Path(__file__).with_name("live.py").stat().st_mtime
    if _live_mod["m"] is None or mt != _live_mod["mtime"]:
        from . import live as m
        if _live_mod["m"] is not None:
            m = importlib.reload(m)
        _live_mod.update(m=m, mtime=mt)
    return _live_mod["m"]


@app.get("/api/live")
def live_board(team: int = 3):
    """Max's matchup while games are on: points so far, rest-of-game projection, P(win), plays."""
    with _live_lock:
        if _live_snap["v"] is None or time.time() - _live_snap["t"] > 120:
            _live_snap.update(t=time.time(), v=snapshot.build(team))
        try:
            return _json(_live().board(_live_snap["v"], _player))
        except Exception as e:
            raise HTTPException(502, f"live feed: {type(e).__name__}: {e}")


@app.get("/api/health")
def health():
    inp = d.load()
    return _json({"ok": True, "season": inp.season, "week": inp.week, "data_age": inp.ages,
                  "engine_version": snapshot.ENGINE_VERSION, "warm": _warm["state"]})


@app.get("/api/warm")
def warm_status():
    return _warm


@app.get("/api/snapshot")
def get_snapshot(team: int = 3):
    return _json(snapshot.build(team))


def _fp_budget() -> dict:
    """FantasyPros API calls spent today, read from the client's own per-day (UTC) ledger file."""
    import fantasypros_api as fp
    day = datetime.now(timezone.utc).date().isoformat()
    try:
        used = int(json.loads((fp.CACHE_DIR / f"calls-{day}.json").read_text()).get("calls", 0))
    except (OSError, ValueError):
        used = 0
    return {"day": day, "used": used, "limit": fp.DAILY_LIMIT, "reserve": fp.DAILY_SAFETY_MARGIN}


def _run_refresh(season: int, week: int, quick: bool = False, full: bool = False) -> None:
    steps = [
        ("ESPN rosters and scores", [sys.executable, "espn_history.py", str(season)]),
        (f"ESPN projections, week {week}", [sys.executable, "espn_history.py", "proj", str(season), str(week), str(week)]),
    ]
    if full:  # the whole Tuesday routine (it pulls ESPN itself): every source, snapshot, health, ledger, season odds
        steps = [("Weekly run: all sources, ledger, season odds", [sys.executable, "tools/weekly_run.py"])]
    if not quick:  # the button always rebuilds usage from nflverse
        steps.append(("nflverse play-by-play (roles)", [sys.executable, "-m", "app.engine.roles_data", str(season)]))
    else:  # the automatic refreshes rebuild it only when nflverse has published new snap/charting data (HEAD checks, no download)
        steps.append(("nflverse usage (only if new data is out)", [sys.executable, "-m", "app.engine.roles_data", "--if-stale", str(season)]))
    # optional: PFF slot/inline alignment for the WR/TE styles (at most one call per 12h; no key or an error
    # exits 0 and the styles fall back to free data)
    steps.append(("PFF alignment (WR/TE styles)", [sys.executable, "-m", "app.engine.roles_pff", str(season)]))
    steps.append(("PFF matchups (context)", [sys.executable, "-m", "app.engine.matchups_pff", str(season)]))
    _refresh.update(state="running", started=time.time(), finished=None, log=[], quick=quick, full=full, step=None)
    ok = True
    for label, cmd in steps:
        _refresh["step"] = label
        t0 = time.time()
        try:
            r = subprocess.run(cmd, cwd=d.ROOT, capture_output=True, text=True, timeout=1800 if full else 600)
            # weekly_run exits 0 even when a step fails; it marks those lines [FAIL]
            fails = [x for x in r.stdout.splitlines() if x.startswith("[FAIL]")]
            good = r.returncode == 0 and not fails
            tail = fails[:3] or (r.stdout + r.stderr).strip().splitlines()[-2:]
        except subprocess.TimeoutExpired:
            good, tail = False, ["timed out"]
        ok &= good
        _refresh["log"].append({"step": label, "ok": good, "seconds": round(time.time() - t0), "tail": tail})
    _clear_caches()
    try:
        changes.capture()
    except Exception:
        pass
    _refresh.update(state="done" if ok else "failed", finished=time.time(), step=None)
    start_warm()  # recompute the slow views on the new data before you click into them


@app.post("/api/refresh")
def start_refresh(quick: bool = False, full: bool = False):
    """quick=true pulls rosters and projections only (what the automatic refreshes use).
    full=true runs tools/weekly_run.py (all sources, ledger, season odds; ~25 FantasyPros API calls)."""
    if _refresh["state"] == "running":
        return _refresh
    inp = d.load()
    threading.Thread(target=_run_refresh, args=(inp.season, inp.week, quick, full), daemon=True).start()
    return {**_refresh, "state": "running", "quick": quick, "full": full, "fp": _fp_budget()}


@app.get("/api/changes")
def changes_digest(since: Optional[int] = None):
    """Injury tags, projections and roster moves since the state you last saw (default: the previous pull)."""
    return _json(changes.digest(since))


@app.get("/api/refresh")
def refresh_status():
    return {**_refresh, "fp": _fp_budget()}


class TradeIn(BaseModel):
    team_a: int = 3
    give_a: List[int]
    team_b: int
    give_b: List[int]
    espn_view: bool = False
    sims: int = 300


def _context(espn_view: bool):
    import trade_value as tv  # imported lazily: pulls in the model stack
    key = ("espn" if espn_view else "model", d.load().week)
    if key not in _ctx_cache:
        params = tv.source_params("espn") if espn_view else None
        _ctx_cache[key] = tv.load_context(d.load().season, d.load().week, params=params)
    return tv, _ctx_cache[key]


@app.post("/api/trade/evaluate")
def trade_evaluate(t: TradeIn):
    tv, ctx = _context(t.espn_view)
    for team, ids in ((t.team_a, t.give_a), (t.team_b, t.give_b)):
        missing = [i for i in ids if i not in ctx.rosters.get(team, [])]
        if missing:
            raise HTTPException(400, f"players {missing} are not on team {team}'s roster")
    return _json(tv.evaluate(ctx, t.team_a, t.give_a, t.team_b, t.give_b, sims=t.sims))


@app.get("/api/league")
def league_base(sims: int = 2000):
    """Live rest-of-season odds for every team, plus all rosters (a few seconds the first time)."""
    return _json({"sims": sims, "week": d.load().week, "odds": league.base(sims), "rosters": league.rosters(),
                  "noise_pp": round(100 * 0.5 / (sims ** 0.5), 1), "analysis": league.analysis(sims)})


class OutIn(BaseModel):
    id: int
    weeks: Optional[int] = None   # None = rest of season


class WhatIfIn(BaseModel):
    out: List[OutIn]
    sims: int = 2000


@app.post("/api/whatif")
def league_whatif(w: WhatIfIn):
    return _json(league.whatif([o.model_dump() for o in w.out], w.sims))


@app.get("/api/trades/history")
def trades_history():
    return _json(trades.history())


class DealIn(BaseModel):
    partner: int
    give: List[int]
    get: List[int]


@app.post("/api/trades/evaluate")
def trades_evaluate(t: DealIn):
    try:
        return _json(trades.evaluate(t.partner, t.give, t.get))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/trades/odds")
def trades_odds(t: DealIn):
    return _json(trades.odds(t.partner, t.give, t.get))


class IdeasIn(BaseModel):
    partner: Optional[int] = None


@app.post("/api/trades/ideas")
def trades_ideas(i: IdeasIn):
    return _json(trades.start_ideas(i.partner))


@app.get("/api/trades/ideas")
def trades_ideas_status():
    return _json(trades.ideas_status())


@app.get("/api/player/{pid}")
def player_file(pid: int):
    """Full player file: role, fingerprint, role map, game log, comps, the read (~1 s, more the first time)."""
    out = _player(pid)
    if out is None:
        raise HTTPException(404, "no usage data for this player")
    return _json(out)


@app.get("/api/roles")
def roles_board():
    """Role profiles (style, tier, traits, trends) for every QB/RB/WR/TE who has played this season."""
    return _json(roles.board())


@app.get("/api/shifts")
def shifts_board():
    """League-wide snap/share changes with owners, zones and how often changes like them held."""
    return _json(shifts.engine_build())


@app.get("/api/waivers")
def waivers_board(top: int = 6):
    """Free-agent shortlist with best drop and team gain (several seconds the first time)."""
    return _json(waivers.board(top))


class KeepIn(BaseModel):
    ids: List[int]


@app.post("/api/waivers/keep")
def waivers_keep(inp: KeepIn):
    """Players you won't drop: saved, and the next /api/waivers never offers them as the drop."""
    return {"keep": waivers.save_keep(inp.ids)}


@app.get("/api/matchups")
def matchups_board(team: int = 3):
    """Every defense in brief, this week's schedule, your players and your opponent's starters against their defenses."""
    try:
        return _json(matchups_card.annotate_board(matchups.view(snapshot.build(team), team)))
    except Exception as e:  # tables not built yet, or the first download failed (offline)
        raise HTTPException(503, f"matchup data isn't available yet: {type(e).__name__}: {e}"[:300])


@app.get("/api/matchups/defense/{abbr}")
def matchups_defense(abbr: str):
    """One defense: scheme history, what it allows this season, who has hurt it, and its defenders."""
    out = matchups.defense_view(abbr)
    if out is None:
        raise HTTPException(404, "unknown team")
    try:
        out["pff"] = matchups_pff.defense(abbr, matchups.current_season())
    except Exception:  # noqa: BLE001 -- optional PFF context
        out["pff"] = None
    return _json(out)


@app.get("/api/matchups/player/{espn_id}")
def matchups_player(espn_id: int, opp: Optional[str] = None, team: int = 3):
    """One player's usage this season and his splits against coverage, seen through a defense (default: this week's)."""
    snap = snapshot.build(team)
    out = matchups.player_view(espn_id, opp, int(snap["season"]), int(snap["week"]))
    if out is None:
        raise HTTPException(404, "no matchup data for this player")
    out["card"] = matchups_card.build(out, snap)
    try:
        out["pff"] = matchups_pff.player(out, int(snap["season"]))
    except Exception:  # noqa: BLE001 -- optional PFF context
        out["pff"] = None
    try:
        out["compare"] = matchups_card.compare(out, out["card"], out["pff"])
    except Exception as e:  # noqa: BLE001 -- the Compare rows are a convenience; the card above stands on its own
        out["compare"] = {"opp": out.get("opp"), "read": None, "rows": [], "error": f"{type(e).__name__}: {e}"[:200]}
    return _json(out)


@app.get("/api/analytics")
def analytics_players():
    try:
        return _json(analytics.players())
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"analytics failed: {type(e).__name__}: {e}")


@app.get("/api/analytics/league")
def analytics_league():
    try:
        return _json(analytics.league())
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"league analytics failed: {type(e).__name__}: {e}")


@app.get("/api/journal")
def journal_list():
    return _json(journal.entries())


class EntryIn(BaseModel):
    kind: str
    title: str
    why: str = ""
    confidence: Optional[float] = None
    payload: dict = {}
    model: dict = {}
    model_p: Optional[float] = None
    week: Optional[int] = None
    status: Optional[str] = None


@app.post("/api/journal")
def journal_add(e: EntryIn):
    try:
        return _json(journal.add(**e.model_dump()))
    except ValueError as x:
        raise HTTPException(400, str(x))


class EntryPatch(BaseModel):
    why: Optional[str] = None
    confidence: Optional[float] = None
    status: Optional[str] = None
    title: Optional[str] = None


@app.patch("/api/journal/{eid}")
def journal_update(eid: str, p: EntryPatch):
    try:
        return _json(journal.update(eid, **p.model_dump()))
    except KeyError:
        raise HTTPException(404, "no such entry")
    except ValueError as x:
        raise HTTPException(400, str(x))


@app.delete("/api/journal/{eid}")
def journal_delete(eid: str):
    journal.delete(eid)
    return {"ok": True}
