"""Live games: Max's matchup while games are on, from ESPN's public NFL feeds.

Read-only. server.py reloads this module whenever the file changes, so it can be improved while
Desk is running (no restart).

Sources (unofficial and undocumented, so expect them to change shape now and then):
  site.api.espn.com  .../scoreboard          games this week: clock, score, status, situation
  site.api.espn.com  .../summary?event=      box score stat lines, team stats, win probability
  sports.core.api.espn.com .../plays         play-by-play with participants (athlete ids + roles)

What it computes, per starter on both sides of the matchup:
  pts        fantasy points so far, from the box score (standard PPR; D/ST and K approximate, see NOTES)
  expected   what the pregame projection says he'd have by this point of his game
  live mean  pts + projection for the rest of his game, tilted by how much he's being used
  p10 / p90  from the position's actual/forecast spread, scaled to the time left
and, for the matchup, P(win) by simulation, plus how it moved play by play.

The rest-of-game model is deliberately simple and NOT fitted: remaining = projection x time left x
usage tilt, with the spread shrinking like sqrt(time left). Usage tilt = a Poisson-gamma update of
his opportunity rate (targets + carries), one game of prior weight. See NOTES.
"""
from __future__ import annotations

import bisect
import math
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None

SITE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"
TTL_LIVE = 12          # seconds; ESPN's own cache headers say 4
TTL_IDLE = 240
SIMS = 20000
PRIOR_GAMES = 4.0      # weight of the season-average opportunity rate, in games (fitted on 2025 play-by-play, see FIT)
TILT_RANGE = (0.5, 1.6)
NOTES = [
    "Points: standard PPR from ESPN's box score (1 per catch, 0.1 per rush/rec yard, 0.04 per pass yard, "
    "6 per rush/rec TD, 4 per pass TD, -2 INT / fumble lost, 2 per two-point try, read from the play text).",
    "D/ST and kickers use ESPN's default scoring as I remember it, not your league's settings (the "
    "league is private, so this feed can't read them). Treat those two as approximate.",
    "Rest-of-game = projection x time left x usage tilt, spread shrinking with sqrt(time left). Both "
    "checked on 2025 play-by-play (1,939 WR/TE/RB games): the sqrt rule matched the spread at every "
    "quarter, and usage so far is worth about 4 games of prior. Points scored so far add nothing beyond "
    "usage. Players are treated as independent (no same-game correlation).",
]
# What the 2025 check found (app/engine/live.py fit, 10/8): remaining points / (season ppg x time left),
# grouped by opportunities so far vs his usual by then. At halftime: under half his usual 0.89,
# 0.5-0.8x 0.82, normal 0.97, 1.2-1.6x 0.95, over 1.6x 1.09. After 3 quarters: 0.79 / 0.90 / 0.94 /
# 1.10 / 1.16. RMSE of remaining points: no tilt 5.283, k=4 5.273, k=1 5.318, k=0.25 5.627. Correlation
# of the leftover error with "points ahead of pace": 0.03. Spread of remaining/ppg at 25/50/75% played:
# 0.60/0.49/0.35 vs sqrt rule 0.60/0.49/0.35.
FIT = {"prior_games": 4.0, "player_games": 1939, "season": 2025}

ALIAS = {"WSH": "WAS", "LA": "LAR", "JAC": "JAX", "LV": "LV", "OAK": "LV", "SD": "LAC", "STL": "LAR"}


def _team(abbr: Optional[str]) -> str:
    a = (abbr or "").upper()
    return ALIAS.get(a, a)


# ---------------------------------------------------------------- fetching
_cache: Dict[str, Tuple[float, Any]] = {}
_lock = threading.Lock()
_errors: List[str] = []


def _get(url: str, ttl: float) -> Any:
    now = time.time()
    hit = _cache.get(url)
    if hit and now - hit[0] < ttl:
        return hit[1]
    try:
        r = httpx.get(url, timeout=10.0, headers={"User-Agent": "Desk/1 (personal fantasy app)"},
                      follow_redirects=True)
        r.raise_for_status()
        js = r.json()
    except Exception as e:  # stale data beats no data mid-game
        _errors.append(f"{time.strftime('%H:%M:%S')} {url.split('?')[0].rsplit('/', 2)[-2:]}: {type(e).__name__}")
        del _errors[:-5]
        if hit:
            return hit[1]
        raise
    _cache[url] = (now, js)
    if len(_cache) > 60:
        for k in sorted(_cache, key=lambda k: _cache[k][0])[:20]:
            _cache.pop(k, None)
    return js


def _id(ref: Optional[str], kind: str) -> Optional[str]:
    if not ref:
        return None
    m = re.search(rf"/{kind}/(-?\d+)", ref)
    return m.group(1) if m else None


# ---------------------------------------------------------------- games
def _clock_elapsed(period: int, clock_s: float, state: str) -> float:
    """Fraction of regulation played (0..1; OT counts as the last sliver)."""
    if state == "post":
        return 1.0
    if state != "in" or period <= 0:
        return 0.0
    if period <= 4:
        sec = (period - 1) * 900 + (900 - max(0.0, min(900.0, clock_s)))
        return max(0.0, min(1.0, sec / 3600.0))
    return 0.995  # overtime: nearly done


def scoreboard() -> List[dict]:
    js = _get(f"{SITE}/scoreboard", TTL_LIVE)
    out = []
    for ev in js.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        st = comp.get("status") or ev.get("status") or {}
        typ = st.get("type") or {}
        teams = {}
        for c in comp.get("competitors", []):
            t = c.get("team") or {}
            teams[c.get("homeAway")] = {"abbr": _team(t.get("abbreviation")), "id": str(t.get("id")),
                                        "score": int(float(c.get("score") or 0)), "name": t.get("shortDisplayName")}
        sit = comp.get("situation") or {}
        state = typ.get("state", "pre")
        period = int(st.get("period") or 0)
        clock = float(st.get("clock") or 0)
        g = {
            "event": str(ev.get("id")), "kickoff": ev.get("date"), "state": state,
            "detail": typ.get("shortDetail") or typ.get("detail"), "period": period,
            "clock": st.get("displayClock"), "elapsed": _clock_elapsed(period, clock, state),
            "home": teams.get("home", {}), "away": teams.get("away", {}),
            "possession": None, "down_dist": sit.get("downDistanceText") or sit.get("shortDownDistanceText"),
            "red_zone": bool(sit.get("isRedZone")), "last_play": (sit.get("lastPlay") or {}).get("text"),
        }
        pos_id = str(sit.get("possession")) if sit.get("possession") else None
        for side in ("home", "away"):
            if pos_id and teams.get(side, {}).get("id") == pos_id:
                g["possession"] = teams[side]["abbr"]
        out.append(g)
    return out


def _summary(event: str, live: bool) -> dict:
    return _get(f"{SITE}/summary?event={event}", TTL_LIVE if live else TTL_IDLE)


def _plays(event: str, live: bool) -> List[dict]:
    js = _get(f"{CORE}/events/{event}/competitions/{event}/plays?limit=500", TTL_LIVE if live else TTL_IDLE)
    items = list(js.get("items", []))
    pages = int(js.get("pageCount") or 1)
    for p in range(2, pages + 1):
        more = _get(f"{CORE}/events/{event}/competitions/{event}/plays?limit=500&page={p}", TTL_LIVE if live else TTL_IDLE)
        items += more.get("items", [])
    items.sort(key=lambda x: int(x.get("sequenceNumber") or 0))
    return items


# ---------------------------------------------------------------- box score -> stat lines
def _num(x) -> float:
    try:
        return float(str(x).replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


def box_lines(summary: dict) -> Dict[str, dict]:
    """athlete id -> {stat: value} from the box score."""
    lines: Dict[str, dict] = {}
    for team in (summary.get("boxscore") or {}).get("players", []) or []:
        abbr = _team((team.get("team") or {}).get("abbreviation"))
        for grp in team.get("statistics", []) or []:
            name = grp.get("name")
            labels = grp.get("labels") or []
            for a in grp.get("athletes", []) or []:
                aid = str((a.get("athlete") or {}).get("id"))
                row = dict(zip(labels, a.get("stats") or []))
                L = lines.setdefault(aid, {"team": abbr})
                if name == "passing":
                    c, _, att = str(row.get("C/ATT", "0/0")).partition("/")
                    L.update(cmp=_num(c), att=_num(att), pass_yd=_num(row.get("YDS")), pass_td=_num(row.get("TD")),
                             ints=_num(row.get("INT")))
                elif name == "rushing":
                    L.update(car=_num(row.get("CAR")), rush_yd=_num(row.get("YDS")), rush_td=_num(row.get("TD")))
                elif name == "receiving":
                    L.update(rec=_num(row.get("REC")), rec_yd=_num(row.get("YDS")), rec_td=_num(row.get("TD")),
                             tgt=_num(row.get("TGTS")))
                elif name == "fumbles":
                    L.update(fl=_num(row.get("LOST")))
                elif name == "kicking":
                    fm, _, fa = str(row.get("FG", "0/0")).partition("/")
                    xm, _, xa = str(row.get("XP", "0/0")).partition("/")
                    L.update(fgm=_num(fm), fga=_num(fa), xpm=_num(xm), xpa=_num(xa))
    return lines


def ppr(L: dict) -> float:
    return round(0.04 * L.get("pass_yd", 0) + 4 * L.get("pass_td", 0) - 2 * L.get("ints", 0)
                 + 0.1 * L.get("rush_yd", 0) + 6 * L.get("rush_td", 0)
                 + L.get("rec", 0) + 0.1 * L.get("rec_yd", 0) + 6 * L.get("rec_td", 0)
                 - 2 * L.get("fl", 0) + 2 * L.get("two_pt", 0), 2)


def _fg_points(dist: float) -> float:
    return 5.0 if dist >= 50 else 4.0 if dist >= 40 else 3.0


def kicker_points(plays: List[dict], name_short: str) -> float:
    """From play text: FGs by distance (3/4/5), PAT made 1, missed FG -1 (ESPN defaults)."""
    pts = 0.0
    for p in plays:
        tx = p.get("text") or ""
        if name_short not in tx or "NO PLAY" in tx.upper():
            continue
        typ = ((p.get("type") or {}).get("text") or "").lower()
        m = re.search(r"(\d+)\s*yard field goal", tx, re.I)
        if m and "field goal good" in typ:
            pts += _fg_points(float(m.group(1)))
        elif m and ("missed" in typ or "blocked" in typ):
            pts -= 1.0
        if re.search(rf"{re.escape(name_short)} extra point is GOOD", tx, re.I):
            pts += 1.0
    return pts


PA_TIERS = [(0, 5), (6, 4), (13, 3), (17, 1), (21, 0), (27, -1), (34, -4), (45, -6), (999, -10)]
YA_TIERS = [(99, 5), (199, 3), (299, 2), (349, 0), (399, -1), (449, -3), (499, -5), (549, -6), (9999, -7)]


def _tier(v: float, tiers) -> float:
    for cap, pts in tiers:
        if v <= cap:
            return float(pts)
    return float(tiers[-1][1])


def dst_points(pa: float, ya: float, sacks: float, ints: float, fr: float, tds: float, safeties: float = 0,
               blocks: float = 0) -> float:
    return (sacks + 2 * ints + 2 * fr + 6 * tds + 2 * safeties + 2 * blocks
            + _tier(pa, PA_TIERS) + _tier(ya, YA_TIERS))


def _team_stats(summary: dict) -> Dict[str, dict]:
    out = {}
    for t in (summary.get("boxscore") or {}).get("teams", []) or []:
        abbr = _team((t.get("team") or {}).get("abbreviation"))
        s = {x.get("name"): x.get("displayValue") for x in t.get("statistics", []) or []}
        out[abbr] = s
    return out


# ---------------------------------------------------------------- plays -> per-player events
TARGET_TYPES = ("pass reception", "pass incompletion", "passing touchdown", "pass interception", "interception")
CATCH_TYPES = ("pass reception", "passing touchdown")
RUSH_TYPES = ("rush", "rushing touchdown")


def _ptype(p: dict) -> str:
    return ((p.get("type") or {}).get("text") or "").lower()


def _clock_of(p: dict) -> Tuple[int, float]:
    return int((p.get("period") or {}).get("number") or 0), float((p.get("clock") or {}).get("value") or 0)


def play_events(plays: List[dict], ids: set, team_ids: Dict[str, str]) -> Tuple[Dict[str, List[dict]], Dict[str, dict]]:
    """Events per tracked athlete, plus per-team pass/rush counts. team_ids: espn team id -> abbr."""
    ev: Dict[str, List[dict]] = {}
    teams: Dict[str, dict] = {}
    for p in plays:
        tx = p.get("text") or ""
        if "NO PLAY" in tx.upper():
            continue
        typ = _ptype(p)
        per, clk = _clock_of(p)
        off = team_ids.get(_id((p.get("team") or {}).get("$ref"), "teams") or "", "")
        start = p.get("start") or {}
        rz = (start.get("yardsToEndzone") or 99) <= 20
        T = teams.setdefault(off, {"pass": 0, "rush": 0, "plays": 0})
        is_pass = any(typ.startswith(t) for t in TARGET_TYPES) or typ == "sack"
        is_rush = typ in RUSH_TYPES
        if is_pass:
            T["pass"] += 1
        if is_rush:
            T["rush"] += 1
        if is_pass or is_rush:
            T["plays"] += 1
        seen = set()
        for part in p.get("participants", []) or []:
            aid = _id((part.get("athlete") or {}).get("$ref"), "athletes")
            role = part.get("type")
            if not aid or aid not in ids or (aid, role) in seen:
                continue
            seen.add((aid, role))
            yd = float(p.get("statYardage") or 0)
            e = {"seq": int(p.get("sequenceNumber") or 0), "period": per, "clock": (p.get("clock") or {}).get("displayValue"),
                 "elapsed": _clock_elapsed(per, clk, "in"), "wall": p.get("wallclock"), "text": tx.strip(),
                 "rz": rz, "kind": None, "pts": 0.0, "score": [p.get("awayScore"), p.get("homeScore")]}
            if role == "receiver" and any(typ.startswith(t) for t in TARGET_TYPES):
                catch = typ in CATCH_TYPES
                td = typ == "passing touchdown"
                e.update(kind="catch" if catch else "target", yards=yd if catch else 0, td=td,
                         pts=(1 + 0.1 * yd + (6 if td else 0)) if catch else 0.0)
            elif role == "rusher" and typ in RUSH_TYPES:
                td = typ == "rushing touchdown"
                e.update(kind="carry", yards=yd, td=td, pts=0.1 * yd + (6 if td else 0))
            elif role == "passer" and is_pass:
                td = typ == "passing touchdown"
                comp = typ in CATCH_TYPES
                pick = "interception" in typ
                e.update(kind="pass", yards=yd if comp else 0, td=td,
                         pts=(0.04 * yd if comp else 0) + (4 if td else 0) - (2 if pick else 0))
            elif role in ("fumbler",) and "opponent" in typ:
                e.update(kind="fumble", pts=-2.0)
            else:
                continue
            if re.search(r"injur", tx, re.I):
                e["injury_text"] = True
            ev.setdefault(aid, []).append(e)
    return ev, teams


def injury_mentions(plays: List[dict], short: str) -> List[dict]:
    """Plays whose text says this player was injured ("C.Lamb was injured during the play")."""
    out = []
    for p in plays:
        tx = p.get("text") or ""
        if short and short in tx and re.search(rf"{re.escape(short)}[^.]*injur", tx, re.I):
            per, clk = _clock_of(p)
            out.append({"period": per, "clock": (p.get("clock") or {}).get("displayValue"), "text": tx.strip()})
    return out


def two_points(plays: List[dict], short: str) -> List[dict]:
    """Successful two-point tries naming this player (box scores leave them out)."""
    out = []
    for p in plays:
        tx = p.get("text") or ""
        i = tx.upper().find("TWO-POINT CONVERSION ATTEMPT")
        if i < 0 or not short:
            continue
        seg = tx[i:]
        if "SUCCEEDS" in seg.upper() and short in seg:
            per, clk = _clock_of(p)
            out.append({"seq": int(p.get("sequenceNumber") or 0) + 1, "period": per, "clock": (p.get("clock") or {}).get("displayValue"),
                        "elapsed": _clock_elapsed(per, clk, "in"), "wall": p.get("wallclock"), "text": seg.strip(), "rz": True,
                        "kind": "two_pt", "pts": 2.0, "score": [p.get("awayScore"), p.get("homeScore")]})
    return out


def short_name(name: str) -> str:
    """'CeeDee Lamb' -> 'C.Lamb' (how ESPN play text names players)."""
    parts = [x for x in re.split(r"\s+", name.strip()) if x]
    if len(parts) < 2:
        return name
    last = parts[-1]
    if last.rstrip(".").upper() in ("JR", "SR", "II", "III", "IV", "V") and len(parts) > 2:
        last = parts[-2]
    return f"{parts[0][0]}.{last}"


# ---------------------------------------------------------------- the model
def _q_sampler(ratio_q: Dict[str, List[float]], rng: np.random.Generator):
    def draw(pos: str, n: int) -> np.ndarray:
        q = np.asarray(ratio_q.get(pos) or ratio_q.get("WR"), dtype=float)
        u = rng.random(n) * (len(q) - 1)
        lo = np.floor(u).astype(int)
        return q[lo] + (q[np.minimum(lo + 1, len(q) - 1)] - q[lo]) * (u - lo)
    return draw


def final_draws(p: dict, draw, n: int) -> np.ndarray:
    """Draws of this player's final points given where his game is."""
    proj = float(p.get("proj") or 0)
    pts = float(p.get("pts") or 0)
    st = p["state"]
    if st in ("out", "bye") or (proj <= 0 and st == "pre"):
        return np.full(n, pts)
    if st == "post":
        return np.full(n, pts)
    q = draw(p["pos"], n)
    if st == "pre":
        return proj * q
    r = max(0.0, 1.0 - float(p.get("elapsed") or 0))
    if p["pos"] == "DST" and p.get("dst"):
        return _dst_final(p["dst"], r, n)
    tilt = float(p.get("tilt") or 1.0)
    rem = proj * tilt * np.maximum(0.0, r + math.sqrt(r) * (q - 1.0))
    return pts + rem


# league-average D/ST event rates per team game (approximate, from memory of recent seasons)
DST_RATES = {"sacks": 2.4, "takeaways": 1.3, "td": 0.12, "ya": 330.0, "ya_sd": 70.0, "pa_sd": 10.0}


def _dst_final(dst: dict, r: float, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Final D/ST points: what's banked + simulated rest of game, with the PA/YA tiers applied to the
    simulated final totals (the tiers are about the final score, so 0 allowed early isn't banked)."""
    rng = rng or np.random.default_rng(5)
    pa = dst["pa"] + np.maximum(0, rng.normal(dst["opp_implied"] * r, DST_RATES["pa_sd"] * math.sqrt(r), n))
    ya = dst["ya"] + np.maximum(0, rng.normal(DST_RATES["ya"] * r, DST_RATES["ya_sd"] * math.sqrt(r), n))
    ev = (rng.poisson(DST_RATES["sacks"] * r, n) + 2 * rng.poisson(DST_RATES["takeaways"] * r, n)
          + 6 * (rng.random(n) < DST_RATES["td"] * r))
    tier = np.vectorize(_tier, excluded={1})
    return dst["banked"] + ev + tier(np.round(pa), PA_TIERS) + tier(np.round(ya), YA_TIERS)


def simulate(me: List[dict], opp: List[dict], ratio_q: dict, seed: int = 7, n: int = SIMS) -> dict:
    rng = np.random.default_rng(seed)
    draw = _q_sampler(ratio_q, rng)
    A = np.zeros(n)
    B = np.zeros(n)
    for p in me:
        x = final_draws(p, draw, n)
        p["p10"], p["p50"], p["p90"] = [round(float(v), 1) for v in np.percentile(x, [10, 50, 90])]
        p["live_mean"] = round(float(x.mean()), 1)
        A += x
    for p in opp:
        x = final_draws(p, draw, n)
        p["p10"], p["p50"], p["p90"] = [round(float(v), 1) for v in np.percentile(x, [10, 50, 90])]
        p["live_mean"] = round(float(x.mean()), 1)
        B += x
    return {"p_win": round(float((A > B).mean() + 0.5 * (A == B).mean()), 4),
            # leverage: win chance one more fantasy point buys right now (density of the margin near 0, over ±3)
            "lev": round(float((np.abs(A - B) < 3).mean() / 6), 4) if (A - B).std() > 0.1 else 0.0,  # 0 once nothing is left to play
            "me": {"mean": round(float(A.mean()), 1), "p10": round(float(np.percentile(A, 10)), 1),
                   "p90": round(float(np.percentile(A, 90)), 1)},
            "opp": {"mean": round(float(B.mean()), 1), "p10": round(float(np.percentile(B, 10)), 1),
                    "p90": round(float(np.percentile(B, 90)), 1)}}


def usage_tilt(n_opps: float, per_game: float, elapsed: float) -> float:
    if per_game <= 0 or elapsed <= 0:
        return 1.0
    rate = (PRIOR_GAMES * per_game + n_opps) / (PRIOR_GAMES + elapsed)
    return float(min(TILT_RANGE[1], max(TILT_RANGE[0], rate / per_game)))


# ---------------------------------------------------------------- history (the P(win) line)
_hist: Dict[Tuple[int, int], List[dict]] = {}


# ---------------------------------------------------------------- board
def _pregame(g: Optional[dict]) -> dict:
    g = g or {}
    it, gt = g.get("implied_total"), g.get("game_total")
    return {"implied": it, "total": gt, "opp_implied": (gt - it) if (it is not None and gt is not None) else None}


def _starters(snap: dict) -> Tuple[List[dict], List[dict]]:
    tw = snap["team_week"]
    by = {int(p["id"]): p for p in tw["players"]}
    me = []
    for slot, pid in (tw.get("lineup", {}).get("current") or {}).items():
        if pid is None or int(pid) not in by:
            continue
        p = by[int(pid)]
        me.append({"id": int(pid), "name": p["name"], "pos": p["pos"], "team": _team(p.get("team")), "slot": slot,
                   "proj": float(p.get("espn_week") or 0), "injury": p.get("injury"), "headshot": p.get("headshot"),
                   "side": "me", "pregame": _pregame(p.get("game"))})
    opp = []
    for o in (tw.get("matchup") or {}).get("opp_starters") or []:
        opp.append({"id": int(o["id"]), "name": o["name"], "pos": o["pos"], "team": _team(o.get("team")),
                    "slot": o.get("slot"), "proj": float(o.get("espn_week") or 0), "injury": o.get("injury"),
                    "headshot": o.get("headshot"), "side": "opp", "pregame": _pregame(o.get("game"))})
    order = ["QB", "RB", "RB/WR", "WR", "WR/TE", "TE", "FLEX", "K", "DST"]
    key = lambda p: (order.index(p["slot"]) if p["slot"] in order else 99)
    return sorted(me, key=key), sorted(opp, key=key)


def _per_game(pfile: Optional[dict]) -> Tuple[float, Optional[float]]:
    """Season opportunities (targets + carries) per game played, and season target share."""
    if not pfile:
        return 0.0, None
    gl = [g for g in pfile.get("gamelog") or [] if g.get("played")]
    opps = [float(g.get("tgt") or 0) + float(g.get("car") or 0) for g in gl]
    per = sum(opps) / len(opps) if opps else 0.0
    share = None
    try:
        s = ((pfile.get("role") or {}).get("series") or {}).get("share") or []
        vals = [float(x) for x in s if x is not None]
        share = sum(vals) / len(vals) if vals else None
        if share is not None and share > 1.5:
            share /= 100.0
    except Exception:
        share = None
    return per, share


def board(snap: dict, player_file: Callable[[int], dict] = None) -> dict:
    """Everything the Live screen shows. snap = snapshot.build(team)."""
    games = scoreboard()
    by_team = {}
    for g in games:
        for side in ("home", "away"):
            if g[side].get("abbr"):
                by_team[g[side]["abbr"]] = g
    me, opp = _starters(snap)
    everyone = me + opp
    live_events = {g["event"] for g in games if g["state"] in ("in", "post")}
    summaries, plays, lines, tstats, team_ids = {}, {}, {}, {}, {}
    for ev in live_events:
        g = next(x for x in games if x["event"] == ev)
        try:
            summaries[ev] = _summary(ev, g["state"] == "in")
            plays[ev] = _plays(ev, g["state"] == "in")
        except Exception:
            continue
        lines.update(box_lines(summaries[ev]))
        tstats.update(_team_stats(summaries[ev]))
        for side in ("home", "away"):
            team_ids[g[side]["id"]] = g[side]["abbr"]
    ids = {str(p["id"]) for p in everyone if p["id"] > 0}
    events: Dict[str, List[dict]] = {}
    team_counts: Dict[str, dict] = {}
    for ev, pl in plays.items():
        e, t = play_events(pl, ids, team_ids)
        for k, v in e.items():
            events.setdefault(k, []).extend(v)
        team_counts.update(t)

    insights: List[dict] = []
    for p in everyone:
        g = by_team.get(p["team"])
        p["event"] = g["event"] if g else None
        p["state"] = "bye" if not g else g["state"]
        if (p.get("injury") or "").upper() in ("OUT", "INJURY_RESERVE", "SUSPENSION"):
            p["state"] = "out" if p["state"] == "pre" else p["state"]
        p["elapsed"] = g["elapsed"] if g else 0.0
        p["opp"] = None
        if g:
            p["opp"] = g["away"]["abbr"] if g["home"]["abbr"] == p["team"] else g["home"]["abbr"]
            p["home"] = g["home"]["abbr"] == p["team"]
        p["pts"] = 0.0
        p["line"] = {}
        p["events"] = []
        p["tilt"] = 1.0
        if p["state"] not in ("in", "post"):
            continue
        aid = str(p["id"])
        if p["pos"] == "DST":
            ts_opp = tstats.get(p["opp"], {})
            ts_me = tstats.get(p["team"], {})
            pa = float(g["away"]["score"] if g["home"]["abbr"] == p["team"] else g["home"]["score"])
            ya = _num(ts_opp.get("totalYards"))
            sk = _num(str(ts_opp.get("sacksYardsLost", "0-0")).split("-")[0])
            ints = _num(ts_opp.get("interceptions"))
            fr = _num(ts_opp.get("fumblesLost"))
            tds = _num(ts_me.get("defensiveTouchdowns"))  # often absent; return TDs counted from plays below
            for pl in plays.get(g["event"], []):
                typ = _ptype(pl)
                tx = (pl.get("text") or "").upper()
                team_ab = team_ids.get(_id((pl.get("team") or {}).get("$ref"), "teams") or "", "")
                if pl.get("scoringPlay") and team_ab == p["team"] and any(
                        k in typ for k in ("interception return", "fumble return", "punt return", "kickoff return",
                                           "blocked")):
                    tds += 1
            p["line"] = {"pa": pa, "ya": ya, "sacks": sk, "ints": ints, "fr": fr, "tds": tds}
            p["pts"] = round(dst_points(pa, ya, sk, ints, fr, tds), 1)  # as it stands (what ESPN would show)
            p["dst"] = {"banked": sk + 2 * ints + 2 * fr + 6 * tds, "pa": pa, "ya": ya,
                        "opp_implied": (p.get("pregame") or {}).get("opp_implied") or 22.0}
        elif p["pos"] == "K":
            p["pts"] = kicker_points(plays.get(g["event"], []), short_name(p["name"]))
            p["line"] = {k: lines.get(aid, {}).get(k, 0) for k in ("fgm", "fga", "xpm", "xpa")}
        else:
            L = dict(lines.get(aid, {}))
            tp = two_points(plays.get(g["event"], []), short_name(p["name"]))
            if tp:
                L["two_pt"] = len(tp)
            p["line"] = {k: v for k, v in L.items() if k != "team"}
            p["pts"] = ppr(L)
            p["events"] = sorted(events.get(aid, []) + tp, key=lambda e: e["seq"])
            deep = sum(1 for e in p["events"] if e["kind"] in ("target", "catch") and re.search(r"\bdeep\b", e["text"], re.I))
            if deep:
                p["line"]["deep_tgt"] = deep
            pf = None
            if player_file:
                try:
                    pf = player_file(p["id"])
                except Exception:
                    pf = None
            per, share = _per_game(pf)
            n_opp = float(L.get("tgt", 0) + L.get("car", 0))
            if p["pos"] == "QB":
                n_opp = float(L.get("att", 0) + L.get("car", 0))
                per = per or 0.0
            p["opps"] = {"now": n_opp, "per_game": round(per, 1), "expected_now": round(per * p["elapsed"], 1)}
            T = team_counts.get(p["team"], {})
            if p["pos"] in ("WR", "TE", "RB") and T.get("pass"):
                p["share"] = {"tgt": round(L.get("tgt", 0) / max(1, T["pass"]), 3), "team_pass": T["pass"],
                              "season": round(share, 3) if share else None}
            if p["state"] == "in" and p["pos"] != "QB":
                p["tilt"] = round(usage_tilt(n_opp, per, p["elapsed"]), 2)
            inj = injury_mentions(plays.get(g["event"], []), short_name(p["name"]))
            if inj:
                p["injury_live"] = inj[-1]
        p["expected_now"] = round(p["proj"] * p["elapsed"], 1)

    sim = simulate(me, opp, snap.get("ratio_q") or {})
    pre_me = [dict(x, state="pre" if x["state"] in ("in", "post") else x["state"], pts=0.0) for x in me]
    pre_opp = [dict(x, state="pre" if x["state"] in ("in", "post") else x["state"], pts=0.0) for x in opp]
    pre = simulate(pre_me, pre_opp, snap.get("ratio_q") or {}, n=8000)

    # history: P(win) at every play that changed a tracked player's points, for tonight's games
    series = _series(me, opp, snap.get("ratio_q") or {}, games, plays)

    insights = _insights(me, opp, games, by_team, sim, pre)
    tw = snap["team_week"]
    return {
        "generated_at": time.time(), "season": snap.get("season"), "week": snap.get("week"),
        "me_name": tw.get("team_name"), "opp_name": (tw.get("matchup") or {}).get("opp_name"),
        "games": games, "me": me, "opp": opp,
        "matchup": {"p_win": sim["p_win"], "p_win_pre": pre["p_win"], "lev": sim["lev"], "lev_pre": pre["lev"], "me": sim["me"], "opp": sim["opp"],
                    "me_pts": round(sum(p["pts"] for p in me), 1), "opp_pts": round(sum(p["pts"] for p in opp), 1),
                    "me_left": sum(1 for p in me if p["state"] in ("pre", "in")),
                    "opp_left": sum(1 for p in opp if p["state"] in ("pre", "in"))},
        "series": series, "insights": insights, "notes": NOTES, "errors": list(_errors),
        "poll_s": TTL_LIVE if any(g["state"] == "in" for g in games) else 60,
    }


def _series(me, opp, ratio_q, games, plays) -> List[dict]:
    """P(win) re-simulated through the night: at every play where a tracked starter gained or lost
    points, every score change in a game with a tracked starter (that moves D/ST), and each quarter's
    end (time passing moves it too). D/ST is rebuilt approximately at each step: points allowed from
    the score at that play, yards and banked events pro-rated by game time."""
    live = [p for p in me + opp if p["state"] in ("in", "post")]
    if not live:
        return []
    gstate = {g["event"]: g for g in games}
    marks = []  # (wall, seq, event, elapsed, score[away, home], who, pts, text, period, clock)
    for ev in {p["event"] for p in live}:
        last_score, last_per = None, None
        for pl in plays.get(ev, []):
            per, clk = _clock_of(pl)
            sc = [pl.get("awayScore"), pl.get("homeScore")]
            el = _clock_elapsed(per, clk, "in")
            if last_score is not None and sc != last_score:
                # a tracked kicker's FG or extra point is his own play, not an anonymous score change
                k = next((p for p in live if p["pos"] == "K" and p["event"] == ev and kicker_points([pl], short_name(p["name"]))), None)
                marks.append((pl.get("wallclock") or "", int(pl.get("sequenceNumber") or 0), ev, el, sc, k and k["id"],
                              kicker_points([pl], short_name(k["name"])) if k else 0.0,
                              (pl.get("text") or "").strip(), per, (pl.get("clock") or {}).get("displayValue")))
            if last_per is not None and per != last_per:
                marks.append((pl.get("wallclock") or "", int(pl.get("sequenceNumber") or 0) - 1, ev, _clock_elapsed(last_per, 0, "in"), last_score, None, 0.0,
                              f"End of Q{last_per}", last_per, "0:00"))
            last_score, last_per = sc, per
    for p in live:
        if p["pos"] in ("DST", "K"):
            continue
        for e in p["events"]:
            if e["pts"]:
                marks.append((e.get("wall") or "", e["seq"], p["event"], e["elapsed"], e["score"], p["id"], e["pts"], e["text"], e["period"], e["clock"]))
    marks.sort(key=lambda m: (m[0], m[1]))
    # each game's plays in wall order, so a player in another game can be put back where his game stood
    tl = {}
    for ev, pls in plays.items():
        rows = sorted(((pl.get("wallclock") or "", int(pl.get("sequenceNumber") or 0), pl) for pl in pls if pl.get("wallclock")),
                      key=lambda r: (r[0], r[1]))
        tl[ev] = ([r[0] for r in rows], [r[2] for r in rows])
    out = []
    for wall, seq, ev, el, score, who, dpts, text, per, clk in marks[-120:]:
        def at(p):
            q = dict(p)
            if p["event"] is None or p["state"] not in ("in", "post"):
                return q
            s_, el_, sc_ = seq, el, score
            if p["event"] != ev:
                # another game: put it back to its last play at or before this wall time
                walls, pls = tl.get(p["event"], ([], []))
                if not walls or not wall:
                    return q
                i = bisect.bisect_right(walls, wall) - 1
                if i < 0:
                    q.update(state="pre", pts=0.0)
                    return q
                if i == len(walls) - 1 and p["state"] == "post":
                    return q
                pl = pls[i]
                per_, clk_ = _clock_of(pl)
                s_, el_, sc_ = int(pl.get("sequenceNumber") or 0), _clock_elapsed(per_, clk_, "in"), [pl.get("awayScore"), pl.get("homeScore")]
            if p["pos"] == "K":
                pts = kicker_points([x for x in plays.get(p["event"], []) if int(x.get("sequenceNumber") or 0) <= s_], short_name(p["name"]))
            else:
                pts = sum(e["pts"] for e in p["events"] if e["seq"] <= s_)
            q.update(state="in", elapsed=el_, pts=pts, tilt=1.0)
            if p["pos"] == "DST" and p.get("dst"):
                g = gstate.get(p["event"])
                home = g and g["home"]["abbr"] == p["team"]
                pa_then = float((sc_[0] if home else sc_[1]) or 0)
                frac = el_ / max(1e-6, p["elapsed"] or 1.0)
                q["dst"] = dict(p["dst"], pa=pa_then, ya=p["dst"]["ya"] * min(1.0, frac),
                                banked=p["dst"]["banked"] * min(1.0, frac))
            return q
        sim = simulate([at(p) for p in me], [at(p) for p in opp], ratio_q, n=3000, seed=11)
        out.append({"wall": wall, "elapsed": el, "event": ev, "p_win": sim["p_win"], "lev": sim["lev"], "who": who,
                    "pts": round(dpts, 1), "text": text[:140], "period": per, "clock": clk})
    return out


def _insights(me, opp, games, by_team, sim, pre) -> List[dict]:
    out: List[dict] = []
    d = sim["p_win"] - pre["p_win"]
    if any(p["state"] in ("in", "post") for p in me + opp):
        out.append({"tone": "up" if d > 0.02 else "down" if d < -0.02 else "flat", "kind": "matchup",
                    "text": f"Win chance {sim['p_win']:.0%} (it was {pre['p_win']:.0%} before kickoff)."})
    for p in me + opp:
        mine = p["side"] == "me"
        if p.get("injury_live"):
            out.append({"tone": "alert", "kind": "injury", "id": p["id"],
                        "text": f"{p['name']}: injury mention at {p['injury_live']['clock']} Q{p['injury_live']['period']}. "
                                f"“{p['injury_live']['text'][:110]}”"})
        if p["state"] != "in" or p["pos"] in ("DST", "K"):
            continue
        o = p.get("opps") or {}
        sh = p.get("share") or {}
        if o.get("per_game") and p["elapsed"] >= 0.12:
            exp = o["expected_now"]
            if o["now"] >= exp + 2.5:
                out.append({"tone": "up" if mine else "down", "kind": "usage", "id": p["id"],
                            "text": f"{p['name']} has {o['now']:.0f} opportunities, about {exp:.0f} is his usual by now."})
            elif o["now"] <= exp - 2.5:
                out.append({"tone": "down" if mine else "up", "kind": "usage", "id": p["id"],
                            "text": f"{p['name']} has {o['now']:.0f} opportunities, about {exp:.0f} is his usual by now."})
        if sh.get("season") and sh.get("team_pass", 0) >= 8:
            diff = sh["tgt"] - sh["season"]
            if abs(diff) >= 0.08:
                out.append({"tone": ("up" if diff > 0 else "down") if mine else ("down" if diff > 0 else "up"),
                            "kind": "share", "id": p["id"],
                            "text": f"{p['name']}: {sh['tgt']:.0%} of {p['team']}'s {sh['team_pass']} pass plays tonight, "
                                    f"{sh['season']:.0%} on the season."})
        if p.get("expected_now") is not None and p["elapsed"] >= 0.25 and p["pos"] != "QB":
            ahead = p["pts"] - p["expected_now"]
            if ahead >= 8:
                out.append({"tone": "flat", "kind": "hot", "id": p["id"],
                            "text": f"{p['name']} is {ahead:.0f} points ahead of pace. In 2025 data a hot start didn't carry on by itself; "
                                    f"the rest of the game tracked his usage, which is why the 'headed for' number moves less than the score."})
        rz = [e for e in p["events"] if e["rz"] and e["kind"] in ("target", "catch", "carry")]
        if len(rz) >= 2:
            out.append({"tone": "up" if mine else "down", "kind": "redzone", "id": p["id"],
                        "text": f"{p['name']} has {len(rz)} looks inside the 20."})
        # quiet stretch: no involvement in his team's last 10 plays
        g = by_team.get(p["team"])
        if p["events"] and g and p["elapsed"] > 0.2:
            last = max(e["elapsed"] for e in p["events"])
            if p["elapsed"] - last >= 0.17:
                out.append({"tone": "down" if mine else "up", "kind": "quiet", "id": p["id"],
                            "text": f"{p['name']} hasn't been involved since Q{p['events'][-1]['period']} {p['events'][-1]['clock']}."})
        # script
        if g and p["elapsed"] >= 0.5:
            mine_score = g["home"]["score"] if g["home"]["abbr"] == p["team"] else g["away"]["score"]
            their = g["away"]["score"] if g["home"]["abbr"] == p["team"] else g["home"]["score"]
            if mine_score - their <= -10 and p["pos"] in ("WR", "TE", "QB"):
                out.append({"tone": "up" if mine else "down", "kind": "script", "id": p["id"],
                            "text": f"{p['team']} trails by {their - mine_score} in the second half: teams in that spot pass more, which leans toward more for {p['name'].split()[-1]}."})
            if mine_score - their >= 14 and p["pos"] in ("WR", "TE", "QB"):
                out.append({"tone": "down" if mine else "up", "kind": "script", "id": p["id"],
                            "text": f"{p['team']} leads by {mine_score - their}: teams protecting a lead run more, which leans against {p['name'].split()[-1]}."})
    rank = {"alert": 0, "matchup": 1}
    out.sort(key=lambda x: (rank.get(x["tone"], 2) if x["kind"] != "matchup" else 1))
    return out
