"""Pilot crawl of public Sleeper leagues for power-ranking backtests.

Snowball BFS: seed users -> their leagues (2021-2025) -> league members -> their
leagues. Every GET is cached gzipped on disk, so reruns cost no API calls.

    python backtest/sleeper_crawl.py crawl            # discover + fetch, then report
    python backtest/sleeper_crawl.py report           # report from cache only
"""
import argparse
import collections
import gzip
import hashlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DATA = Path(os.environ.get("SLEEPER_DATA", Path.home() / "Developer" / "sleeper-data"))
CACHE = DATA / "raw"
BASE = "https://api.sleeper.app/v1"
SEASONS = range(2021, 2026)
WEEKS = range(1, 18)
MIN_INTERVAL = 0.25  # ~4 req/s

# Diversified seeds: fantasy content creators/sites (each anchors a different cluster)
# plus league IDs cited in public Sleeper API docs/guides.
SEED_USERS = ["matthewberry", "adamlevitan", "fantasypros", "rotounderworld", "sigmund", "sleeper"]
SEED_LEAGUES = ["289646328504385536", "378845311639904256"]
MAX_KEPT_PER_USER = 3  # keep a single power user's leagues from dominating

STATS = {"net_calls": 0, "cache_hits": 0, "errors": 0, "best_ball_skipped": 0, "no_matchups": 0}
FILTER = False
_last = [0.0]


def _cache_path(url):
    key = url.split("://", 1)[1]
    safe = key.replace("/", "_").replace("?", "_").replace("=", "-")
    if len(safe) > 150:
        safe = safe[:100] + hashlib.sha1(key.encode()).hexdigest()
    return CACHE / (safe + ".json.gz")


def get(url, offline=False):
    """GET JSON with disk cache, throttle, and retry. Returns parsed JSON or None."""
    p = _cache_path(url)
    if p.exists():
        STATS["cache_hits"] += 1
        with gzip.open(p, "rt") as f:
            return json.load(f)["body"]
    if offline:
        return None
    for attempt in range(5):
        wait = MIN_INTERVAL - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        STATS["net_calls"] += 1
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "desk-backtest-pilot"})
            with urllib.request.urlopen(req, timeout=20) as r:
                status, body = r.status, json.loads(r.read() or b"null")
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            status, body = e.code, None
            break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(2 ** attempt)
    else:
        STATS["errors"] += 1
        return None
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wt") as f:
        json.dump({"url": url, "status": status, "body": body}, f)
    return body


def league_meta(lg):
    s = lg.get("settings") or {}
    return {
        "league_id": lg["league_id"], "season": int(lg["season"]), "name": lg.get("name"),
        "type": s.get("type", 0), "teams": lg.get("total_rosters"),
        "rec": (lg.get("scoring_settings") or {}).get("rec"),
        "playoff_week_start": s.get("playoff_week_start"), "playoff_teams": s.get("playoff_teams"),
        "playoff_type": s.get("playoff_type"), "previous_league_id": lg.get("previous_league_id"),
    }


def fetch_league(lid):
    base = f"{BASE}/league/{lid}"
    users = get(f"{base}/users") or []
    members = [u["user_id"] for u in users if u.get("user_id")]
    if not get(f"{base}/matchups/1"):  # ~18% of leagues have matchup data purged; skip the rest
        STATS["no_matchups"] += 1
        return members, False
    get(f"{base}/rosters")
    get(f"{base}/drafts")
    for w in WEEKS:
        get(f"{base}/matchups/{w}")
        get(f"{base}/transactions/{w}")
    return members, True


def backtest_ok(lg):
    """Cheap pre-filter from the league object: managed redraft/keeper/dynasty, 8-14 teams, PPR/half."""
    s = lg.get("settings") or {}
    return (s.get("type", 0) in (0, 1, 2) and 8 <= (lg.get("total_rosters") or 0) <= 14
            and (lg.get("scoring_settings") or {}).get("rec") in (0.5, 1))


def crawl(target, clean_target=None, max_hours=None):
    t0 = last_log = time.time()
    clean = 0
    rng = random.Random(7)
    kept = {}  # league_id -> meta
    seen_leagues, seen_users = set(), set()
    queue = collections.deque()
    for name in SEED_USERS:
        u = get(f"{BASE}/user/{name}")
        if u and u.get("user_id"):
            queue.append(u["user_id"])
    seed_lgs = [get(f"{BASE}/league/{lid}") for lid in SEED_LEAGUES]
    for lg in seed_lgs:
        if lg and lg.get("league_id"):
            for uid in fetch_league(lg["league_id"])[0]:  # pull members as seeds
                queue.append(uid)
    print(f"seed users in queue: {len(queue)}", flush=True)

    def log(done=False):
        (DATA / "kept_leagues.json").write_text(json.dumps(list(kept.values()), indent=1))
        with open(DATA / "crawl.log", "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} kept={len(kept)} clean={clean} "
                    f"calls={STATS['net_calls']} elapsed={(time.time() - t0) / 3600:.2f}h{' DONE' if done else ''}\n")

    def more():
        if clean_target:
            return clean < clean_target and (time.time() - t0) / 3600 < (max_hours or 1e9)
        return len(kept) < target

    while queue and more():
        uid = queue.popleft()
        if uid in seen_users:
            continue
        seen_users.add(uid)
        cands = []
        for season in SEASONS:
            for lg in get(f"{BASE}/user/{uid}/leagues/nfl/{season}") or []:
                if lg["league_id"] in seen_leagues:
                    continue
                seen_leagues.add(lg["league_id"])
                if lg.get("status") != "complete" or lg.get("sport") != "nfl":
                    continue
                if (lg.get("settings") or {}).get("best_ball"):  # no lineups/waivers to manage
                    STATS["best_ball_skipped"] += 1
                    continue
                if FILTER and not backtest_ok(lg):
                    continue
                cands.append(lg)
        rng.shuffle(cands)
        for lg in cands[:MAX_KEPT_PER_USER]:
            if not more():
                break
            members, ok = fetch_league(lg["league_id"])
            if ok:
                meta = league_meta(lg)
                meta["owners"] = {r["roster_id"]: [r.get("owner_id")] + (r.get("co_owners") or [])
                                  for r in get(f"{BASE}/league/{lg['league_id']}/rosters") or []}
                kept[lg["league_id"]] = meta
                q = analyze_league(meta)
                clean += q["complete"] and q["inactive_strict"] == 0
            if time.time() - last_log > 300:
                log()
                last_log = time.time()
            rng.shuffle(members)
            queue.extend(m for m in members if m not in seen_users)
            print(f"[{len(kept)}/{target}] {lg['season']} {lg.get('name')!r} "
                  f"calls={STATS['net_calls']} t={time.time() - t0:.0f}s", flush=True)

    runtime = time.time() - t0
    log(done=True)
    runlog = DATA / "runs.json"
    runs = json.loads(runlog.read_text()) if runlog.exists() else []
    runs = [r for r in runs if r["target"] == (clean_target or target)]  # report counts this target's runs only
    runs.append({"target": clean_target or target, "kept": len(kept), "runtime_s": round(runtime, 1),
                 "users_expanded": len(seen_users), "leagues_seen": len(seen_leagues), **STATS})
    runlog.write_text(json.dumps(runs, indent=1))


def _scoring(rec):
    return {1: "PPR", 0.5: "half", 0: "standard", None: "standard"}.get(rec, f"other({rec})")


def analyze_league(m):
    """Per-league quality flags from cached matchups/transactions."""
    base = f"{BASE}/league/{m['league_id']}"
    reg_weeks = (m["playoff_week_start"] or 15) - 1
    n = m["teams"]
    complete, has_pp = True, False
    empty_wk, zero_wk = collections.Counter(), collections.Counter()
    late = set(range(max(1, reg_weeks - 3), reg_weeks + 1))
    for w in range(1, reg_weeks + 1):
        mu = get(f"{base}/matchups/{w}", offline=True) or []
        ok = len(mu) == n and all(e.get("starters_points") is not None and e.get("players_points") is not None
                                  for e in mu) and any((e.get("points") or 0) > 0 for e in mu)
        complete &= ok
        has_pp |= any(e.get("players_points") for e in mu)
        if w in late:
            for e in mu:
                st = e.get("starters") or []
                pts = e.get("starters_points") or []
                if any(s in ("0", None, "") for s in st):
                    empty_wk[e["roster_id"]] += 1
                elif any(p == 0 for s, p in zip(st, pts) if s not in ("0", None, "")):
                    zero_wk[e["roster_id"]] += 1
    active_after4 = set()
    for w in range(5, 18):
        for t in get(f"{base}/transactions/{w}", offline=True) or []:
            if t.get("status") == "complete":
                active_after4.update(t.get("roster_ids") or [])
    rosters = get(f"{base}/rosters", offline=True) or []
    rids = [r["roster_id"] for r in rosters]
    empty2 = {r for r in rids if empty_wk[r] >= 2}
    zero2 = {r for r in rids if empty_wk[r] + zero_wk[r] >= 2}
    no_tx = {r for r in rids if r not in active_after4}
    return {"complete": complete, "players_points": has_pp, "teams_n": len(rids),
            "empty2": len(empty2), "zero2": len(zero2), "no_tx": len(no_tx),
            "inactive_strict": len(empty2 | no_tx), "inactive_loose": len(zero2 | no_tx)}


def probe_projections():
    out = []
    for kind in ("projections", "stats"):
        url = f"https://api.sleeper.app/{kind}/nfl/2023/5?season_type=regular"
        body = get(url)
        p = _cache_path(url)
        status = json.load(gzip.open(p, "rt"))["status"] if p.exists() else "error"
        rows = body if isinstance(body, list) else list((body or {}).values())
        sample = rows[0] if rows else {}
        keys = sorted(sample.keys()) if isinstance(sample, dict) else []
        stat_keys = sorted((sample.get("stats") or {}).keys())[:15] if isinstance(sample, dict) else []
        out.append(f"- `{url}` -> status {status}, {len(rows)} rows; sample keys: {keys}; "
                   f"sample stats keys (first 15): {stat_keys}")
    return out


def report():
    kept = json.loads((DATA / "kept_leagues.json").read_text())
    runs = json.loads((DATA / "runs.json").read_text())
    calls = sum(r["net_calls"] for r in runs)
    runtime = sum(r["runtime_s"] for r in runs)
    rows = [(m, analyze_league(m)) for m in kept]
    L = [f"# Sleeper pilot crawl report\n", f"League-seasons kept (status complete, best ball excluded): "
         f"**{len(kept)}**; complete best-ball league-seasons skipped: {runs[-1].get('best_ball_skipped', 0)}\n"]

    def table(title, counter):
        L.append(f"## {title}\n\n| value | n |\n|---|---|")
        L.extend(f"| {k} | {v} |" for k, v in sorted(counter.items(), key=lambda kv: str(kv[0])))
        L.append("")

    table("By season", collections.Counter(m["season"] for m in kept))
    table("By type (0 redraft, 1 keeper, 2 dynasty)", collections.Counter(m["type"] for m in kept))
    table("Season x type", collections.Counter(f"{m['season']}/{m['type']}" for m in kept))
    table("Team count", collections.Counter(m["teams"] for m in kept))
    table("Scoring (rec)", collections.Counter(_scoring(m["rec"]) for m in kept))
    table("Playoff teams", collections.Counter(m["playoff_teams"] for m in kept))
    table("Playoff week start", collections.Counter(m["playoff_week_start"] for m in kept))

    n = len(rows) or 1
    teams = sum(a["teams_n"] for _, a in rows) or 1
    L.append("## Data quality\n")
    L.append(f"- Complete matchups (every regular-season week, all rosters, starters_points + players_points): "
             f"{sum(a['complete'] for _, a in rows)}/{len(rows)} ({sum(a['complete'] for _, a in rows) / n:.0%})")
    L.append(f"- Leagues with non-empty players_points: {sum(a['players_points'] for _, a in rows)}/{len(rows)}")
    L.append(f"- Inactive-manager rate, team level (of {teams} teams; late = last 4 regular weeks):")
    for k, lab in [("empty2", "empty starter slot in 2+ late weeks"),
                   ("zero2", "empty slot or a 0-pt starter in 2+ late weeks"),
                   ("no_tx", "no completed transaction after week 4"),
                   ("inactive_strict", "UNION empty-slot OR no-tx"),
                   ("inactive_loose", "UNION empty/0-pt OR no-tx")]:
        v = sum(a[k] for _, a in rows)
        L.append(f"  - {lab}: {v} ({v / teams:.1%})")
    clean = sum(1 for _, a in rows if a["complete"] and a["inactive_strict"] == 0)
    L.append(f"- League-seasons complete AND zero strict-inactive teams: {clean}/{len(rows)}\n")

    per_league = calls / n
    L.append("## Cost\n")
    L.append(f"- Network API calls: {calls} ({per_league:.0f} per kept league-season incl. discovery); "
             f"runtime {runtime / 60:.1f} min over {len(runs)} run(s); errors {sum(r['errors'] for r in runs)}")
    est = per_league * 5000
    L.append(f"- Scale to 5,000 league-seasons: ~{est:,.0f} calls, ~{est * MIN_INTERVAL / 3600:.1f} h at 4 req/s "
             f"(~{est * 0.0005:.0f} MB raw gz at ~0.5KB/call, rough)\n")
    L.append("## Projections/stats history probe\n")
    L.extend(probe_projections())
    text = "\n".join(L) + "\n"
    (DATA / "pilot_report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["crawl", "report"])
    ap.add_argument("--target", type=int, default=200)
    ap.add_argument("--clean-target", type=int, help="stop at N clean league-seasons (overrides --target)")
    ap.add_argument("--max-hours", type=float)
    ap.add_argument("--backtest-only", action="store_true", help="pre-filter leagues to backtest-usable ones")
    a = ap.parse_args()
    FILTER = a.backtest_only
    if a.cmd == "crawl":
        crawl(a.target, a.clean_target, a.max_hours)
    report()
