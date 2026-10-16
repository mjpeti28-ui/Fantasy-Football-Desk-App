"""Fetch PFF Premium Stats league-wide leaderboards (PFF Developer API) into .cache/pff/.

Auth is a bearer API key (``pffApiKey`` in https://developer.pff.com/openapi.json),
read from PFF_API_KEY in the environment or the repo's .env.local. The key is never
printed or written anywhere. Responses are cached raw under .cache/pff/ (gitignored --
PFF data must not be committed) and a cached request is never re-sent.

``week`` takes a comma list (``1,2,3``); the facet then returns one row per player
aggregated over those weeks, which is how season-to-date / last-3 features are built
without leaking later games.

    python tools/fetch_pff_stats.py --whoami
    python tools/fetch_pff_stats.py --season 2024 --facet receiving/summary --weeks 1-2
    python tools/fetch_pff_stats.py --backtest 2021-2025     # all checkpoint windows
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache" / "pff"
BASE = "https://api.pff.com"
FACETS = ["receiving/summary", "rushing/summary", "passing/summary"]
MIN_GAP = 1.0  # seconds between live requests


class RateLimited(RuntimeError):
    pass


def _key() -> str:
    k = os.environ.get("PFF_API_KEY")
    if not k:
        for p in (ROOT / ".env.local", Path.home() / "Developer" / "Fantasy-Football-Desk-App" / ".env.local"):
            if p.exists():
                for line in p.read_text().splitlines():
                    if line.strip().startswith("PFF_API_KEY="):
                        k = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
            if k:
                break
    if not k:
        sys.exit("PFF_API_KEY not found in environment or .env.local")
    return k


_last = [0.0]


def _get(path: str, params: dict) -> requests.Response:
    wait = MIN_GAP - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    r = requests.get(BASE + path, params=params, headers={"Authorization": f"Bearer {_key()}"}, timeout=60)
    _last[0] = time.time()
    rem = r.headers.get("x-ratelimit-remaining")
    if r.status_code == 429:
        raise RateLimited(f"429 {r.headers.get('Retry-After')}s: {r.text[:300]}")
    if rem is not None and int(rem) <= 2:
        raise RateLimited(f"rate budget nearly spent (remaining={rem}, reset={r.headers.get('x-ratelimit-reset')})")
    return r


def weeks_str(lo: int, hi: int) -> str:
    return ",".join(str(w) for w in range(lo, hi + 1))


def cache_path(facet: str, season: int, lo: int, hi: int) -> Path:
    return CACHE / facet.replace("/", "_") / str(season) / f"w{lo:02d}-{hi:02d}.json"


def fetch(facet: str, season: int, lo: int, hi: int) -> dict:
    p = cache_path(facet, season, lo, hi)
    if p.exists():
        return json.loads(p.read_text())
    r = _get(f"/v1/facet/{facet}", {"league": "nfl", "season": str(season), "week": weeks_str(lo, hi)})
    if r.status_code != 200:
        raise RuntimeError(f"{facet} {season} w{lo}-{hi}: HTTP {r.status_code} {r.text[:300]}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(r.text)
    rem = r.headers.get("x-ratelimit-remaining")
    print(f"  {facet} {season} w{lo}-{hi}: {len(r.content)//1024}KB remaining={rem}", flush=True)
    return r.json()


def windows(W: int):
    """Feature windows for checkpoint W: season-to-date (1..W-1) and last 3 (W-3..W-1)."""
    yield 1, W - 1
    if W - 3 > 1:
        yield W - 3, W - 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--whoami", action="store_true")
    ap.add_argument("--season", type=int)
    ap.add_argument("--facet")
    ap.add_argument("--weeks", help="lo-hi, e.g. 1-2")
    ap.add_argument("--backtest", help="season range, e.g. 2021-2025: every checkpoint window W=3..13")
    a = ap.parse_args()
    if a.whoami:
        r = _get("/v1/auth/whoami", {})
        print(r.status_code, {k: v for k, v in r.headers.items() if k.lower().startswith("x-ratelimit")})
        return
    try:
        if a.backtest:
            lo, hi = (int(s) for s in a.backtest.split("-"))
            for y in range(lo, hi + 1):
                for facet in FACETS:
                    for W in range(3, 14):
                        for wl, wh in windows(W):
                            fetch(facet, y, wl, wh)
        else:
            wl, wh = (int(s) for s in a.weeks.split("-"))
            d = fetch(a.facet, a.season, wl, wh)
            rows = next(iter(d.values())) if isinstance(d, dict) else d
            print(f"{len(rows)} rows; columns: {sorted(rows[0])[:80] if rows else []}")
    except RateLimited as e:
        sys.exit(f"stopped: {e}")


if __name__ == "__main__":
    main()
