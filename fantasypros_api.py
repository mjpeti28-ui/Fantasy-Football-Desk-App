"""Minimal client for the FantasyPros public API v2 (free tier).

Free-tier facts, verified against the live API on 2026-09-22 (not just the spec):

- Every response is capped at 10 players (``"limit": 10, "public_api_limited": true``).
  ``limit``/``page``/``offset`` params are ignored -- there is no paging.
- Filtering by FantasyPros player IDs works (``players=id1:id2:...``), still
  10 per call, so a full board costs ceil(N/10) calls.
- Terms of use: 1 call/second, 100 calls/day, personal non-commercial use,
  cache responses rather than re-polling, and *no* historical player stats
  (don't use ``/player-points`` for past weeks). Published work built on this
  data must say it uses FantasyPros data.

This client enforces the rate limit and keeps a per-day call ledger on disk so
separate runs can't blow through the daily quota together. Raw responses are
cached per day under ``.cache/fantasypros/`` (gitignored).
"""

from __future__ import annotations

import json
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import httpx

from env import load_local_env

BASE_URL = "https://api.fantasypros.com/public/v2/json"
BATCH_SIZE = 10            # free-tier response cap
DAILY_LIMIT = 100          # terms of use
DAILY_SAFETY_MARGIN = 10   # never plan to use the last 10 calls
MIN_INTERVAL_S = 1.1       # terms: 1 call/second

CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "fantasypros"


class QuotaExceeded(RuntimeError):
    pass


class FantasyProsClient:
    def __init__(self, api_key: Optional[str] = None, cache_dir: Path = CACHE_DIR, today: Optional[date] = None):
        load_local_env()
        self.api_key = api_key or os.getenv("FANTASYPROS_API_KEY")
        if not self.api_key:
            raise RuntimeError("FANTASYPROS_API_KEY is not set (add it to .env.local).")
        # UTC so the Mac and the Linux VM (which share this folder) agree on the ledger day.
        self.today = today or datetime.now(timezone.utc).date()
        self.cache_dir = cache_dir / self.today.isoformat()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._ledger_path = cache_dir / f"calls-{self.today.isoformat()}.json"
        self._last_call = 0.0

    # ---------------------------------------------------------------- quota

    def calls_used_today(self) -> int:
        try:
            return int(json.loads(self._ledger_path.read_text()).get("calls", 0))
        except (OSError, ValueError):
            return 0

    def calls_remaining_today(self) -> int:
        return max(0, DAILY_LIMIT - DAILY_SAFETY_MARGIN - self.calls_used_today())

    def _record_call(self) -> None:
        used = self.calls_used_today() + 1
        self._ledger_path.write_text(json.dumps({"date": self.today.isoformat(), "calls": used}))

    # ---------------------------------------------------------------- http

    def get(self, path: str, params: Dict[str, Any], cache_key: Optional[str] = None) -> Dict[str, Any]:
        """GET with daily cache, rate limiting and quota accounting."""
        cache_file = self.cache_dir / f"{cache_key}.json" if cache_key else None
        if cache_file and cache_file.exists():
            return json.loads(cache_file.read_text())
        if self.calls_remaining_today() <= 0:
            raise QuotaExceeded(
                f"FantasyPros daily budget spent ({self.calls_used_today()} calls today; "
                f"limit {DAILY_LIMIT}, keeping {DAILY_SAFETY_MARGIN} in reserve)."
            )
        wait = MIN_INTERVAL_S - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        resp = httpx.get(
            f"{BASE_URL}{path}",
            headers={"x-api-key": self.api_key},
            params=params,
            timeout=30,
        )
        self._last_call = time.monotonic()
        self._record_call()
        resp.raise_for_status()
        payload = resp.json()
        if cache_file:
            cache_file.write_text(json.dumps(payload))
        return payload

    # ---------------------------------------------------------------- endpoints

    def ros_projections(self, season: int, position: str, player_ids: Iterable[int]) -> List[Dict[str, Any]]:
        """Rest-of-season projections for specific players, batched 10 per call."""
        want = {int(i) for i in player_ids}
        # reuse any player already fetched today, whatever batch he came in: a re-run after the
        # rankings or rosters shift would otherwise re-buy every batch whose membership changed
        have: Dict[int, Dict[str, Any]] = {}
        for f in self.cache_dir.glob(f"proj-ros-{season}-{position}-*.json"):
            try:
                for p in json.loads(f.read_text()).get("players", []):
                    if p.get("fpid") is not None:
                        have[int(p["fpid"])] = p
            except (OSError, ValueError):
                continue
        players: List[Dict[str, Any]] = [have[i] for i in sorted(want & have.keys())]
        ids = sorted(want - have.keys())
        for start in range(0, len(ids), BATCH_SIZE):
            batch = ids[start:start + BATCH_SIZE]
            payload = self.get(
                f"/nfl/{season}/projections",
                {"position": position, "ros": "true", "players": ":".join(str(i) for i in batch)},
                cache_key=f"proj-ros-{season}-{position}-{batch[0]}-{batch[-1]}-{len(batch)}",
            )
            if str(payload.get("ros_projections", "true")).lower() not in ("true", "1"):
                raise RuntimeError(f"{position}: API did not return ROS projections: {payload.get('ros_projections')!r}")
            players.extend(payload.get("players", []))
        return players

    def calls_needed(self, n_players_by_position: Dict[str, int]) -> int:
        return sum(-(-n // BATCH_SIZE) for n in n_players_by_position.values() if n)


def ppr_points(player: Dict[str, Any]) -> Optional[float]:
    """``stats`` is an object in live responses (the spec says array); accept both."""
    stats = player.get("stats")
    if isinstance(stats, list):
        stats = stats[0] if stats else {}
    if not isinstance(stats, dict):
        return None
    value = stats.get("points_ppr")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
