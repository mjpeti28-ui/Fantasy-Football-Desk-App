"""Cached GET against the PFF Developer API (https://developer.pff.com/openapi.json).

Every response (any status) is cached raw under .cache/pff/raw/ in the main checkout (gitignored) and a
cached request is never re-sent. The key comes from PFF_API_KEY or .env.local and is never printed.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

MAIN = Path.home() / "Developer" / "Fantasy-Football-Desk-App"
CACHE = MAIN / ".cache" / "pff" / "raw"
BASE = "https://api.pff.com"
MIN_GAP = 1.0
_last = [0.0]


def _key() -> str:
    k = os.environ.get("PFF_API_KEY")
    if not k:
        for line in (MAIN / ".env.local").read_text().splitlines():
            if line.strip().startswith("PFF_API_KEY="):
                k = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not k:
        raise LookupError("PFF_API_KEY not set")
    return k


def cache_file(path: str, params: dict) -> Path:
    q = urllib.parse.urlencode(sorted((k, str(v)) for k, v in params.items()))
    h = hashlib.sha1(f"{path}?{q}".encode()).hexdigest()[:12]
    return CACHE / (path.strip("/").replace("/", "_").replace("{", "").replace("}", "") + f"__{h}.json")


def get(path: str, params: dict | None = None) -> dict:
    """Returns {"status", "path", "params", "ratelimit", "body"}; body is parsed JSON or text."""
    params = params or {}
    p = cache_file(path, params)
    if p.exists():
        return json.loads(p.read_text())
    wait = MIN_GAP - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    url = BASE + path + ("?" + urllib.parse.urlencode(params) if params else "")
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {_key()}", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            status, hdr, raw = r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        status, hdr, raw = e.code, e.headers, e.read()
    _last[0] = time.time()
    rl = {k: hdr.get(k) for k in ("x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset", "retry-after")}
    try:
        body = json.loads(raw)
    except ValueError:
        body = raw.decode("utf-8", "replace")
    out = {"status": status, "path": path, "params": params, "ratelimit": rl, "body": body}
    if status == 429:
        raise RuntimeError(f"429 rate limited: {rl}")
    if status < 500:  # don't cache transient failures
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out))
    rem = rl.get("x-ratelimit-remaining")
    if rem is not None and int(rem) <= 2:
        time.sleep(max(1.0, float(rl.get("x-ratelimit-reset") or 0) - time.time() + 1))
    return out
