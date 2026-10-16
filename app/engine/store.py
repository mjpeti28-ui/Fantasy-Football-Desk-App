"""Desk's local cache: computed responses and Desk-built data, kept out of iCloud.

The repo sits in an iCloud-synced Desktop with Optimize Mac Storage on, which evicts files that
haven't been read lately (they turn into placeholders that re-download on the next read, or fail
to read at all from a VM). Everything Desk can regenerate lives in app/cache.nosync/: iCloud
Drive never uploads or evicts names ending in .nosync, and the folder also gets the File
Provider "ignore" attribute that newer macOS versions check.

Responses are keyed by everything they depend on (input file times, week, code version), so a
cache hit is the same answer the engine would compute, not a stale guess.
"""
from __future__ import annotations

import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "app" / "cache.nosync"
# Bump when a cached computation changes shape or method.
VERSION = 3
_marked = False


def local_dir(*parts: str) -> Path:
    """A folder under app/cache.nosync/ (created and marked on first use)."""
    global _marked
    if not _marked:
        DIR.mkdir(parents=True, exist_ok=True)
        if sys.platform == "darwin":
            try:
                subprocess.run(["xattr", "-w", "com.apple.fileprovider.ignore#P", "1", str(DIR)],
                               capture_output=True, timeout=5)
            except Exception:
                pass
        _marked = True
    p = DIR.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get(name: str, key: Any) -> Optional[Tuple[Any, float]]:
    """(value, saved_at) if a response with exactly this key is on disk."""
    p = DIR / "responses" / f"{name}.pkl"
    try:
        with open(p, "rb") as f:
            obj = pickle.load(f)
        if obj.get("key") == repr((VERSION, key)):
            return obj["value"], obj["at"]
    except Exception:
        pass
    return None


def put(name: str, key: Any, value: Any) -> None:
    p = local_dir("responses") / f"{name}.pkl"
    tmp = p.with_suffix(".tmpw")
    try:
        with open(tmp, "wb") as f:
            pickle.dump({"key": repr((VERSION, key)), "value": value, "at": time.time()}, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, p)
    except Exception:
        pass


def cached(name: str, key: Any, fn: Callable[[], Any]) -> Any:
    hit = get(name, key)
    if hit is not None:
        return hit[0]
    v = fn()
    put(name, key, v)
    return v
