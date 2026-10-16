"""Single .env loader for the whole repo.

Import-order matters: ``config.py`` reads env vars (``PROJECTIONS_CSV``) at
import time and is imported before ``espn_client.py``, so the loader must live
somewhere both can call first. This is that place -- add new secrets
(e.g. ``FANTASYPROS_API_KEY``) to ``.env.local`` and they are picked up here.

Real environment variables always win over file values; ``.env.local`` wins
over ``.env``. Idempotent: safe to call from every module that needs it.
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_DIR = Path(__file__).resolve().parent
_loaded = False


def load_local_env() -> None:
    global _loaded
    if _loaded:
        return
    for filename in (".env.local", ".env"):
        path = _ENV_DIR / filename
        if not path.exists():
            continue
        try:
            for raw_line in path.read_text().splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export "):]
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        except OSError:
            continue
    _loaded = True
