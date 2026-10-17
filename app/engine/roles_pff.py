"""PFF alignment for the WR/TE role styles: slot, inline and route rates per player, season to date.

The PFF Developer API (bearer key PFF_API_KEY from the environment or the repo's .env.local; never
printed or written) serves receiving/summary aggregated over a list of weeks. One call covers a whole
season to date. Raw responses are cached under .cache/pff/ (gitignored: PFF data is never committed),
in the same layout tools/fetch_pff_stats.py uses for the 2021-25 backtest windows.

Everything here is optional. No key, no network or an API error leaves the cache as it was, and
roles.py then types WR/TE with the free-data styles (model["style_free"]).

  python -m app.engine.roles_pff 2026        # pull this season to date (+ last season once)
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".cache" / "pff"
FACET = "receiving/summary"
API = "https://api.pff.com/v1/facet/"
MAX_AGE = 12 * 3600  # re-pull the season-to-date window at most twice a day
POS = {"WR": "WR", "TE": "TE", "HB": "RB", "FB": "RB"}
FEATS = ["slot_rate", "inline_rate", "route_rate"]


def window_path(season: int, lo: int, hi: int, cache: Path | None = None) -> Path:
    return (cache or CACHE) / FACET.replace("/", "_") / str(season) / f"w{lo:02d}-{hi:02d}.json"


def _key() -> str | None:
    k = os.environ.get("PFF_API_KEY")
    if k:
        return k.strip()
    for name in (".env.local", ".env"):
        p = ROOT / name
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip().startswith("PFF_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    return None


def fetch(season: int, lo: int, hi: int, max_age: float | None = None) -> bool:
    """Pull one window into the cache. Returns True if a request was made. Raises on HTTP/network errors."""
    p = window_path(season, lo, hi)
    if p.exists() and (max_age is None or time.time() - p.stat().st_mtime < max_age):
        return False
    key = _key()
    if not key:
        raise LookupError("PFF_API_KEY not set")
    q = urllib.parse.urlencode({"league": "nfl", "season": str(season), "week": ",".join(str(w) for w in range(lo, hi + 1))})
    req = urllib.request.Request(API + FACET + "?" + q, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read()
    rows = next(iter(json.loads(body).values()))
    if not isinstance(rows, list) or not rows:
        raise ValueError("empty PFF response")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmpw")
    tmp.write_bytes(body)
    tmp.replace(p)
    return True


def load_counts(p: Path) -> pd.DataFrame:
    """Counts per PFF player from one cached window (rates are rebuilt from counts so windows can be summed)."""
    d = pd.DataFrame(next(iter(json.loads(p.read_text()).values())))
    d["pos"] = d["position"].map(POS)
    cols = ["player_game_count", "slot_snaps", "inline_snaps", "wide_snaps", "routes", "pass_plays", "targets"]
    d[cols] = d[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    return d.dropna(subset=["pos"])[["player_id", "player", "pos", "team"] + cols]


def rates(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    pp = d["pass_plays"].where(d["pass_plays"] > 0)
    d["slot_rate"] = 100 * d["slot_snaps"] / pp
    d["inline_rate"] = 100 * d["inline_snaps"] / pp
    d["route_rate"] = 100 * d["routes"] / pp
    return d


def norm(s) -> str:
    s = re.sub(r"[^a-z0-9]+", " ", str(s).lower().replace(".", "").replace("'", "").replace("’", ""))
    s = re.sub(r" (jr|sr|ii|iii|iv|v)$", "", s.strip())
    return re.sub(r"\s+", " ", s).strip()


def match(pff: pd.DataFrame, weekly: pd.DataFrame, ids: pd.DataFrame | None = None) -> pd.Series:
    """PFF player_id -> gsis for one season: normalized name + position among players in that season's
    nflverse role table; a name shared by two players at a position is dropped rather than guessed."""
    w = weekly[["gsis", "pos"] + (["pname"] if "pname" in weekly else [])].copy()
    names = []
    if "pname" in w:
        names.append(w.dropna(subset=["pname"]).rename(columns={"pname": "nm"})[["gsis", "pos", "nm"]])
    if ids is not None:
        i = ids.dropna(subset=["gsis_id", "name"]).rename(columns={"gsis_id": "gsis", "name": "nm"})[["gsis", "nm"]]
        names.append(i.merge(w[["gsis", "pos"]].drop_duplicates("gsis"), on="gsis"))
    n = pd.concat(names).assign(nn=lambda x: x["nm"].map(norm)).drop_duplicates(["nn", "pos", "gsis"])
    n = n[~n.duplicated(["nn", "pos"], keep=False)]
    p = pff.drop_duplicates("player_id").assign(nn=lambda x: x["player"].map(norm))
    m = p.merge(n[["nn", "pos", "gsis"]], on=["nn", "pos"], how="inner")
    m = m[~m.duplicated("gsis", keep=False)]
    return m.set_index("player_id")["gsis"]


def latest_window(season: int, cache: Path | None = None) -> Path | None:
    d = (cache or CACHE) / FACET.replace("/", "_") / str(season)
    best = None
    for f in d.glob("w01-*.json"):
        hi = int(f.stem.split("-")[1])
        if best is None or hi > best[0]:
            best = (hi, f)
    return best[1] if best else None


def stamp(season: int) -> tuple:
    """Changes whenever a newer PFF pull lands (cache key for the role board)."""
    out = []
    for y in (season, season - 1):
        p = latest_window(y)
        out.append((p.name, p.stat().st_mtime) if p else None)
    return tuple(out)


_MEMO: dict = {}


def season_features(season: int, weekly: pd.DataFrame | None, ids: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rates per gsis from the widest cached w01-* window of the season (empty frame if none or on any error)."""
    empty = pd.DataFrame(columns=FEATS + ["pff_through", "pff_games", "pff_routes"])
    try:
        p = latest_window(season)
        if p is None or weekly is None or not len(weekly):
            return empty
        key = (season, str(p), p.stat().st_mtime, len(weekly))
        if key in _MEMO:
            return _MEMO[key]
        c = rates(load_counts(p))
        c["gsis"] = c["player_id"].map(match(c, weekly, ids))
        c = c.dropna(subset=["gsis"]).drop_duplicates("gsis").set_index("gsis")
        c = c[c["pass_plays"] > 0]
        out = c[FEATS].assign(pff_through=int(p.stem.split("-")[1]), pff_games=c["player_game_count"], pff_routes=c["routes"])
        _MEMO[key] = out
        return out
    except Exception:
        return empty


def join(agg: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Adds the PFF rates to an aggregate indexed by gsis (NaN where PFF has no row)."""
    cols = FEATS + ["pff_routes"]
    agg = agg.drop(columns=[c for c in cols if c in agg.columns])
    if feats is None or not len(feats):
        return agg.assign(**{c: np.nan for c in cols})
    return agg.join(feats[[c for c in cols if c in feats.columns]], how="left")


def main(argv=None) -> int:
    import sys
    argv = sys.argv[1:] if argv is None else argv
    season = int(argv[0])
    f = ROOT / "app" / "cache.nosync" / "roles" / str(season) / "roles_weekly.csv"
    through = int(pd.read_csv(f, usecols=["week"])["week"].max()) if f.exists() else None
    if not through:
        print("no nflverse role table yet; nothing to align")
        return 0
    calls = 0
    try:
        calls += fetch(season, 1, through, MAX_AGE)
        if not window_path(season - 1, 1, 18).exists():
            calls += fetch(season - 1, 1, 18)
        print(f"PFF alignment through week {through}: {calls} API call(s)")
    except LookupError:
        print("PFF_API_KEY not set: WR/TE styles use free data")
    except Exception as e:  # noqa: BLE001 -- optional source; Desk keeps the free-data styles
        msg = getattr(e, "code", None) or type(e).__name__
        print(f"PFF unavailable ({msg}): WR/TE styles use free data or the last cached pull")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
