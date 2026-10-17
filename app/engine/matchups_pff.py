"""PFF matchup context for the Matchups screen: who a player lines up against and how the units compare.

Descriptive only. backtest/matchup_pff.py (2023-25) found that none of this moves a player beyond ESPN's weekly
projection: the coverage grade a receiver faces given where he lines up (slot vs outside), pass rush vs his line's
pass blocking, run defense vs run blocking. So the screen shows it as context and never changes a number.
PFF has no left/right side or shadow flag for corners, so "across from him" is only as good as his slot rate.

Three season-to-date facet pulls (defense/summary, offense/blocking, defense/coverage_matchup) at most every 12h,
cached raw under .cache/pff/ (gitignored). No key, no network or an error leaves the cache as it was; with no
cache the screen hides the PFF parts. The receiver's own alignment comes from roles_pff's receiving pull.

  python -m app.engine.matchups_pff 2026
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from . import roles_pff as rp

CACHE = rp.CACHE
FACETS = ["defense/summary", "offense/blocking", "defense/coverage_matchup"]
FIX = {"ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "LAR": "LA", "OAK": "LV", "SD": "LAC", "WSH": "WAS", "JAC": "JAX"}
OL = {"T", "G", "C"}
_M: dict = {}


def _path(facet: str, season: int, hi: int) -> Path:
    return CACHE / facet.replace("/", "_") / str(season) / f"w01-{hi:02d}.json"


def fetch(facet: str, season: int, hi: int, max_age: float = rp.MAX_AGE) -> bool:
    p = _path(facet, season, hi)
    if p.exists() and time.time() - p.stat().st_mtime < max_age:
        return False
    key = rp._key()
    if not key:
        raise LookupError("PFF_API_KEY not set")
    q = urllib.parse.urlencode({"league": "nfl", "season": str(season), "week": ",".join(str(w) for w in range(1, hi + 1))})
    req = urllib.request.Request(rp.API + facet + "?" + q, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read()
    if not _rows(json.loads(body)):
        raise ValueError("empty PFF response")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmpw")
    tmp.write_bytes(body)
    tmp.replace(p)
    return True


def _rows(d) -> list:
    """First list in a PFF body; coverage_matchup nests {defenders, receivers, versus} one level down (returns versus)."""
    v = next(iter(d.values()), None) if isinstance(d, dict) else d
    if isinstance(v, dict):
        return v.get("versus") or []
    return v if isinstance(v, list) else []


def _latest(facet: str, season: int) -> Path | None:
    fs = sorted((CACHE / facet.replace("/", "_") / str(season)).glob("w01-*.json"))
    return fs[-1] if fs else None


def _load(facet: str, season: int) -> pd.DataFrame | None:
    p = _latest(facet, season)
    if p is None:
        return None
    k = (facet, str(p), p.stat().st_mtime)
    if _M.get(facet, (None,))[0] != k:
        d = pd.DataFrame(_rows(json.loads(p.read_text())))
        for c in d.columns:
            if c.startswith(("grades_", "snap_counts_")) or c in ("targets", "receptions", "yards", "touchdowns", "interceptions", "total_pressures", "pressures_allowed",
                                                                  "sacks", "sacks_allowed", "stops", "pass_break_ups", "player_game_count", "player_id", "coverage_player_id"):
                d[c] = pd.to_numeric(d[c], errors="coerce")
        if "team" in d:
            d["abbr"] = d["team"].astype(str).str.upper().map(lambda t: FIX.get(t, t))
        _M[facet] = (k, d, int(p.stem.split("-")[1]))
    return _M[facet][1]


def _through(season: int) -> int | None:
    p = _latest("defense/summary", season)
    return int(p.stem.split("-")[1]) if p else None


def _col(d: pd.DataFrame, *names: str) -> pd.Series:
    for n in names:
        if n in d:
            return d[n].fillna(0)
    return pd.Series(0.0, index=d.index)


def _f(x, nd: int = 1):
    return None if x is None or pd.isna(x) else round(float(x), nd)


def _units(season: int) -> dict:
    """Snap-weighted unit grades per team and their place in line (1 = best of 32)."""
    D, O = _load("defense/summary", season), _load("offense/blocking", season)
    out: dict = {}
    spec = [("coverage", D, "grades_coverage_defense", ("snap_counts_coverage",), None), ("pass_rush", D, "grades_pass_rush_defense", ("snap_counts_pass_rush",), None),
            ("run_defense", D, "grades_run_defense", ("snap_counts_run_defense",), None),
            ("pass_block", O, "grades_pass_block", ("snap_counts_pass_block", "pass_block_snaps"), OL), ("run_block", O, "grades_run_block", ("snap_counts_run_block", "run_block_snaps"), OL)]
    for k, d, g, w, pos in spec:
        if d is None or g not in d:
            continue
        x = d if pos is None else d[d["position"].isin(pos)]
        x = x.assign(w=_col(x, *w)).dropna(subset=[g])
        x = x[x["w"] > 0]
        t = (x[g] * x["w"]).groupby(x["abbr"]).sum() / x.groupby("abbr")["w"].sum()
        rk = t.rank(ascending=False, method="min")
        for a, v in t.items():
            out.setdefault(a, {})[k] = {"grade": round(float(v), 1), "rank": int(rk[a]), "of": int(len(t))}
    return out


def _def_row(r: pd.Series) -> dict:
    cov = r.get("snap_counts_coverage") or 0
    return {"pff_id": int(r["player_id"]), "name": r.get("player"), "pos": r.get("position"), "games": int(r.get("player_game_count") or 0),
            "corner": int(r.get("snap_counts_corner") or 0), "slot": int(r.get("snap_counts_slot") or 0), "box": int(r.get("snap_counts_box") or 0),
            "cov_snaps": int(cov), "cov_grade": _f(r.get("grades_coverage_defense")), "tgt": int(r.get("targets") or 0), "rec": int(r.get("receptions") or 0),
            "yds": int(r.get("yards") or 0), "td": int(r.get("touchdowns") or 0), "int": int(r.get("interceptions") or 0),
            "yds_cov": _f((r.get("yards") or 0) / cov, 2) if cov else None,
            "rush_snaps": int(r.get("snap_counts_pass_rush") or 0), "rush_grade": _f(r.get("grades_pass_rush_defense")), "pressures": int(r.get("total_pressures") or 0),
            "sacks": _f(r.get("sacks")), "prp": _f(100 * (r.get("total_pressures") or 0) / r["snap_counts_pass_rush"]) if (r.get("snap_counts_pass_rush") or 0) > 0 else None,
            "run_snaps": int(r.get("snap_counts_run_defense") or 0), "run_grade": _f(r.get("grades_run_defense")), "stops": int(r.get("stops") or 0)}


def defense(abbr: str, season: int) -> dict | None:
    """One defense's players by job: outside corners, slot defenders, pass rushers, run defenders."""
    D = _load("defense/summary", season)
    if D is None:
        return None
    t = D[D["abbr"] == abbr]
    if not len(t):
        return None
    c, s = _col(t, "snap_counts_corner"), _col(t, "snap_counts_slot")
    rush, run = _col(t, "snap_counts_pass_rush"), _col(t, "snap_counts_run_defense")
    pick = lambda m, by, n: [_def_row(r) | {"share": _f(100 * r[by] / max(1, t[by].sum()), 0)} for _, r in t[m].sort_values(by, ascending=False).head(n).iterrows()]
    return {"abbr": abbr, "season": season, "through": _through(season), "units": _units(season).get(abbr, {}),
            "outside": pick(c >= 0.1 * c.sum(), "snap_counts_corner", 4), "slot": pick(s >= 0.15 * s.sum(), "snap_counts_slot", 3),
            "rushers": pick(rush >= 0.2 * rush.max(), "total_pressures", 5) if "total_pressures" in t else [],
            "run": pick(run >= 0.25 * run.max(), "stops", 5) if "stops" in t else []}


def _line(team: str, season: int) -> list[dict]:
    O = _load("offense/blocking", season)
    if O is None:
        return []
    o = O[(O["abbr"] == team) & O["position"].isin(OL)].copy()
    slots = [c for c in ("snap_counts_lt", "snap_counts_lg", "snap_counts_ce", "snap_counts_rg", "snap_counts_rt") if c in o]
    o["snaps"] = o[slots].fillna(0).sum(axis=1) if slots else 0
    o = o.sort_values("snaps", ascending=False).head(5)  # the five who played most; a rotating sixth is left out
    out = []
    for _i, r in o.iterrows():
        at = max(slots, key=lambda c: r[c] or 0)[len("snap_counts_"):].upper().replace("CE", "C") if slots else r.get("position")
        out.append({"name": r.get("player"), "at": at, "snaps": int(r["snaps"]), "pass_grade": _f(r.get("grades_pass_block")), "run_grade": _f(r.get("grades_run_block")),
                    "pressures": int(r.get("pressures_allowed") or 0), "sacks": int(r.get("sacks_allowed") or 0)})
    order = {"LT": 0, "LG": 1, "C": 2, "RG": 3, "RT": 4}
    return sorted(out, key=lambda x: order.get(x["at"], 9))


def _receiver(name: str, team: str | None, season: int) -> pd.Series | None:
    p = rp.latest_window(season)
    if p is None:
        return None
    d = json.loads(p.read_text())
    r = pd.DataFrame(_rows(d))
    if not len(r):
        return None
    r["nn"] = r["player"].map(rp.norm)
    m = r[r["nn"] == rp.norm(name)]
    if team and len(m) > 1:
        m = m[m["team"].astype(str).str.upper().map(lambda t: FIX.get(t, t)) == team]
    return m.iloc[0] if len(m) == 1 else None


def player(pv: dict, season: int) -> dict | None:
    """The PFF panel for one player against pv['opp']. None when nothing is cached."""
    opp, pos, team = pv.get("opp"), pv.get("pos"), pv.get("team")
    if not opp:
        return None
    dfn = defense(opp, season)
    if dfn is None:
        return None
    units = _units(season)
    out = {"opp": opp, "season": season, "through": dfn["through"], "pos": pos, "them": dfn["units"], "us": units.get(team or "", {}), "line": _line(team, season) if team else []}
    if pos in ("WR", "TE"):
        me = _receiver(pv.get("name", ""), team, season)
        al = None
        if me is not None and (me.get("pass_plays") or 0) > 0:
            pp_ = float(me["pass_plays"])
            al = {k: round(100 * float(me.get(f"{k}_snaps") or 0) / pp_) for k in ("slot", "wide", "inline")}
            al["routes"] = int(me.get("routes") or 0)
        out["align"] = al
        out["outside"], out["slot"] = dfn["outside"], dfn["slot"]
        if al:
            out["likely"] = "slot" if al["slot"] >= 50 else ("inline" if al["inline"] >= 50 else "outside")
        V = _load("defense/coverage_matchup", season)
        D = _load("defense/summary", season)
        out["covered_by"] = []
        if V is not None and len(V) and me is not None and D is not None:
            v = V[V["player_id"] == int(me["player_id"])].groupby("coverage_player_id")[["targets", "receptions", "yards", "touchdowns"]].sum()
            names = D.drop_duplicates("player_id").set_index("player_id")
            v = v.sort_values("targets", ascending=False).head(6)
            out["covered_by"] = [{"name": names["player"].get(i), "team": names["abbr"].get(i), "pos": names["position"].get(i), "tgt": int(x.targets), "rec": int(x.receptions),
                                  "yds": int(x.yards), "td": int(x.touchdowns), "theirs": names["abbr"].get(i) == opp} for i, x in v.iterrows()]
    elif pos == "QB":
        out["rushers"] = dfn["rushers"]
    elif pos == "RB":
        out["run"] = dfn["run"]
    return out


def main(argv=None) -> int:
    import sys
    argv = sys.argv[1:] if argv is None else argv
    season = int(argv[0])
    f = rp.ROOT / "app" / "cache.nosync" / "roles" / str(season) / "roles_weekly.csv"
    through = int(pd.read_csv(f, usecols=["week"])["week"].max()) if f.exists() else None
    if not through:
        print("no nflverse role table yet; nothing to pull")
        return 0
    calls = 0
    try:
        for fc in FACETS:
            calls += fetch(fc, season, through)
        print(f"PFF matchups through week {through}: {calls} API call(s)")
    except LookupError:
        print("PFF_API_KEY not set: Matchups shows free data only")
    except Exception as e:  # noqa: BLE001 -- optional source
        print(f"PFF unavailable ({getattr(e, 'code', None) or type(e).__name__}): Matchups keeps the last cached pull")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
