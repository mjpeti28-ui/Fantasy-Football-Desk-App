"""Weekly role table for QB/RB/WR/TE from nflverse play-by-play, FTN charting and snap counts.

Pandas only (no nflreadpy / pyarrow): reads the nflverse release CSVs over HTTPS and caches one
small aggregated file per season at history/nflverse/<season>/roles_weekly.csv.

One row per player-game: his counts (targets, carries, red-zone and goal-line work, late-down and
two-minute targets, air yards, screens, contested targets, dropbacks, scrambles ...), his snap share,
and the team totals the shares are taken against.
"""
from __future__ import annotations

import io
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://github.com/nflverse/nflverse-data/releases/download"
PBP_COLS = ["game_id", "play_id", "season", "week", "season_type", "posteam", "down", "yardline_100",
            "half_seconds_remaining", "play_type", "qb_dropback", "qb_scramble", "sack", "pass_attempt",
            "rush_attempt", "complete_pass", "air_yards", "yards_after_catch", "rushing_yards",
            "passer_player_id", "receiver_player_id", "rusher_player_id", "two_point_attempt"]
FTN_COLS = ["nflverse_game_id", "nflverse_play_id", "is_screen_pass", "is_contested_ball"]
SNAP_COLS = ["season", "game_type", "week", "pfr_player_id", "team", "offense_snaps", "offense_pct"]
STAT_COLS = ["player_id", "player_display_name", "position", "season", "week", "season_type", "team", "opponent_team",
             "completions", "attempts", "passing_yards", "passing_tds", "passing_interceptions", "sack_fumbles_lost",
             "carries", "rushing_yards", "rushing_tds", "rushing_fumbles_lost", "receptions", "receiving_yards",
             "receiving_tds", "receiving_fumbles_lost", "fantasy_points_ppr"]


def _get(url: str, retries: int = 2) -> bytes:
    for i in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except Exception:
            if i == retries:
                raise
            time.sleep(2)
    raise RuntimeError(url)


def _csv(url: str, cols, gz: bool = False) -> pd.DataFrame:
    raw = _get(url)
    return pd.read_csv(io.BytesIO(raw), usecols=lambda c: c in cols, compression="gzip" if gz else None, low_memory=False)


def _bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(float)
    return s.astype(str).str.upper().map({"TRUE": 1.0, "FALSE": 0.0, "1": 1.0, "0": 0.0, "1.0": 1.0, "0.0": 0.0})


def build(season: int, ids: pd.DataFrame) -> pd.DataFrame:
    """ids: nflverse id map with gsis_id, pfr_id, espn_id, position."""
    b = _csv(f"{BASE}/pbp/play_by_play_{season}.csv.gz", PBP_COLS, gz=True)
    b = b[(b["season_type"] == "REG") & b["play_type"].isin(["pass", "run"]) & (b["two_point_attempt"].fillna(0) != 1)].copy()
    try:
        f = _csv(f"{BASE}/ftn_charting/ftn_charting_{season}.csv", FTN_COLS)
        f = f.rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
        f["is_screen_pass"] = _bool(f["is_screen_pass"]); f["is_contested_ball"] = _bool(f["is_contested_ball"])
        b = b.merge(f, on=["game_id", "play_id"], how="left")
    except Exception:  # FTN charting starts in 2022 and can lag a few days
        b["is_screen_pass"] = np.nan; b["is_contested_ball"] = np.nan
    b["rz"] = b["yardline_100"] <= 20
    b["gl"] = b["yardline_100"] <= 5
    b["late_down"] = b["down"].isin([3, 4])
    b["two_min"] = b["half_seconds_remaining"] <= 120
    b["ez"] = b["air_yards"].notna() & (b["air_yards"] >= b["yardline_100"])
    k = ["season", "week", "posteam"]

    tg = b[b["receiver_player_id"].notna() & (b["pass_attempt"] == 1)].copy()
    tg["ayp"] = tg["air_yards"].clip(lower=0)
    tg["deep"] = tg["air_yards"] >= 20
    tg["behind"] = tg["air_yards"] <= 0
    ru = b[(b["rush_attempt"] == 1) & b["rusher_player_id"].notna()].copy()
    ru["designed"] = (ru["qb_scramble"].fillna(0) != 1).astype(int)
    ru["rz_d"] = (ru["rz"] & (ru["designed"] == 1)).astype(int)
    ru["gl_d"] = (ru["gl"] & (ru["designed"] == 1)).astype(int)
    pa = b[(b["qb_dropback"] == 1) & b["passer_player_id"].notna()].copy()
    pa["deep"] = pa["air_yards"] >= 20

    team = pd.DataFrame({
        "t_tgt": tg.groupby(k).size(), "t_ay": tg.groupby(k)["ayp"].sum(),
        "t_rz_tgt": tg[tg.rz].groupby(k).size(), "t_ld_tgt": tg[tg.late_down].groupby(k).size(),
        "t_2m_tgt": tg[tg.two_min].groupby(k).size(),
        "t_car": ru.groupby(k)["designed"].sum(), "t_rz_car": ru.groupby(k)["rz_d"].sum(), "t_gl_car": ru.groupby(k)["gl_d"].sum(),
        "t_db": b[b["qb_dropback"] == 1].groupby(k).size(), "t_plays": b.groupby(k).size(),
    }).fillna(0).reset_index()

    R = tg.groupby(k + ["receiver_player_id"]).agg(
        tgt=("play_id", "size"), ay=("ayp", "sum"), ay_sum=("air_yards", "sum"), deep=("deep", "sum"), behind=("behind", "sum"),
        screen=("is_screen_pass", "sum"), charted=("is_screen_pass", "count"), contested=("is_contested_ball", "sum"),
        rz_tgt=("rz", "sum"), ez_tgt=("ez", "sum"), ld_tgt=("late_down", "sum"), tm_tgt=("two_min", "sum"),
        rec=("complete_pass", "sum"), yac=("yards_after_catch", "sum")).reset_index().rename(columns={"receiver_player_id": "gsis"})
    C = ru.groupby(k + ["rusher_player_id"]).agg(
        car=("designed", "sum"), scr=("qb_scramble", "sum"), rz_car=("rz_d", "sum"), gl_car=("gl_d", "sum"),
        ryd=("rushing_yards", "sum")).reset_index().rename(columns={"rusher_player_id": "gsis"})
    P = pa.groupby(k + ["passer_player_id"]).agg(
        db=("play_id", "size"), att=("pass_attempt", "sum"), sacks=("sack", "sum"), p_ay=("air_yards", "sum"),
        p_deep=("deep", "sum")).reset_index().rename(columns={"passer_player_id": "gsis"})
    m = R.merge(C, on=k + ["gsis"], how="outer").merge(P, on=k + ["gsis"], how="outer")

    s = _csv(f"{BASE}/snap_counts/snap_counts_{season}.csv", SNAP_COLS)
    s = s[s["game_type"] == "REG"]
    pfr = ids.dropna(subset=["pfr_id"]).drop_duplicates("pfr_id").set_index("pfr_id")["gsis_id"]
    s["gsis"] = s["pfr_player_id"].map(pfr)
    s = s[s["gsis"].notna()].rename(columns={"team": "posteam"})
    s["snap"] = s["offense_pct"] * (100 if s["offense_pct"].max() <= 1.5 else 1)
    m = m.merge(s[k + ["gsis", "offense_snaps", "snap"]], on=k + ["gsis"], how="outer")
    num = [c for c in m.columns if c not in k + ["gsis"]]
    m[num] = m[num].fillna(0)
    m = m.merge(team, on=k, how="left")
    # box-score stats and PPR points (nflverse weekly player stats)
    try:
        st = _csv(f"{BASE}/stats_player/stats_player_week_{season}.csv", STAT_COLS)
        st = st[st["season_type"] == "REG"].rename(columns={"player_id": "gsis", "team": "posteam", "opponent_team": "opp",
            "player_display_name": "pname", "position": "spos", "attempts": "pass_att", "completions": "cmp",
            "passing_yards": "pass_yd", "passing_tds": "pass_td", "passing_interceptions": "ints", "rushing_yards": "rush_yd",
            "rushing_tds": "rush_td", "receptions": "recs", "receiving_yards": "rec_yd", "receiving_tds": "rec_td",
            "fantasy_points_ppr": "ppr"})
        st["fl"] = st[["sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost"]].fillna(0).sum(axis=1)
        keep = ["season", "week", "posteam", "gsis", "opp", "pname", "spos", "cmp", "pass_att", "pass_yd", "pass_td", "ints",
                "rush_yd", "rush_td", "recs", "rec_yd", "rec_td", "fl", "ppr"]
        m = m.merge(st[keep], on=k + ["gsis"], how="left")
        box = ["cmp", "pass_att", "pass_yd", "pass_td", "ints", "rush_yd", "rush_td", "recs", "rec_yd", "rec_td", "fl", "ppr"]
        m[box] = m[box].fillna(0)  # snaps but no touches: a zero line
        m["opp"] = m.groupby(k)["opp"].transform("first")
    except Exception:
        pass
    pos = ids.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id").set_index("gsis_id")
    m["pos"] = m["gsis"].map(pos["position"])
    if "spos" in m:
        m["pos"] = m["pos"].fillna(m["spos"])
    m["espn_id"] = pd.to_numeric(m["gsis"].map(pos["espn_id"]), errors="coerce")
    m = m[m["pos"].isin(["QB", "RB", "WR", "TE"]) & (m["snap"] > 0)]
    return m.sort_values(["gsis", "week"]).reset_index(drop=True)


# ---------- has nflverse published anything new? (HEAD requests only, nothing is downloaded)
# A new week's rows need its snap counts and FTN charting, both refreshed once a day by nflverse. The
# table is rebuilt when either file's Last-Modified differs from the one recorded at the last build.
WATCH = ("snap_counts/snap_counts_{s}.csv", "ftn_charting/ftn_charting_{s}.csv")
STAMPS_FILE = "roles_stamps.json"
MIN_GAP = 600  # seconds between upstream checks


def upstream_stamps(season: int) -> dict:
    """{'snap_counts': Last-Modified, 'ftn_charting': ...}; {} when offline or anything fails."""
    out = {}
    for tpl in WATCH:
        try:
            req = urllib.request.Request(f"{BASE}/{tpl.format(s=season)}", method="HEAD")
            with urllib.request.urlopen(req, timeout=20) as r:
                out[tpl.split("/")[0]] = r.headers.get("Last-Modified") or r.headers.get("ETag") or ""
        except Exception:
            return {}
    return out


def _stamps_path(cache: Path, season: int) -> Path:
    return cache / str(season) / STAMPS_FILE


def _read_stamps(cache: Path, season: int) -> dict:
    try:
        return json.loads(_stamps_path(cache, season).read_text())
    except Exception:
        return {}


def save_stamps(cache: Path, season: int, stamps: dict) -> None:
    p = _stamps_path(cache, season)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmpw")
        tmp.write_text(json.dumps({"stamps": stamps, "checked": time.time()}))
        tmp.replace(p)
    except OSError:
        pass


def is_stale(season: int, cache: Path, live: dict | None = None) -> bool:
    """True when nflverse has published new snap/charting data since the table was built."""
    live = upstream_stamps(season) if live is None else live
    f = cache / str(season) / "roles_weekly.csv"
    if not live:
        return False  # offline: keep what we have
    if not f.exists():
        return True
    saved = _read_stamps(cache, season).get("stamps")
    if saved:
        return live != saved
    try:  # table built before stamps existed: stale if upstream is newer than the file
        from email.utils import parsedate_to_datetime
        newest = max(parsedate_to_datetime(v).timestamp() for v in live.values() if v)
        return newest > f.stat().st_mtime
    except Exception:
        return True


def refresh_if_stale(season: int, cache: Path, ids: pd.DataFrame, min_gap: float = MIN_GAP) -> bool:
    """Rebuild the season's table only if nflverse has new data; returns whether it did."""
    if time.time() - float(_read_stamps(cache, season).get("checked", 0)) < min_gap:
        return False
    live = upstream_stamps(season)
    if not is_stale(season, cache, live):
        if live:
            save_stamps(cache, season, live)
        return False
    load(season, cache, ids, refresh=True)
    return True


_adopted = False


def cache_dir() -> Path:
    """Where Desk keeps its per-season role tables: app/cache.nosync/roles (out of iCloud).
    Tables built before this moved (history/nflverse/<season>/) are copied over once."""
    from . import store
    global _adopted
    new = store.local_dir("roles")
    if _adopted:
        return new
    _adopted = True
    old = Path(__file__).resolve().parents[2] / "history" / "nflverse"
    for f in old.glob("*/roles_weekly.csv"):
        dst = new / f.parent.name / f.name
        if not dst.exists():
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_suffix(".tmpw")
                tmp.write_bytes(f.read_bytes())
                tmp.replace(dst)
            except OSError:
                pass
    return new


def load(season: int, cache: Path, ids: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    p = cache / str(season) / "roles_weekly.csv"
    if p.exists() and not refresh:
        return pd.read_csv(p)
    stamps = upstream_stamps(season)  # taken before the download, so an update during it is seen next time
    d = build(season, ids)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmpw")
    d.to_csv(tmp, index=False)
    tmp.replace(p)
    if stamps:
        save_stamps(cache, season, stamps)
    return d


if __name__ == "__main__":  # python -m app.engine.roles_data [--if-stale] 2026 [2025 ...]
    import sys
    root = Path(__file__).resolve().parents[2]
    ids = pd.read_csv(root / "history" / "nflverse" / "ids.csv")
    only_if_stale = "--if-stale" in sys.argv
    for y in [int(a) for a in sys.argv[1:] if a.isdigit()]:
        t = time.time()
        if only_if_stale:
            if not refresh_if_stale(y, cache_dir(), ids):
                print(y, "nflverse usage data unchanged since the last build")
                continue
            d = pd.read_csv(cache_dir() / str(y) / "roles_weekly.csv")
        else:
            d = load(y, cache_dir(), ids, refresh=True)
        print(y, len(d), "player-games through week", int(d["week"].max()), f"{time.time() - t:.0f}s")
