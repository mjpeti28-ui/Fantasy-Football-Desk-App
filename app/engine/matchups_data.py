"""Per-play tables behind the Matchups screen: nflverse play-by-play + FTN charting + participation.

Sources and licences (checked 2026-10-10):
  play-by-play   nflverse / nflfastR (nflreadr's page states no licence)
  FTN charting   CC-BY-SA 4.0, credit "FTN Data via nflverse" (2022 on, charted within ~48 h of each game)
  participation  2023 on is FTN data (CC-BY-SA 4.0, "FTN Data via nflverse"): coverage shell, man/zone,
                 targeted route, pressure flag, who was on the field. It is published after the
                 postseason, so the current season has no coverage tags.
Deliberately NOT used: Pro Football Reference per-defender stats and snap counts (PFR's data policy bars
using its data to prompt or train AI models, and nflverse redistributes them with no licence).

One row per pass dropback or designed run, cached per season in app/cache.nosync/matchups/<season>.csv.gz.
    python -m app.engine.matchups_data 2023 2024 2025 2026
"""
from __future__ import annotations

import io
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://github.com/nflverse/nflverse-data/releases/download"
HISTORY = (2023, 2024, 2025)  # seasons with FTN-sourced coverage tags
EVENTS = ["sack_player_id", "half_sack_1_player_id", "half_sack_2_player_id", "qb_hit_1_player_id", "qb_hit_2_player_id",
          "tackle_for_loss_1_player_id", "tackle_for_loss_2_player_id", "interception_player_id",
          "pass_defense_1_player_id", "pass_defense_2_player_id", "solo_tackle_1_player_id", "assist_tackle_1_player_id",
          "assist_tackle_2_player_id", "assist_tackle_3_player_id", "assist_tackle_4_player_id", "forced_fumble_player_1_player_id"]
PBP_COLS = ["game_id", "play_id", "season", "week", "season_type", "posteam", "defteam", "play_type", "qb_dropback", "pass_attempt",
            "rush_attempt", "complete_pass", "sack", "interception", "pass_touchdown", "rush_touchdown", "yards_gained",
            "rushing_yards", "air_yards", "yards_after_catch", "epa", "success", "pass_location", "pass_length", "run_location",
            "run_gap", "shotgun", "no_huddle", "qb_scramble", "qb_hit", "down", "ydstogo", "yardline_100", "two_point_attempt",
            "passer_player_id", "receiver_player_id", "rusher_player_id"] + EVENTS
FTN_COLS = ["nflverse_game_id", "nflverse_play_id", "is_play_action", "is_screen_pass", "is_rpo", "is_motion", "is_qb_out_of_pocket",
            "is_throw_away", "is_drop", "is_contested_ball", "is_catchable_ball", "is_interception_worthy", "read_thrown",
            "n_blitzers", "n_pass_rushers", "starting_hash"]
PART_COLS = ["nflverse_game_id", "play_id", "offense_formation", "offense_personnel", "defense_personnel", "defenders_in_box",
             "number_of_pass_rushers", "offense_players", "defense_players", "offense_positions", "defense_positions",
             "time_to_throw", "was_pressure", "route", "defense_man_zone_type", "defense_coverage_type"]


def _get(url: str, retries: int = 2) -> bytes:
    for i in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                return r.read()
        except Exception:
            if i == retries:
                raise
            time.sleep(2)
    raise RuntimeError(url)


def _csv(url: str, cols, gz: bool = False) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(_get(url)), usecols=lambda c: c in cols, compression="gzip" if gz else None, low_memory=False)


def _bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(float)
    return s.astype(str).str.upper().map({"TRUE": 1.0, "FALSE": 0.0, "1": 1.0, "0": 0.0, "1.0": 1.0, "0.0": 0.0})


def build(season: int) -> pd.DataFrame:
    b = _csv(f"{BASE}/pbp/play_by_play_{season}.csv.gz", PBP_COLS, gz=True)
    b = b[(b["season_type"] == "REG") & b["play_type"].isin(["pass", "run"]) & (b["two_point_attempt"].fillna(0) != 1)].copy()
    b["db"] = (b["qb_dropback"] == 1).astype(int)
    b["run"] = ((b["rush_attempt"] == 1) & (b["qb_dropback"] != 1)).astype(int)
    b = b[(b["db"] == 1) | (b["run"] == 1)]
    try:
        f = _csv(f"{BASE}/ftn_charting/ftn_charting_{season}.csv", FTN_COLS)
        f = f.rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"}).drop_duplicates(["game_id", "play_id"])
        for c in ("is_play_action", "is_screen_pass", "is_rpo", "is_motion", "is_qb_out_of_pocket", "is_throw_away", "is_drop",
                  "is_contested_ball", "is_catchable_ball", "is_interception_worthy"):
            if c in f:
                f[c] = _bool(f[c])
        b = b.merge(f, on=["game_id", "play_id"], how="left")
    except Exception:
        pass
    try:
        p = _csv(f"{BASE}/pbp_participation/pbp_participation_{season}.csv", PART_COLS)
        p = p.rename(columns={"nflverse_game_id": "game_id", "number_of_pass_rushers": "n_rushers_p"}).drop_duplicates(["game_id", "play_id"])
        p["was_pressure"] = _bool(p["was_pressure"])
        b = b.merge(p, on=["game_id", "play_id"], how="left")
    except Exception:  # not published yet for the current season
        pass
    b = b.rename(columns={"passer_player_id": "passer", "receiver_player_id": "receiver", "rusher_player_id": "rusher"})
    return b.sort_values(["week", "game_id", "play_id"]).reset_index(drop=True)


def cache_dir() -> Path:
    from . import store
    return store.local_dir("matchups")


def load(season: int, refresh: bool = False, max_age_h: float = 6.0) -> pd.DataFrame:
    """History seasons are built once; the current season is refreshed when older than max_age_h."""
    p = cache_dir() / f"{season}.csv.gz"
    if p.exists() and not refresh and (season in HISTORY or (time.time() - p.stat().st_mtime) / 3600 < max_age_h):
        return pd.read_csv(p, low_memory=False)
    try:
        d = build(season)
    except Exception:
        if p.exists():  # offline: a stale file beats an empty screen
            return pd.read_csv(p, low_memory=False)
        raise
    tmp = p.with_suffix(".tmpw")
    d.to_csv(tmp, index=False, compression="gzip")
    tmp.replace(p)
    return d


if __name__ == "__main__":
    for y in [int(a) for a in sys.argv[1:]]:
        t = time.time()
        d = load(y, refresh=True)
        cov = d["defense_coverage_type"].notna().mean() if "defense_coverage_type" in d else float("nan")
        ftn = d["n_blitzers"].notna().mean() if "n_blitzers" in d else float("nan")
        print(y, len(d), "plays through week", int(d["week"].max()), f"coverage tagged {cov:.2f} ftn {ftn:.2f} {time.time() - t:.0f}s", flush=True)
