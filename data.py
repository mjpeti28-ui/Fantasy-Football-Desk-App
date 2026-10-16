import copy
import logging
import os
import re
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from config import CSV_RENAME, PROJECTIONS_CSV, settings
from espn_client import fetch_rosters_from_espn
from rosters import Fantasy_Rosters
from teams import canonical_dst_name, canonical_team

SUFFIXES = [' jr', ' sr', ' iii', ' ii', ' iv', ' v']
DEFAULT_PROJECTIONS_PATH = Path(__file__).with_name(PROJECTIONS_CSV)

def normalize_name(s: str) -> str:
    s = s.lower()
    s = s.replace('.', '').replace("’", "'").replace("'", '')
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    for suf in SUFFIXES:
        s = re.sub(rf'{suf}$', '', s.strip())
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def load_projections(path: Optional[str] = None) -> pd.DataFrame:
    target = Path(path) if path is not None else DEFAULT_PROJECTIONS_PATH
    if not target.exists():
        return pd.DataFrame(columns=["name_norm", "ProjPoints"])
    proj_df = pd.read_csv(target)
    if "playerName" not in proj_df.columns or "fantasyPoints" not in proj_df.columns:
        raise ValueError("Projections CSV must contain 'playerName' and 'fantasyPoints' columns.")
    if "position" in proj_df.columns:
        is_dst = proj_df["position"].astype(str).str.upper().isin({"DST", "D/ST", "DEF"})
        teams = proj_df["teamName"] if "teamName" in proj_df.columns else pd.Series([None] * len(proj_df), index=proj_df.index)
        proj_df.loc[is_dst, "playerName"] = [
            canonical_dst_name(name, team) or name
            for name, team in zip(proj_df.loc[is_dst, "playerName"], teams[is_dst])
        ]
    proj_df["name_norm"] = proj_df["playerName"].apply(normalize_name)
    proj_df["ProjPoints"] = proj_df["fantasyPoints"].astype(float)
    team_series = proj_df["teamName"] if "teamName" in proj_df.columns else pd.Series([None] * len(proj_df), index=proj_df.index)
    pos_series = proj_df["position"] if "position" in proj_df.columns else pd.Series([None] * len(proj_df), index=proj_df.index)
    proj_df["fallback_key"] = [
        _fallback_key(n, t, p) for n, t, p in zip(proj_df["name_norm"], team_series, pos_series)
    ]
    proj_df = proj_df[["name_norm", "ProjPoints", "fallback_key"]]
    proj_df = proj_df.drop_duplicates("name_norm", keep="first")
    return proj_df


def _fallback_key(name_norm: str, team, position) -> Optional[str]:
    """Last name + team + position, for joining sources that spell first names
    differently ("Kenny"/"Kenneth Gainwell", "Chig"/"Chigoziem Okonkwo").

    Stopgap until joins can use a shared player ID (the FantasyPros API exposes
    ESPN/Yahoo external IDs, which makes name matching unnecessary).
    """
    if not name_norm or team is None or position is None:
        return None
    team_code = canonical_team(team)
    if not isinstance(team_code, str) or not team_code or team_code.upper() in {"FA", "NAN"}:
        return None
    return f"{name_norm.split()[-1]}|{team_code}|{str(position).upper()}"


def _impute_missing_projections(df: pd.DataFrame) -> pd.Series:
    """Fill missing ProjPoints from the same position's rank->points curve.

    A player with a ranking but no projection (injured-and-back, a name the
    projection source spells differently, a rookie the projection file omits)
    used to get the position *median*, so e.g. RB40 and RB90 got identical
    points. Interpolating on RankOriginal against players at that position who
    do have projections keeps the imputed value consistent with where the
    rankings put him -- neutral with respect to the rank rescale below.
    """
    import numpy as np

    out = df["ProjPoints"].copy()
    for _, idx in df.groupby("Position").groups.items():
        sub = df.loc[idx]
        known = sub[sub["ProjPoints"].notna()].sort_values("RankOriginal")
        missing = sub[sub["ProjPoints"].isna()]
        if missing.empty or known.empty:
            continue
        if len(known) == 1:
            out.loc[missing.index] = known["ProjPoints"].iloc[0]
            continue
        # np.interp clamps outside the known range, which is the right behavior
        # for deep-tail players (they get the worst known projection, not an extrapolation).
        out.loc[missing.index] = np.interp(
            missing["RankOriginal"].astype(float),
            known["RankOriginal"].astype(float),
            known["ProjPoints"].astype(float),
        )
    return out


def _read_rankings_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except pd.errors.ParserError:
        return pd.read_csv(path, skiprows=1)
    # If the file still comes back as a single generic column, try skipping the first row
    if len(df.columns) == 1:
        try:
            df_alt = pd.read_csv(path, skiprows=1)
            if len(df_alt.columns) > 1:
                df = df_alt
        except pd.errors.ParserError:
            pass
    return df


def _prepare_rankings_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.astype(str).str.strip().str.replace('\ufeff', '', regex=False)

    rename_map = dict(CSV_RENAME)
    rename_map.update({
        "Overall Rank": "Rank",
        "Full Name": "Name",
        "Team Abbreviation": "Team",
        "Team": "Team",
        "Name": "Name",
        "Position Rank": "PosRank",
    })
    df = df.rename(columns=rename_map)
    if "Rank" not in df.columns or "Position" not in df.columns:
        raise ValueError("Rankings CSV must have columns that map to ['Rank','Position','Name','Team']")

    extra_cols = {
        "Bye Week": "ByeWeek",
        "ADP": "ADP",
        "Projected Points": "ProjPointsCsv",
        "Auction Value": "AuctionValue",
    }
    for src, dst in extra_cols.items():
        if src in df.columns:
            df[dst] = df[src]
    df["Rank"] = df["Rank"].astype(int)
    # Sources disagree on team codes (JAX/JAC, ARI/ARZ) and on defense names
    # ("Rams D/ST" vs "Los Angeles Rams"); canonicalize both before anything
    # keys on them, otherwise every roster D/ST silently fails to match.
    if "Team" in df.columns:
        df["Team"] = df["Team"].apply(canonical_team)
    team_col = df["Team"] if "Team" in df.columns else pd.Series([None] * len(df), index=df.index)
    is_dst = df["Position"].astype(str).str.upper().isin({"DST", "D/ST", "DEF"})
    df.loc[is_dst, "Name"] = [
        canonical_dst_name(name, team) or name
        for name, team in zip(df.loc[is_dst, "Name"], team_col[is_dst])
    ]
    df["name_norm"] = df["Name"].apply(normalize_name)
    df["PosRank"] = df.groupby("Position")["Rank"].rank(method="min").astype(int)
    df["RankOriginal"] = df["Rank"]
    return df


def load_rankings(
    csv_path: str,
    projections_path: Optional[str] = None,
    *,
    projection_scale_beta: Optional[float] = None,
    supplemental_path: Optional[str] = None,
) -> pd.DataFrame:
    df_primary = _prepare_rankings_dataframe(_read_rankings_csv(csv_path))

    if supplemental_path:
        df_supp = _prepare_rankings_dataframe(_read_rankings_csv(supplemental_path))
        df_supp = df_supp[~df_supp["Position"].isin({"K", "DST"})]
        existing = set(df_primary["name_norm"])
        df_supp = df_supp[~df_supp["name_norm"].isin(existing)]
        df = pd.concat([df_primary, df_supp], ignore_index=True)
        df = df.sort_values(["Position", "RankOriginal", "Rank"]).reset_index(drop=True)
        df["Rank"] = df.index + 1
        df["PosRank"] = df.groupby("Position")["Rank"].rank(method="min").astype(int)
    else:
        df = df_primary

    proj_df = load_projections(projections_path)
    beta_default = settings.get("projection_scale_beta")
    beta = beta_default if projection_scale_beta is None else projection_scale_beta
    if not proj_df.empty:
        df = df.merge(proj_df[["name_norm", "ProjPoints"]], on="name_norm", how="left", suffixes=('', '_proj'))
        # Provenance of each row's ProjPoints, for the health check.
        df["ProjSource"] = df["ProjPoints"].notna().map({True: "exact", False: None})
        # Second pass for name-spelling mismatches: last name + team + position,
        # only where that key is unique on both sides.
        fb = proj_df.dropna(subset=["fallback_key"])
        fb = fb[~fb["fallback_key"].duplicated(keep=False)].set_index("fallback_key")["ProjPoints"]
        team_col = df["Team"] if "Team" in df.columns else pd.Series([None] * len(df), index=df.index)
        df_keys = pd.Series(
            [_fallback_key(n, t, p) for n, t, p in zip(df["name_norm"], team_col, df["Position"])],
            index=df.index,
        )
        dup_keys = set(df_keys[df_keys.duplicated(keep=False)].dropna())
        need = df["ProjPoints"].isna() & df_keys.notna() & ~df_keys.isin(dup_keys)
        df.loc[need, "ProjPoints"] = df_keys[need].map(fb)
        df.loc[need & df["ProjPoints"].notna() & df["ProjSource"].isna(), "ProjSource"] = "fallback"
        if df["ProjPoints"].notna().any():
            overall_median = df["ProjPoints"].median()
            if pd.isna(overall_median):
                overall_median = 0.0
            df["ProjPoints"] = _impute_missing_projections(df)
            df.loc[df["ProjPoints"].notna() & df["ProjSource"].isna(), "ProjSource"] = "imputed"
            df["ProjPoints"] = df["ProjPoints"].fillna(overall_median)
            df["ProjSource"] = df["ProjSource"].fillna("median")
            pos_means = df.groupby("Position")["ProjPoints"].transform("mean")
            pos_stds = df.groupby("Position")["ProjPoints"].transform("std").fillna(0.0)
            pos_stds = pos_stds.replace(0, 1.0)
            df["ProjZ"] = (df["ProjPoints"] - pos_means) / pos_stds
            denom = (1 + beta * df["ProjZ"]).clip(lower=0.1)
            scaled = df["RankOriginal"] / denom
            df["RankScaled"] = scaled
            df["Rank"] = (
                df["RankScaled"].rank(method="dense", ascending=True)
                .astype(int)
                .clip(lower=1)
            )
            # PosRank was computed from the pre-rescale Rank; recompute it from the
            # same RankScaled ordering so it stays consistent with the Rank column
            # callers actually use (previously PosRank silently went stale here).
            df["PosRank"] = df.groupby("Position")["RankScaled"].rank(method="min").astype(int)
        else:
            df["ProjPoints"] = None
            df["ProjZ"] = 0.0
            df["RankScaled"] = df["Rank"].astype(float)
    else:
        df["ProjPoints"] = None
        df["ProjZ"] = 0.0
        df["RankScaled"] = df["Rank"].astype(float)
        df["ProjSource"] = "none"
    return df

def build_lookups(df: pd.DataFrame):
    rank_by_name = {row["Name"]: int(row["Rank"]) for _, row in df.iterrows()}
    pos_by_name = {row["Name"]: row["Position"] for _, row in df.iterrows()}
    posrank_by_name = df.set_index("Name")["PosRank"].to_dict()
    proj_by_name = df.set_index("Name")["ProjPoints"].to_dict()
    return rank_by_name, pos_by_name, posrank_by_name, proj_by_name

_ROSTER_CACHE: Optional[Dict[str, Dict[str, list[str]]]] = None


def load_rosters() -> Dict:
    """Return a deep copy of the canonical roster data.

    If ESPN credentials are configured, fetch live rosters for the configured league;
    otherwise fall back to the static `rosters.py` data.
    """

    global _ROSTER_CACHE
    if _ROSTER_CACHE is not None:
        return copy.deepcopy(_ROSTER_CACHE)

    espn_configured = bool(
        os.getenv("ESPN_S2") or os.getenv("espn_s2")
    ) and bool(os.getenv("ESPN_SWID") or os.getenv("swid"))

    if espn_configured:
        try:
            rosters, _ = fetch_rosters_from_espn()
            _ROSTER_CACHE = rosters
            logger = logging.getLogger(__name__)
            logger.info("Loaded rosters from ESPN API for league %s", os.getenv("ESPN_LEAGUE_ID", "unknown"))
            return copy.deepcopy(rosters)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logging.getLogger(__name__).warning("Falling back to static rosters; ESPN fetch failed: %s", exc)

    _ROSTER_CACHE = copy.deepcopy(Fantasy_Rosters)
    return copy.deepcopy(_ROSTER_CACHE)
