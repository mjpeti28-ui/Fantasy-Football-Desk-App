"""Blend PFF and ESPN's weekly player projections for start/sit decisions.

Why this exists
----------------
PFF's ``projections.csv`` is a single rest-of-season point total, refreshed
periodically. ESPN publishes its own per-player, per-week projection natively
through its API (``statSourceId == 0`` in the box-score payload). Neither
source is uniformly better -- backtested against the real 2025 season (6
weeks, ~900 player-week observations, PFF's most-recently-available snapshot
at the time vs ESPN's native weekly projection, both evaluated against that
week's actual result), same time horizon for both:

    Pos   PFF Spearman   ESPN Spearman   PFF MAE   ESPN MAE
    QB    0.42            0.43            5.25      5.26
    RB    0.69            0.64            3.33      5.83
    WR    0.66            0.66            3.28      4.67
    TE    0.62            0.47            2.17      5.02

PFF's point ESTIMATES are consistently more accurate in absolute terms (lower
MAE) across every position. Its RANK ordering is comparably good to ESPN's at
WR, better at RB/TE, and roughly tied at QB. A linear blend
(``actual ~ w_pff * pff_per_game + w_espn * espn_proj``), fit on weeks 4-7 of
the 2025 season and validated out-of-sample on weeks 8-9, beat *both* single
sources on rank correlation pooled across positions (0.79 vs 0.64 for either
source alone) and matched or beat both on MAE -- except at TE, where blending
in ESPN's weight made the *held-out* prediction WORSE than just using PFF
alone (TE test set was thin, n=40, so treat that as a real but lower-confidence
result). The weights below reflect that: TE has no ESPN weight.

These weights are trained on one season's worth of data (6 weeks, one league's
worth of active players) -- treat them as a reasonable starting point, not a
permanent constant. Re-fit them once a full season of 2026 same-vintage data
is archived (see ``tools/fetch_pff.py`` cadence) rather than assuming last
season's fit still holds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from data import normalize_name, load_rankings
from espn_client import ESPNConfig, ESPNClient, DEFAULT_HISTORY_VIEWS

SEASON_END_WEEK = 17

# (w_pff, w_espn, intercept). Fit by OLS on 2025 weeks 4-7, validated on
# held-out weeks 8-9 (see module docstring). TE deliberately has w_espn=0 --
# blending tested worse than PFF alone out-of-sample at that position.
BLEND_WEIGHTS: Dict[str, tuple] = {
    "QB": (0.403, 0.478, 1.28),
    "RB": (0.554, 0.284, 2.56),
    "WR": (0.485, 0.358, 0.85),
    "TE": (1.000, 0.000, 0.00),  # PFF-only: validated blend underperformed here
}


@dataclass
class PlayerProjection:
    name: str
    position: str
    pff_ros_points: Optional[float]
    pff_per_game: Optional[float]
    espn_week_proj: Optional[float]
    blended: Optional[float]
    note: str = ""


def remaining_games(bye_week: Optional[int], current_week: int, season_end_week: int = SEASON_END_WEEK) -> int:
    """Games left in the regular season from `current_week` onward, inclusive,
    minus a bye if it hasn't happened yet. Mirrors the convention PFF's own
    archived snapshots used (see module docstring's backtest)."""
    remaining = max(0, season_end_week - current_week + 1)
    if bye_week is not None and bye_week >= current_week:
        remaining = max(0, remaining - 1)
    return remaining


def _current_week() -> int:
    """Best-effort current NFL week from ESPN's live league status."""
    config = ESPNConfig.from_environment()
    client = ESPNClient(config)
    payload = client.fetch_endpoint(view=["mStatus"]) if hasattr(client, "fetch_endpoint") else None
    if payload is None:
        raise RuntimeError("Could not determine current week from ESPN (client has no fetch_endpoint).")
    status = payload.get("status", {})
    week = status.get("latestScoringPeriod") or status.get("currentMatchupPeriod")
    if not week:
        raise RuntimeError("ESPN status payload did not include a current scoring period.")
    return int(week)


def fetch_espn_week_projections(week: int) -> Dict[str, Dict[str, Optional[float]]]:
    """Live pull of ESPN's own weekly projection for every rostered player in
    this league, for `week`. Returns {name_norm: {"proj": float|None, "position": str}}.
    Only covers players rostered on some team in this league -- fine for
    start/sit among your own roster, not a full free-agent projection feed."""
    config = ESPNConfig.from_environment()
    if not config.espn_s2 or not config.swid:
        raise RuntimeError("ESPN credentials not configured (ESPN_S2 / ESPN_SWID).")
    client = ESPNClient(config)
    payload = client.fetch_endpoint(view=list(DEFAULT_HISTORY_VIEWS), params={"scoringPeriodId": int(week)})

    pos_map = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "DST"}
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for team in payload.get("teams", []) or []:
        for entry in (team.get("roster") or {}).get("entries", []) or []:
            player = (entry.get("playerPoolEntry") or {}).get("player", {})
            name = player.get("fullName")
            if not name:
                continue
            proj = None
            for stat in player.get("stats", []) or []:
                if stat.get("scoringPeriodId") != week or stat.get("statSplitTypeId") != 1:
                    continue
                if stat.get("statSourceId") == 0:
                    proj = stat.get("appliedTotal")
            out[normalize_name(name)] = {
                "proj": float(proj) if proj is not None else None,
                "position": pos_map.get(player.get("defaultPositionId"), "?"),
            }
    return out


def blended_weekly_points(pff_per_game: Optional[float], espn_proj: Optional[float], position: str) -> Optional[float]:
    """Combine PFF's implied per-game rate and ESPN's native weekly projection
    using the position-specific weights validated in the module docstring.
    Falls back to whichever single source is available if the other is missing."""
    weights = BLEND_WEIGHTS.get(position)
    if weights is None:
        # No validated weights for this position (K/DST) -- just pass through PFF.
        return pff_per_game
    w_pff, w_espn, intercept = weights
    if pff_per_game is None and espn_proj is None:
        return None
    if pff_per_game is None:
        return espn_proj
    if espn_proj is None or w_espn == 0.0:
        return pff_per_game
    return w_pff * pff_per_game + w_espn * espn_proj + intercept


def compare_players(
    player_names: List[str],
    *,
    rankings_path: str = "PFF_rankings.csv",
    projections_path: str = "projections.csv",
    week: Optional[int] = None,
) -> pd.DataFrame:
    """Build a blended-projection comparison table for a list of player names,
    e.g. for deciding who to start. Pulls PFF's current rankings/projections
    (local files) and ESPN's live weekly projection (one API call) for the
    resolved current week unless `week` is given explicitly."""
    if week is None:
        week = _current_week()

    df = load_rankings(rankings_path, projections_path)
    df = df.set_index("name_norm")

    espn_wk = fetch_espn_week_projections(week)

    rows: List[PlayerProjection] = []
    for raw_name in player_names:
        key = normalize_name(raw_name)
        pff_row = df.loc[key] if key in df.index else None
        position = str(pff_row["Position"]).upper() if pff_row is not None else espn_wk.get(key, {}).get("position", "?")
        ros_points = float(pff_row["ProjPoints"]) if pff_row is not None and pd.notna(pff_row.get("ProjPoints")) else None
        bye_week = int(pff_row["ByeWeek"]) if pff_row is not None and "ByeWeek" in df.columns and pd.notna(pff_row.get("ByeWeek")) else None
        games_left = remaining_games(bye_week, week) if ros_points is not None else None
        per_game = (ros_points / games_left) if (ros_points is not None and games_left) else None

        espn_entry = espn_wk.get(key)
        espn_proj = espn_entry["proj"] if espn_entry else None

        blended = blended_weekly_points(per_game, espn_proj, position)
        note = ""
        if pff_row is None:
            note = "no PFF match"
        if espn_entry is None:
            note = (note + "; " if note else "") + "not on an ESPN roster this week"
        elif espn_proj is None:
            note = (note + "; " if note else "") + "ESPN has not published a projection for this week yet (using PFF only)"

        rows.append(PlayerProjection(raw_name, position, ros_points, per_game, espn_proj, blended, note))

    out = pd.DataFrame([r.__dict__ for r in rows])
    return out.sort_values("blended", ascending=False, na_position="last").reset_index(drop=True)
