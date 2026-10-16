"""Canonical NFL team codes and team-defense (D/ST) names.

Every data source spells teams differently:

- PFF / ESPN rosters: ``ARI``, ``BAL``, ``JAX``, ``LAR``; defenses as ``"Rams D/ST"``.
- FantasyPros rankings (site scrape and API): ``JAC`` instead of ``JAX``;
  defenses as ``"Los Angeles Rams"``.
- FantasyPros strength-of-schedule exports: ``ARZ``, ``BLT``, ``CLV``, ``HST``, ``LA``.

Anything that joins on a team code or a defense name should go through
:func:`canonical_team` / :func:`canonical_dst_name`, so a new source only has
to be mapped here once.

Canonical codes are the PFF/ESPN set. The canonical defense name is
``"<Nickname> D/ST"`` -- the form ESPN rosters (and so ``rosters.py``) use.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

# canonical code -> (city/region, nickname)
TEAMS: Dict[str, Tuple[str, str]] = {
    "ARI": ("Arizona", "Cardinals"),
    "ATL": ("Atlanta", "Falcons"),
    "BAL": ("Baltimore", "Ravens"),
    "BUF": ("Buffalo", "Bills"),
    "CAR": ("Carolina", "Panthers"),
    "CHI": ("Chicago", "Bears"),
    "CIN": ("Cincinnati", "Bengals"),
    "CLE": ("Cleveland", "Browns"),
    "DAL": ("Dallas", "Cowboys"),
    "DEN": ("Denver", "Broncos"),
    "DET": ("Detroit", "Lions"),
    "GB": ("Green Bay", "Packers"),
    "HOU": ("Houston", "Texans"),
    "IND": ("Indianapolis", "Colts"),
    "JAX": ("Jacksonville", "Jaguars"),
    "KC": ("Kansas City", "Chiefs"),
    "LAC": ("Los Angeles", "Chargers"),
    "LAR": ("Los Angeles", "Rams"),
    "LV": ("Las Vegas", "Raiders"),
    "MIA": ("Miami", "Dolphins"),
    "MIN": ("Minnesota", "Vikings"),
    "NE": ("New England", "Patriots"),
    "NO": ("New Orleans", "Saints"),
    "NYG": ("New York", "Giants"),
    "NYJ": ("New York", "Jets"),
    "PHI": ("Philadelphia", "Eagles"),
    "PIT": ("Pittsburgh", "Steelers"),
    "SEA": ("Seattle", "Seahawks"),
    "SF": ("San Francisco", "49ers"),
    "TB": ("Tampa Bay", "Buccaneers"),
    "TEN": ("Tennessee", "Titans"),
    "WAS": ("Washington", "Commanders"),
}

# Non-canonical codes seen in the wild -> canonical code.
_CODE_ALIASES: Dict[str, str] = {
    "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "JAC": "JAX",
    "LA": "LAR", "STL": "LAR", "LVR": "LV", "OAK": "LV", "SD": "LAC",
    "WSH": "WAS", "GNB": "GB", "KAN": "KC", "NWE": "NE", "NOR": "NO",
    "SFO": "SF", "TAM": "TB",
}

_BY_NICKNAME: Dict[str, str] = {nick.lower(): code for code, (_, nick) in TEAMS.items()}
# Only cities with a single team ("Los Angeles"/"New York" are ambiguous and excluded).
_CITY_COUNTS: Dict[str, int] = {}
for _city, _ in TEAMS.values():
    _CITY_COUNTS[_city.lower()] = _CITY_COUNTS.get(_city.lower(), 0) + 1
_BY_CITY: Dict[str, str] = {
    city.lower(): code for code, (city, _) in TEAMS.items() if _CITY_COUNTS[city.lower()] == 1
}
_DST_MARKERS = re.compile(r"\b(d/st|dst|defense)\b", re.IGNORECASE)


def canonical_team(code: Optional[str]) -> Optional[str]:
    """Return the canonical team code; unknown values (FA, blank) pass through unchanged."""
    if code is None:
        return None
    raw = str(code).strip().upper()
    if not raw or raw == "NAN":
        return code
    if raw in TEAMS:
        return raw
    return _CODE_ALIASES.get(raw, raw)


def dst_name_for_team(code: Optional[str]) -> Optional[str]:
    team = canonical_team(code)
    return f"{TEAMS[team][1]} D/ST" if team in TEAMS else None


def canonical_dst_name(name: Optional[str], team: Optional[str] = None) -> Optional[str]:
    """Map any spelling of a team defense to ``"<Nickname> D/ST"``.

    Prefers the team code when it resolves; otherwise parses the name
    ("San Francisco 49ers", "49ers DST", "SF D/ST", "Seattle Defense").
    Returns None when nothing resolves, so callers keep the original.
    """
    by_team = dst_name_for_team(team) if team is not None else None
    if by_team:
        return by_team
    if not name:
        return None
    stripped = re.sub(r"\s+", " ", _DST_MARKERS.sub(" ", str(name))).strip()
    if not stripped:
        return None
    code = _BY_NICKNAME.get(stripped.split()[-1].lower())
    if code is None:
        code = _BY_CITY.get(stripped.lower())
    if code is None:
        candidate = canonical_team(stripped)
        code = candidate if candidate in TEAMS else None
    return f"{TEAMS[code][1]} D/ST" if code else None


def looks_like_dst(name: Optional[str]) -> bool:
    """True for 'X D/ST', 'X DST', 'X Defense', or a bare full team name like 'Houston Texans'."""
    if not name:
        return False
    text = str(name).strip()
    if _DST_MARKERS.search(text):
        return True
    lowered = text.lower()
    return any(lowered == f"{city} {nick}".lower() for city, nick in TEAMS.values())
