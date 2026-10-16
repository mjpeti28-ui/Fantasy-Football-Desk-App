"""Points-based player and team valuation (experimental, 2026-09-23).

Replaces "VOR + combined z-score" with one currency: **expected fantasy points
your starting lineup will score over the rest of the season**, which is exactly
what this league rewards (playoff seeding is by total points).

Pieces:
1. ``Forecaster`` -- per-player expected points per game = preseason prior
   shrunk toward observed points-per-game as games accumulate
   (prior weight ``k`` games, fit per position), times an availability rate
   (fit per position x current injury status). Bye weeks are 0.
2. ``team_value`` -- Monte Carlo over player availability; each simulated week
   the best legal lineup is started (bench players fill in automatically, so
   depth is valued by the points it actually supplies), and every team can
   stream the waiver wire (replacement = the best unrostered player at each
   position), so a traded-away roster spot is worth what you can pick up.
3. Values are split into regular season (drives seeding) and playoff weeks.

League size: the league was 12 teams 2021-2025 and is 10 teams in 2026. Nothing
here assumes a size -- lineup slots come from each season's settings and the
waiver wire from that season's actual rosters, so replacement level is deeper
in the 10-team league automatically.

Fitted parameters are learned from past seasons (see backtest/multi_season.py),
never from the season being evaluated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
HIST = ROOT / "history" / "actuals"

POSITIONS = ["QB", "RB", "WR", "TE", "K", "DST"]
SLOT_ELIGIBILITY = {
    "QB": ("QB",), "RB": ("RB",), "WR": ("WR",), "TE": ("TE",), "K": ("K",), "DST": ("DST",),
    "RB/WR": ("RB", "WR"), "WR/TE": ("WR", "TE"), "FLEX": ("RB", "WR", "TE"),
}
GAMES_PER_SEASON = 17
FANTASY_LAST_WEEK = 17

OUT_STATUSES = {"OUT", "INJURY_RESERVE", "SUSPENSION", "IR", "PUP"}
IFFY_STATUSES = {"QUESTIONABLE", "DOUBTFUL", "DAY_TO_DAY", "PROBABLE"}


# Availability buckets use only information that existed at the time:
#   "ir"          -- sitting in an IR lineup slot that week (historical roster data)
#   "missed_last" -- his NFL team played last week and he didn't
#   "active"      -- everyone else
# ESPN's injury_status field is NOT used for fitting: for past seasons the API
# returns today's status for every week (a look-ahead leak, confirmed 2026-09-23).
# For live use, ``live_status_overrides`` can zero out a player ESPN lists OUT this week.
STATUS_BUCKETS = ("active", "missed_last", "ir")


def status_bucket(missed_last: bool, slot: Optional[str]) -> str:
    if slot == "IR":
        return "ir"
    if missed_last:
        return "missed_last"
    return "active"


# ------------------------------------------------------------------ season data


@dataclass
class Season:
    season: int
    weekly: pd.DataFrame          # espn_id, name, position, week, points, played
    prior_total: Dict[int, float]  # espn_id -> ESPN season projection
    rosters: pd.DataFrame
    team_weekly: pd.DataFrame
    settings: dict
    byes: Dict[int, Optional[int]]  # espn_id -> bye week
    position: Dict[int, str]
    name: Dict[int, str]
    espn_weekly: Optional[pd.DataFrame] = None  # week, espn_id, espn_proj (ESPN's published weekly projection)

    @property
    def slots(self) -> List[str]:
        out: List[str] = []
        for slot, n in self.settings["lineup_slot_counts"].items():
            if slot in SLOT_ELIGIBILITY:
                out += [slot] * int(n)
        return out

    @property
    def reg_weeks(self) -> int:
        return int(self.settings.get("regular_season_weeks", 14))

    def roster_at(self, week: int) -> pd.DataFrame:
        return self.rosters[self.rosters["week"] == week]


def _infer_byes(weekly: pd.DataFrame) -> Dict[int, Optional[int]]:
    """A player's bye = the single mid-season week with no stat row (a row exists
    for every game his NFL team plays, played or not). Byes are published before
    the season, so using them is not look-ahead."""
    out: Dict[int, Optional[int]] = {}
    for pid, weeks in weekly.groupby("espn_id")["week"]:
        missing = [w for w in range(4, 15) if w not in set(weeks)]
        out[int(pid)] = missing[0] if len(missing) == 1 else None
    return out


@lru_cache(maxsize=None)
def load_season(season: int) -> Season:
    d = HIST / str(season)
    pw = pd.read_csv(d / "players_weekly.csv")
    weekly = pw[pw["week"] > 0].copy()
    weekly["played"] = weekly["played"].astype(str).str.lower().eq("true")
    prior = pw[pw["week"] == 0].groupby("espn_id")["points"].first().to_dict()
    ids = pw.drop_duplicates("espn_id")
    byes = _infer_byes(weekly)
    # In-progress season: future byes can't be inferred from missing rows, so use the
    # NFL schedule via each player's current pro team (correct for the live season;
    # only a fallback for past seasons, where a player's current team may differ).
    if (d / "byes.csv").exists():
        team_bye = pd.read_csv(d / "byes.csv").set_index("pro_team_id")["bye_week"].to_dict()
        pro = dict(zip(ids["espn_id"].astype(int), ids["pro_team_id"]))
        for pid in ids["espn_id"].astype(int):
            if byes.get(pid) is None and pro.get(pid) in team_bye:
                b = team_bye[pro[pid]]
                byes[pid] = int(b) if pd.notna(b) and int(b) > 0 else None
    rosters = pd.read_csv(d / "rosters_by_week.csv")
    espn_weekly = pd.read_csv(d / "espn_weekly_proj.csv") if (d / "espn_weekly_proj.csv").exists() else None
    prior_total = {int(k): float(v) for k, v in prior.items()}
    # Data-quality guards (found in the 2026-09-23 audit):
    # - 2023's stored ESPN *season* projections are mostly 0 (89% of players vs ~49% in
    #   other seasons, incl. Aaron Rodgers/Nick Chubb) and rank-correlate NEGATIVELY with
    #   results, so a season whose drafted players are mostly 0 is treated as having no
    #   season projection; the prior then falls back to ESPN's published weekly numbers.
    # - 2023 week 1 weekly projections are 0 for 86% of active rostered players (normal
    #   is 3-10%); a week like that is dropped rather than read as "everyone is out".
    draft_path = d / "draft.csv"
    if draft_path.exists() and draft_path.stat().st_size > 10:
        drafted = pd.read_csv(draft_path)["espn_id"].dropna().astype(int)
        vals = [prior_total.get(int(i), 0.0) for i in drafted]
        if vals and np.mean([x <= 0 for x in vals]) > 0.6:
            prior_total = {}
    if espn_weekly is not None:
        r = rosters[rosters.get("lineup_slot", pd.Series(index=rosters.index, dtype=object)) != "IR"][["week", "espn_id"]]
        m = r.merge(espn_weekly, on=["week", "espn_id"])
        m = m[[byes.get(int(i)) != int(w) for i, w in zip(m["espn_id"], m["week"])]]
        zero_share = m.groupby("week")["espn_proj"].apply(lambda x: (x == 0).mean())
        bad = [int(w) for w, z in zero_share.items() if z > 0.4]
        if bad:
            espn_weekly = espn_weekly[~espn_weekly["week"].isin(bad)]
    return Season(
        season=season,
        weekly=weekly,
        prior_total=prior_total,
        rosters=rosters,
        team_weekly=pd.read_csv(d / "team_weekly.csv"),
        settings=json.loads((d / "settings.json").read_text()),
        byes=byes,
        position=dict(zip(ids["espn_id"].astype(int), ids["position"])),
        name=dict(zip(ids["espn_id"].astype(int), ids["name"])),
        espn_weekly=espn_weekly,
    )


# ------------------------------------------------------------------ forecaster


@dataclass
class ForecastParams:
    # prior weight in games, per position (shrinkage of observed ppg toward preseason)
    k: Dict[str, float] = field(default_factory=lambda: {p: 6.0 for p in POSITIONS})
    # P(plays), per position x status bucket, for the next game ("now") and for
    # every game after it ("later") -- an OUT tag costs this week, not the season.
    availability: Dict[str, Dict[str, Dict[str, float]]] = field(
        default_factory=lambda: {
            p: {"now": {"active": 0.9, "missed_last": 0.5, "ir": 0.2},
                "later": {"active": 0.87, "missed_last": 0.75, "ir": 0.6}}
            for p in POSITIONS
        }
    )
    # multiplier on the preseason prior per position (priors run hot/cold by position)
    prior_scale: Dict[str, float] = field(default_factory=lambda: {p: 1.0 for p in POSITIONS})
    # weight on ESPN's published projection for the checkpoint week (as a per-game rate);
    # the rest goes to our prior+observed estimate. Fit per position.
    espn_weight: Dict[str, float] = field(default_factory=lambda: {p: 0.0 for p in POSITIONS})
    # P(plays next game) when ESPN projects him for 0 that week (ESPN knows he's out)
    espn_zero_now: Dict[str, float] = field(default_factory=lambda: {p: 0.05 for p in POSITIONS})
    # realized / forecast for the best unrostered players (winner's curse: the forecaster
    # is fit on rostered players and overrates the top of the waiver pool)
    waiver_scale: Dict[str, float] = field(default_factory=lambda: {p: 1.0 for p in POSITIONS})
    # P(plays) for every game after this one when ESPN projects a 0 in a game week,
    # split by whether someone rosters him ("rostered") or he's on waivers ("free_agent").
    # A 0 usually means more than a one-week absence, and a good player sitting on waivers
    # with a 0 is usually out long-term (ACL, released). Empty = use the status bucket.
    espn_zero_later: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # logistic availability model for QB/RB/WR/TE (avail_model.py); empty = bucket rates
    avail_model: Dict = field(default_factory=dict)
    # weight on FantasyPros rest-of-season expert consensus (rank-matched onto our own
    # per-game estimates at the position), per position; empty = off
    ecr_weight: Dict[str, float] = field(default_factory=dict)
    # sd of a player's persistent rest-of-season level (multiplier on est_ppg, mean 1) by
    # position -> role ("starter"/"backup"/"deep") -> checkpoint half ("early" <= wk7 / "late").
    # Used by team_value to value upside (a bench player only counts when he outplays the
    # alternatives). Empty = point forecasts only.
    level_sd: Dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"k": self.k, "availability": self.availability, "prior_scale": self.prior_scale,
                           "espn_weight": self.espn_weight, "espn_zero_now": self.espn_zero_now,
                           "waiver_scale": self.waiver_scale,
                           "espn_zero_later": self.espn_zero_later, "avail_model": self.avail_model,
                           "ecr_weight": self.ecr_weight, "level_sd": self.level_sd}, indent=2)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "ForecastParams":
        return cls.from_json((path or ROOT / "valuation_params.json").read_text())

    @classmethod
    def from_json(cls, text: str) -> "ForecastParams":
        d = json.loads(text)
        return cls(k=d["k"], availability=d["availability"], prior_scale=d.get("prior_scale", {p: 1.0 for p in POSITIONS}),
                   espn_weight=d.get("espn_weight", {p: 0.0 for p in POSITIONS}),
                   espn_zero_now=d.get("espn_zero_now", {p: 0.05 for p in POSITIONS}),
                   waiver_scale=d.get("waiver_scale", {p: 1.0 for p in POSITIONS}),
                   espn_zero_later=d.get("espn_zero_later", {}),
                   avail_model=d.get("avail_model", {}),
                   ecr_weight=d.get("ecr_weight", {}),
                   level_sd=d.get("level_sd", {}))


def player_table(s: Season, week: int, params: ForecastParams, *, live_injuries: bool = False,
                 rostered: Optional[Iterable[int]] = None, ecr: Optional[pd.Series] = None) -> pd.DataFrame:
    """Per-player forecast as of the start of ``week`` (uses only weeks < week).

    ``live_injuries=True`` (current season only) layers ESPN's real-time tags on top
    of the fitted buckets: OUT/SUSPENSION -> can't play this week; INJURY_RESERVE ->
    the IR bucket. Never use it on past seasons -- ESPN returns today's status there.
    """
    past = s.weekly[(s.weekly["week"] < week) & s.weekly["played"]]
    obs = past.groupby("espn_id")["points"].agg(["sum", "count"])
    known = set(s.prior_total) | set(obs.index)
    if s.espn_weekly is not None:
        known |= set(s.espn_weekly.loc[s.espn_weekly["week"] <= week, "espn_id"].astype(int))
    ids = sorted(set(s.position) & known)
    df = pd.DataFrame({"espn_id": ids})
    df["position"] = df["espn_id"].map(s.position)
    df = df[df["position"].isin(POSITIONS)].copy()
    df["prior_ppg"] = df["espn_id"].map(s.prior_total).fillna(np.nan) / GAMES_PER_SEASON
    if s.espn_weekly is not None:
        # fallback prior: ESPN's weekly projections published so far (weeks <= week; all
        # public before that week's games), for players without a season projection
        pub = s.espn_weekly[(s.espn_weekly["week"] <= week) & (s.espn_weekly["espn_proj"] > 0)]
        wk_mean = pub.groupby("espn_id")["espn_proj"].mean()
        missing = ~(df["prior_ppg"] > 0)
        df.loc[missing, "prior_ppg"] = df.loc[missing, "espn_id"].map(wk_mean)
    # players ESPN didn't project: low prior (25th pct of projected players at the position)
    fill = df.groupby("position")["prior_ppg"].transform(lambda x: x[x > 0].quantile(0.25) if (x > 0).any() else 0.0)
    df["prior_ppg"] = df["prior_ppg"].where(df["prior_ppg"] > 0, fill)
    df["prior_ppg"] *= df["position"].map(lambda p: params.prior_scale.get(p, 1.0))
    df["n"] = df["espn_id"].map(obs["count"]).fillna(0.0)
    df["obs_ppg"] = (df["espn_id"].map(obs["sum"]) / df["n"]).fillna(0.0)
    k = df["position"].map(lambda p: params.k.get(p, 6.0))
    df["ours_ppg"] = (k * df["prior_ppg"] + df["n"] * df["obs_ppg"]) / (k + df["n"])
    df["espn_now"] = np.nan
    df["bye"] = df["espn_id"].map(s.byes)
    if s.espn_weekly is not None:
        e = s.espn_weekly[s.espn_weekly["week"] == week].set_index("espn_id")["espn_proj"]
        prev = s.espn_weekly[s.espn_weekly["week"] == week - 1].set_index("espn_id")["espn_proj"]
        # on his bye ESPN publishes 0 for this week; use last week's (already published) number
        df["espn_now"] = [prev.get(i, np.nan) if b == week else e.get(i, np.nan) for i, b in zip(df["espn_id"], df["bye"])]
    w_espn = df["position"].map(lambda p: params.espn_weight.get(p, 0.0))
    use = df["espn_now"] > 0
    df["est_ppg"] = np.where(use, w_espn * df["espn_now"] + (1 - w_espn) * df["ours_ppg"], df["ours_ppg"])
    df["ecr_rank"] = np.nan
    if params.ecr_weight:
        if ecr is None:
            try:
                import nflverse_data as nd
                ecr = nd.ros_ecr(s.season, week - 1)  # last Friday's scrape: known when waivers run
            except Exception:
                ecr = None
        if ecr is not None:
            df = blend_ecr(df, ecr, params.ecr_weight)
    roster = s.roster_at(week)
    slot = dict(zip(roster["espn_id"], roster["lineup_slot"])) if "lineup_slot" in roster else {}
    last = s.weekly[s.weekly["week"] == week - 1]
    missed = set(last.loc[~last["played"], "espn_id"])
    df["status"] = [status_bucket(i in missed, slot.get(i)) for i in df["espn_id"]]
    df["avail_now"] = [params.availability[p]["now"][b] for p, b in zip(df["position"], df["status"])]
    df["avail"] = [params.availability[p]["later"][b] for p, b in zip(df["position"], df["status"])]
    zero = (df["espn_now"] == 0) & (df["bye"] != week)  # ESPN published a 0 for a game week: it knows he's out
    df.loc[zero, "avail_now"] = [params.espn_zero_now.get(p, 0.05) for p in df.loc[zero, "position"]]
    if params.espn_zero_later and zero.any():
        # ``rostered`` unknown -> treat everyone as rostered (the milder adjustment)
        rset = None if rostered is None else {int(i) for i in rostered}
        for idx in df.index[zero]:
            pos_ = df.at[idx, "position"]
            grp = "free_agent" if rset is not None and int(df.at[idx, "espn_id"]) not in rset else "rostered"
            val = params.espn_zero_later.get(pos_, {}).get(grp)
            if val is not None:
                df.at[idx, "avail"] = val
    if params.avail_model:
        df = _apply_avail_model(s, week, df, params.avail_model, rostered, params.espn_zero_now)
    if live_injuries and "injury_status" in roster:
        tag = dict(zip(roster["espn_id"], roster["injury_status"]))
        for idx, pid in zip(df.index, df["espn_id"]):
            t = tag.get(pid)
            t = t.upper() if isinstance(t, str) else ""
            pos_ = df.at[idx, "position"]
            if t == "INJURY_RESERVE":
                df.at[idx, "status"] = "ir"
                df.at[idx, "avail_now"] = params.availability[pos_]["now"]["ir"]
                ir_later = params.availability[pos_]["later"]["ir"]
                # keep the lower of the IR bucket and an ESPN-zero adjustment already applied
                df.at[idx, "avail"] = min(ir_later, df.at[idx, "avail"]) if zero.at[idx] else ir_later
            elif t in ("OUT", "SUSPENSION"):
                df.at[idx, "avail_now"] = 0.0
    df["bye"] = df["espn_id"].map(s.byes)
    df["from_week"] = week
    df["level_sd"] = 0.0
    df["learn_k"] = df["position"].map(lambda p: params.k.get(p, 6.0))
    if params.level_sd:
        df["level_sd"] = _level_sd(s, week, df, params.level_sd)
    df["name"] = df["espn_id"].map(s.name)
    return df.set_index("espn_id")


def player_role(s: "Season", week: int, ids) -> list:
    """Depth-chart role at ``week`` (starter / backup / deep) from the nflverse cache;
    "deep" when he's not on the chart or the cache is missing."""
    try:
        import avail_model as am
        cache = am._season_cache(s.season)
    except Exception:  # noqa: BLE001
        cache = None
    dmin = cache[1] if cache else {}
    out = []
    for i in ids:
        d = dmin.get((int(i), week), dmin.get((int(i), week - 1)))
        out.append("starter" if d == 1 else "backup" if d == 2 else "deep")
    return out


def _level_sd(s: "Season", week: int, df: pd.DataFrame, table: Dict) -> np.ndarray:
    half = "early" if week <= 7 else "late"
    roles = player_role(s, week, df["espn_id"])
    out = []
    for pos, role in zip(df["position"], roles):
        t = table.get(pos, {})
        t = t.get(role, t.get("-", {})) if isinstance(t, dict) else {}
        out.append(float(t.get(half, 0.0)) if isinstance(t, dict) else 0.0)
    return np.asarray(out)


def blend_ecr(df: pd.DataFrame, ecr: pd.Series, weights: Dict[str, float]) -> pd.DataFrame:
    """Blend expert consensus into est_ppg. Within each position, players with an ECR rank
    get the per-game estimate at that rank in our own sorted distribution (so the scale
    stays ours), then est = (1-w)*est + w*that. Players without an ECR rank are untouched."""
    df = df.copy()
    df["ecr_rank"] = df["espn_id"].map(ecr)
    for pos, w in weights.items():
        m = (df["position"] == pos) & df["ecr_rank"].notna() & (df["est_ppg"] > 0)
        if m.sum() < 2 or not w:
            continue
        vals = np.sort(df.loc[m, "est_ppg"].to_numpy())[::-1]
        order = df.loc[m, "ecr_rank"].rank(method="first").astype(int).to_numpy() - 1
        df.loc[m, "est_ppg"] = (1 - w) * df.loc[m, "est_ppg"].to_numpy() + w * vals[order]
    return df


def _apply_avail_model(s: "Season", week: int, df: pd.DataFrame, model: Dict,
                       rostered: Optional[Iterable[int]], params_zero_now: Dict[str, float]) -> pd.DataFrame:
    import avail_model as am

    sk = df["position"].isin(am.SKILL)
    if not sk.any():
        return df
    played = set(zip(s.weekly.loc[s.weekly["played"] & (s.weekly["week"] < week), "espn_id"],
                     s.weekly.loc[s.weekly["played"] & (s.weekly["week"] < week), "week"]))
    Fe = am.raw_features(s.season, week, df.loc[sk, ["espn_id", "position", "status", "espn_now", "bye"]], played, rostered)
    if Fe is None or Fe.empty:
        return df
    cp = am.curve_params(model, Fe).set_index("espn_id")
    idx = df.index[sk]
    ids = df.loc[idx, "espn_id"]
    # next game: the model, except when ESPN already projects a 0 -- there the fitted
    # espn_zero_now rate is better calibrated (backtest Brier .0708 vs .0728)
    zero_now = ids.map(Fe.set_index("espn_id")["espn0"]).to_numpy() > 0
    p_next = ids.map(cp["p_next"]).to_numpy()
    ez = df.loc[idx, "position"].map(lambda p: params_zero_now.get(p, 0.05)).to_numpy()
    df.loc[idx, "avail_now"] = np.where(zero_now, ez, p_next)
    df["av_a"] = np.nan
    df["av_b"] = np.nan
    df.loc[idx, "av_a"] = ids.map(cp["av_a"]).to_numpy()
    df.loc[idx, "av_b"] = ids.map(cp["av_b"]).to_numpy()
    # summary "later" rate (games 1-8 ahead) for display and for code that reads `avail`
    df.loc[idx, "avail"] = np.mean([1 / (1 + np.exp(-(df.loc[idx, "av_a"] + df.loc[idx, "av_b"] * np.log1p(h))))
                                   for h in range(1, 9)], axis=0)
    return df


def _avail_for_week(tbl: pd.DataFrame, week: int) -> np.ndarray:
    """Availability for ``week``: the "now" rate for the first game after the
    checkpoint, then the later-game rate (a horizon curve when the availability
    model is on, else the flat "later" bucket); 0 on the bye."""
    fw = tbl["from_week"].to_numpy()
    bye = tbl["bye"].to_numpy(dtype=float)
    nxt = np.where(fw == bye, fw + 1, fw)
    first = week == nxt
    later = tbl["avail"].to_numpy(dtype=float)
    if "av_a" in tbl:
        h = (week - fw) - ((bye >= fw) & (bye < week)).astype(int)  # game weeks ahead of the next one
        curve = 1 / (1 + np.exp(-(tbl["av_a"].to_numpy(dtype=float) + tbl["av_b"].to_numpy(dtype=float) * np.log1p(np.maximum(h, 0)))))
        later = np.where(np.isnan(curve), later, curve)
    a = np.where(first, tbl["avail_now"].to_numpy(dtype=float), later)
    return np.where(bye == week, 0.0, a)


def expected_ros_points(tbl: pd.DataFrame, start: int, end: int) -> pd.Series:
    total = sum(_avail_for_week(tbl, w) for w in range(start, end + 1))
    return tbl["est_ppg"] * pd.Series(total, index=tbl.index)


def hit_probability(tbl: pd.DataFrame, task: str = "start") -> pd.Series:
    """P(rest-of-season ppg clears the position's starter/boom line), or P(bust: < 0.7 x est_ppg),
    from est_ppg and position only (backtest/hit_prob.py; LOSO ECE ~.01 for start/boom). NaN
    outside QB/RB/WR/TE. Display next to est_ppg; never rank by it."""
    hp = json.loads((ROOT / "valuation_params.json").read_text()).get("hit_prob", {}).get("logit", {}).get(task, {})
    ab = [hp.get(p, [np.nan, np.nan]) for p in tbl["position"]]
    a = np.array([x[0] for x in ab], dtype=float)
    b = np.array([x[1] for x in ab], dtype=float)
    return pd.Series(1 / (1 + np.exp(-(a + b * tbl["est_ppg"].to_numpy(dtype=float)))), index=tbl.index)


# ------------------------------------------------------------------ lineups


def best_lineup_points(pts_by_pos: Dict[str, List[float]], slots: Sequence[str]) -> float:
    """Max points from a set of players (lists of weekly points by position, sorted desc)
    for the given slots. Single-position slots first, then flex slots; the two
    orderings of the RB/WR and WR/TE slots are both tried (greedy is otherwise exact
    for this nested eligibility structure)."""
    pools = {p: list(v) for p, v in pts_by_pos.items()}
    total = 0.0
    flex = []
    for slot in slots:
        elig = SLOT_ELIGIBILITY[slot]
        if len(elig) == 1:
            pool = pools.get(elig[0], [])
            if pool:
                total += pool.pop(0)
        else:
            flex.append(slot)
    if not flex:
        return total
    best = 0.0
    orders = {tuple(sorted(flex, key=lambda s: len(SLOT_ELIGIBILITY[s]))), tuple(sorted(flex, key=lambda s: (len(SLOT_ELIGIBILITY[s]), s != "WR/TE")))}
    for order in orders:
        pl = {p: list(v) for p, v in pools.items()}
        sub = 0.0
        for slot in order:
            cands = [(pl[p][0], p) for p in SLOT_ELIGIBILITY[slot] if pl.get(p)]
            if cands:
                v, p = max(cands)
                sub += v
                pl[p].pop(0)
        best = max(best, sub)
    return total + best


def best_lineup_points_bt(pools_bt: Dict[str, List[tuple]], slots: Sequence[str]) -> float:
    """Like best_lineup_points, but each entry is (belief, truth): the lineup is chosen on
    what the manager believes, and scored on what actually happens."""
    pools = {p: list(v) for p, v in pools_bt.items()}
    total = 0.0
    flex = []
    for slot in slots:
        elig = SLOT_ELIGIBILITY[slot]
        if len(elig) == 1:
            pool = pools.get(elig[0], [])
            if pool:
                total += pool.pop(0)[1]
        else:
            flex.append(slot)
    if not flex:
        return total
    best_b, best_t = -1.0, 0.0
    orders = {tuple(sorted(flex, key=lambda s: len(SLOT_ELIGIBILITY[s]))), tuple(sorted(flex, key=lambda s: (len(SLOT_ELIGIBILITY[s]), s != "WR/TE")))}
    for order in orders:
        pl = {p: list(v) for p, v in pools.items()}
        sb = st = 0.0
        for slot in order:
            cands = [(pl[p][0][0], pl[p][0][1], p) for p in SLOT_ELIGIBILITY[slot] if pl.get(p)]
            if cands:
                b, t, p = max(cands)
                sb += b
                st += t
                pl[p].pop(0)
        if sb > best_b:
            best_b, best_t = sb, st
    return total + best_t


def team_value(
    player_ids: Iterable[int],
    tbl: pd.DataFrame,
    slots: Sequence[str],
    weeks: Sequence[int],
    replacement: Dict[str, float],
    *,
    sims: int = 200,
    seed: int = 0,
    levels: bool = True,
) -> float:
    """Expected total starting-lineup points over ``weeks``.

    Each sim-week: each rostered player plays with prob ``avail`` (0 on his bye);
    the waiver-wire replacement at every position is always available to stream.

    With ``levels`` and a ``level_sd`` column in ``tbl``, each sim also draws every player's
    persistent rest-of-season level (lognormal, mean 1). The lineup is chosen on what a manager
    would believe after t weeks (the level shows through at rate t/(t+k), k = the forecaster's
    prior weight) and scored on the true level -- so a bench player's upside counts only as
    fast as it can be recognized.
    """
    ids = [i for i in player_ids if i in tbl.index]
    sub = tbl.loc[ids]
    pos = sub["position"].tolist()
    ppg = sub["est_ppg"].to_numpy()
    # Common random numbers keyed by player id: a player's simulated availability (and level)
    # is identical whichever roster he's on, so a trade's before/after difference isn't
    # swamped by simulation noise.
    u = (np.stack([np.random.default_rng([seed, int(i) & 0xFFFFFFFF]).random((FANTASY_LAST_WEEK + 2, sims)) for i in ids], axis=-1)
         if ids else np.zeros((FANTASY_LAST_WEEK + 2, sims, 0)))
    use_lv = levels and "level_sd" in sub and bool(ids) and float(np.nan_to_num(sub["level_sd"].to_numpy(dtype=float)).max()) > 0
    if use_lv:
        sd = np.nan_to_num(sub["level_sd"].to_numpy(dtype=float))
        s2 = np.log1p(sd ** 2)
        z = np.stack([np.random.default_rng([seed, int(i) & 0xFFFFFFFF, 7]).standard_normal(sims) for i in ids], axis=-1)
        L = np.exp(-s2 / 2 + np.sqrt(s2) * z)  # (sims, players), mean 1
        k = sub["learn_k"].to_numpy(dtype=float) if "learn_k" in sub else np.full(len(ids), 6.0)
        start = int(min(weeks)) if len(weeks) else 0
    total = 0.0
    for w in weeks:
        av = _avail_for_week(sub, w)
        draws = u[w] < av
        if not use_lv:
            for sidx in range(sims):
                pts: Dict[str, List[float]] = {p: [replacement.get(p, 0.0)] for p in POSITIONS}
                for j in np.flatnonzero(draws[sidx]):
                    pts[pos[j]].append(ppg[j])
                for p in pts:
                    pts[p].sort(reverse=True)
                total += best_lineup_points(pts, slots)
            continue
        t = max(w - start, 0)
        learn = t / (t + k)
        truth = ppg[None, :] * L
        belief = ppg[None, :] * (1 + (L - 1) * learn[None, :])
        for sidx in range(sims):
            ptsbt: Dict[str, List[tuple]] = {p: [(replacement.get(p, 0.0), replacement.get(p, 0.0))] for p in POSITIONS}
            for j in np.flatnonzero(draws[sidx]):
                ptsbt[pos[j]].append((belief[sidx, j], truth[sidx, j]))
            for p in ptsbt:
                ptsbt[p].sort(reverse=True)
            total += best_lineup_points_bt(ptsbt, slots)
    return total / sims


def waiver_replacement(tbl: pd.DataFrame, rostered: Iterable[int], depth: int = 2,
                       scale: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Expected ppg of the ``depth``-th best unrostered player per position
    (the best one tends to get claimed by someone else)."""
    free = tbl[~tbl.index.isin(set(rostered))]
    out = {}
    for p in POSITIONS:
        fp = free[(free["position"] == p) & (free["status"].isin(["active", "iffy"]))]
        vals = (fp["est_ppg"] * fp["avail"]).sort_values(ascending=False)
        out[p] = float(vals.iloc[min(depth - 1, len(vals) - 1)]) if len(vals) else 0.0
        out[p] *= (scale or {}).get(p, 1.0)
    return out
