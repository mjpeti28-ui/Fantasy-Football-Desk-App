"""Weekly NFL usage data from nflverse (via nflreadpy), cached under history/nflverse/.

nflverse updates player stats during the season (usually the morning after games).
What we keep, per season (regular season only):
  player_stats.csv  targets, carries, attempts, shares, PPR points (gsis ids)
  opportunity.csv   expected fantasy points (xFP) from the ffopportunity model
  snaps.csv         offensive snap share (pfr ids)
  injuries.csv      official game-status reports as filed that week (no look-ahead)
plus ids.csv mapping gsis/pfr/espn ids.

`usage_table(season)` joins it all onto ESPN ids, one row per player-week.

CLI:  python3 nflverse_data.py 2021 2022 ... [--ids]
Needs `pip install --user nflreadpy` (pulls polars). Nothing else in the repo imports
nflreadpy, so the rest of the tool works without it as long as the cache exists.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "history" / "nflverse"
REG_SEASON_LAST = 18

STAT_COLS = ["player_id", "player_display_name", "position", "season", "week", "team",
             "attempts", "carries", "targets", "receptions", "target_share", "air_yards_share",
             "wopr", "receiving_air_yards", "fantasy_points_ppr"]
OPP_COLS = ["season", "week", "player_id", "posteam", "position", "pass_attempt", "rec_attempt",
            "rush_attempt", "total_fantasy_points", "total_fantasy_points_exp"]
SNAP_COLS = ["season", "week", "pfr_player_id", "player", "position", "team", "offense_snaps", "offense_pct"]
INJ_COLS = ["season", "week", "gsis_id", "full_name", "position", "team", "report_status",
            "practice_status", "report_primary_injury"]
ID_COLS = ["gsis_id", "pfr_id", "espn_id", "sleeper_id", "fantasypros_id", "name", "position",
           "team", "draft_year", "draft_round", "draft_pick", "birthdate"]


def _pd(df) -> pd.DataFrame:
    """polars -> pandas via CSV text, so pyarrow isn't needed."""
    if hasattr(df, "write_csv"):
        import io
        return pd.read_csv(io.StringIO(df.write_csv()), low_memory=False)
    return pd.DataFrame(df)


def fetch_season(season: int) -> dict:
    import nflreadpy as nfl  # optional dependency

    out = CACHE / str(season)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    jobs = {
        "player_stats": (lambda: nfl.load_player_stats([season]), STAT_COLS),
        "opportunity": (lambda: nfl.load_ff_opportunity([season]), OPP_COLS),
        "snaps": (lambda: nfl.load_snap_counts([season]), SNAP_COLS),
        "injuries": (lambda: nfl.load_injuries([season]), INJ_COLS),
    }
    for name, (load, cols) in jobs.items():
        try:
            d = _pd(load())
        except Exception as e:  # a feed may not exist yet early in a season
            counts[name] = f"unavailable ({type(e).__name__})"
            continue
        if "season_type" in d.columns:
            d = d[d["season_type"] == "REG"]
        d = d[[c for c in cols if c in d.columns]]
        d = d[pd.to_numeric(d["week"], errors="coerce") <= REG_SEASON_LAST]
        d.to_csv(out / f"{name}.csv", index=False)
        counts[name] = len(d)
    return counts


def fetch_ids() -> int:
    import nflreadpy as nfl

    d = _pd(nfl.load_ff_playerids())
    d = d[[c for c in ID_COLS if c in d.columns]]
    CACHE.mkdir(parents=True, exist_ok=True)
    d.to_csv(CACHE / "ids.csv", index=False)
    return len(d)


def _ids() -> pd.DataFrame:
    ids = pd.read_csv(CACHE / "ids.csv", dtype={"espn_id": "Int64"})
    return ids.dropna(subset=["espn_id"])


def usage_table(season: int) -> pd.DataFrame:
    """One row per (espn_id, week) with usage columns. Missing feeds -> NaN columns."""
    d = CACHE / str(season)
    ids = _ids()
    g2e = ids.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id").set_index("gsis_id")["espn_id"]
    p2e = ids.dropna(subset=["pfr_id"]).drop_duplicates("pfr_id").set_index("pfr_id")["espn_id"]

    st = pd.read_csv(d / "player_stats.csv")
    st["espn_id"] = st["player_id"].map(g2e)
    st = st.dropna(subset=["espn_id"])
    keep = ["espn_id", "week", "team", "attempts", "carries", "targets", "receptions", "target_share",
            "air_yards_share", "wopr", "fantasy_points_ppr"]
    u = st[keep]
    if (d / "opportunity.csv").exists():
        op = pd.read_csv(d / "opportunity.csv")
        op["espn_id"] = op["player_id"].map(g2e)
        op = op.dropna(subset=["espn_id"]).groupby(["espn_id", "week"], as_index=False)[
            ["total_fantasy_points_exp", "total_fantasy_points"]].sum()
        u = u.merge(op.rename(columns={"total_fantasy_points_exp": "xfp", "total_fantasy_points": "fp_opp"}),
                    on=["espn_id", "week"], how="outer")
    if (d / "snaps.csv").exists():
        sn = pd.read_csv(d / "snaps.csv")
        sn["espn_id"] = sn["pfr_player_id"].map(p2e)
        sn = sn.dropna(subset=["espn_id"]).groupby(["espn_id", "week"], as_index=False)["offense_pct"].max()
        u = u.merge(sn, on=["espn_id", "week"], how="outer")
    if (d / "injuries.csv").exists():
        inj = pd.read_csv(d / "injuries.csv")
        inj["espn_id"] = inj["gsis_id"].map(g2e)
        inj = inj.dropna(subset=["espn_id"]).drop_duplicates(["espn_id", "week"], keep="last")
        u = u.merge(inj[["espn_id", "week", "report_status", "practice_status"]], on=["espn_id", "week"], how="outer")
    u["espn_id"] = u["espn_id"].astype("int64")
    return u.sort_values(["espn_id", "week"]).reset_index(drop=True)


# ------------------------------------------------------------------ extras (9/23 evening)
# Expert consensus rankings (FantasyPros ECR archive, scraped weekly on Fridays), schedules
# with Vegas lines, and depth charts. Cached so experiments don't hit the network.

ECR_PAGES = {"weekly-qb", "weekly-rb", "weekly-wr", "weekly-te", "weekly-k", "weekly-dst",
             "redraft-qb", "redraft-rb", "redraft-wr", "redraft-te", "redraft-k", "redraft-overall"}
ECR_COLS = ["scrape_date", "page_type", "id", "player", "pos", "team", "ecr", "sd", "best", "worst",
            "player_owned_espn", "player_owned_avg"]
SCHED_COLS = ["game_id", "season", "game_type", "week", "gameday", "weekday", "gametime", "away_team",
              "home_team", "away_score", "home_score", "spread_line", "total_line", "roof", "wind", "temp"]


def fetch_ecr(since: str = "2021-08-01") -> int:
    import nflreadpy as nfl
    import polars as pl

    d = nfl.load_ff_rankings("all")
    d = d.filter(pl.col("page_type").is_in(list(ECR_PAGES)) & (pl.col("scrape_date") >= since))
    d = d.select([c for c in ECR_COLS if c in d.columns])
    out = _pd(d)
    out.to_csv(CACHE / "ecr.csv", index=False)
    return len(out)


def fetch_schedules(seasons) -> int:
    import nflreadpy as nfl

    d = _pd(nfl.load_schedules(list(seasons)))
    d = d[[c for c in SCHED_COLS if c in d.columns]]
    f = CACHE / "schedules.csv"
    if f.exists():  # merge: refreshing one season must not drop the others
        old = pd.read_csv(f)
        d = pd.concat([old[~old["season"].isin(list(seasons))], d], ignore_index=True)
    d.sort_values(["season", "week", "gameday"]).to_csv(f, index=False)
    return len(d)


def fetch_depth(season: int) -> int:
    """Normalized to one row per (week, team, player) with depth rank at his position.
    2021-2024: weekly charts. 2025+: daily snapshots -> the last snapshot before the
    Wednesday of each week (what a manager could see when waivers run)."""
    import nflreadpy as nfl

    d = _pd(nfl.load_depth_charts([season]))
    ids = pd.read_csv(CACHE / "ids.csv")
    g2e = ids.dropna(subset=["gsis_id", "espn_id"]).drop_duplicates("gsis_id").set_index("gsis_id")["espn_id"]
    if "week" in d.columns:
        d = d[(d["game_type"] == "REG") & (d["formation"] == "Offense") & d["position"].isin(["QB", "RB", "WR", "TE", "FB"])]
        d = d.assign(espn_id=d["gsis_id"].map(g2e), depth=pd.to_numeric(d["depth_team"], errors="coerce"))
        out = d[["week", "club_code", "espn_id", "gsis_id", "full_name", "position", "depth"]].rename(columns={"club_code": "team"})
    else:
        sched = pd.read_csv(CACHE / "schedules.csv")
        sched = sched[(sched["season"] == season) & (sched["game_type"] == "REG")]
        first = pd.to_datetime(sched.groupby("week")["gameday"].min())
        d = d[d["pos_abb"].isin(["QB", "RB", "WR", "TE", "FB"])].copy()
        d["dt"] = pd.to_datetime(d["dt"]).dt.tz_localize(None)
        snaps = sorted(d["dt"].unique())
        rows = []
        for wk, day in first.items():
            cutoff = day - pd.Timedelta(days=1)  # Wednesday before a Thursday opener
            prior = [s for s in snaps if s <= cutoff]
            if not prior:
                continue
            x = d[d["dt"] == prior[-1]]
            rows.append(x.assign(week=int(wk)))
        if not rows:
            return 0
        x = pd.concat(rows)
        x["espn_id"] = pd.to_numeric(x["espn_id"], errors="coerce")
        # WR slots (LWR/RWR/SWR) are separate depth lists; keep his best rank at any slot
        out = (x.groupby(["week", "team", "espn_id", "gsis_id", "player_name", "pos_abb"], dropna=False)["pos_rank"].min()
                 .reset_index().rename(columns={"player_name": "full_name", "pos_abb": "position", "pos_rank": "depth"}))
    (CACHE / str(season)).mkdir(parents=True, exist_ok=True)
    out.to_csv(CACHE / str(season) / "depth.csv", index=False)
    return len(out)


def ecr_table() -> pd.DataFrame:
    """ECR rows with espn_id and the NFL (season, week) they were published for.
    A Friday scrape belongs to the week whose Sunday games follow it."""
    e = pd.read_csv(CACHE / "ecr.csv", dtype={"id": str})
    ids = pd.read_csv(CACHE / "ids.csv", dtype={"fantasypros_id": str})
    ids = ids.dropna(subset=["fantasypros_id", "espn_id"])
    norm = lambda s: s.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    f2e = ids.assign(k=norm(ids["fantasypros_id"])).drop_duplicates("k").set_index("k")["espn_id"]
    e["espn_id"] = norm(e["id"]).map(f2e)
    s = pd.read_csv(CACHE / "schedules.csv")
    s = s[s["game_type"] == "REG"]
    s["gameday"] = pd.to_datetime(s["gameday"])
    wk = s.groupby(["season", "week"])["gameday"].agg(["min", "max"]).reset_index()
    e["scrape_date"] = pd.to_datetime(e["scrape_date"])
    out = []
    for _, r in wk.iterrows():
        m = (e["scrape_date"] >= r["min"] - pd.Timedelta(days=6)) & (e["scrape_date"] <= r["max"])
        out.append(e[m].assign(season=int(r["season"]), week=int(r["week"])))
    x = pd.concat(out)
    # if two scrapes fall in one week, keep the latest one before that week's Sunday
    x = x.sort_values("scrape_date").drop_duplicates(["season", "week", "page_type", "id"], keep="last")
    return x



_ECR_CACHE = {}


def ros_ecr(season: int, week: int, live_file: Path = ROOT / "FantasyPros_ROS_rankings.csv") -> "pd.Series | None":
    """Rest-of-season expert consensus as of the start of ``week``: espn_id -> position
    rank (lower = better). Uses the latest archived scrape published for a week <= ``week``
    (a Friday scrape belongs to that week's games; one week old is fine -- the backtest
    edge holds with a lagged scrape). For the season the repo's live FantasyPros file
    covers, that file is used when it's newer than the archive."""
    key = "table"
    if key not in _ECR_CACHE:
        try:
            e = ecr_table().dropna(subset=["espn_id"])
            e = e[e["page_type"].isin(["redraft-qb", "redraft-rb", "redraft-wr", "redraft-te"])]
            _ECR_CACHE[key] = e.assign(espn_id=e["espn_id"].astype("int64"))
        except FileNotFoundError:
            _ECR_CACHE[key] = None
    e = _ECR_CACHE[key]
    arch = None
    if e is not None:
        x = e[(e["season"] == season) & (e["week"] <= week)]
        if not x.empty:
            wk = x["week"].max()
            x = x[x["week"] == wk]
            arch = (x.groupby("espn_id")["ecr"].min(), x["scrape_date"].max())
    if live_file.exists() and (e is None or season >= int(e["season"].max())):
        import os
        mtime = pd.Timestamp(os.path.getmtime(live_file), unit="s")
        if arch is None or mtime > pd.Timestamp(arch[1]):
            f = pd.read_csv(live_file)
            if "FPID" in f.columns and "Position Rank" in f.columns:
                ids = pd.read_csv(CACHE / "ids.csv", dtype={"fantasypros_id": str}).dropna(subset=["fantasypros_id", "espn_id"])
                m = ids.assign(k=ids["fantasypros_id"].str.replace(r"\.0$", "", regex=True)).drop_duplicates("k").set_index("k")["espn_id"]
                f["espn_id"] = f["FPID"].astype(str).str.replace(r"\.0$", "", regex=True).map(m)
                f = f.dropna(subset=["espn_id"])
                return pd.Series(f["Position Rank"].to_numpy(float), index=f["espn_id"].astype("int64")).groupby(level=0).min()
    return None if arch is None else arch[0]


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--ids" in args or not (CACHE / "ids.csv").exists():
        print("ids", fetch_ids())
    if "--extras" in args:
        seasons = [int(a) for a in args if a.isdigit()] or list(range(2021, 2027))
        print("schedules", fetch_schedules(seasons))
        print("ecr", fetch_ecr())
        for y in seasons:
            print("depth", y, fetch_depth(y))
    else:
        for a in args:
            if a.isdigit():
                print(a, fetch_season(int(a)))
