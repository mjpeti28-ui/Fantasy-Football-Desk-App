"""Prediction ledger: record the model's forecasts every week, score them once results arrive.

write(season, week, tbl, team_values, odds) -> history/ledger/<season>/wk<NN>_players.csv, _teams.csv
score(season) -> per-ledger metrics, comparing each recorded forecast with what happened:
  next game   : P(plays) Brier + calibration; points rank corr / MAE (players who played)
  ROS so far  : forecast points for the weeks already played vs actual (rank corr, MAE,
                realized/predicted), so calibration drift in the 10-team league shows up early
  teams       : forecast lineup points for elapsed weeks vs actual team points (rank corr)
Everything reads the same files the tools use, so it works for past seasons too (backfill).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

import valuation as v

ROOT = Path(__file__).resolve().parent
LEDGER = ROOT / "history" / "ledger"
COLS = ["name", "position", "est_ppg", "avail_now", "avail", "bye", "status", "espn_now", "ecr_rank"]


def _dir(season: int) -> Path:
    d = LEDGER / str(season)
    d.mkdir(parents=True, exist_ok=True)
    return d


def write(season: int, week: int, tbl: pd.DataFrame, rostered_by_team: Dict[int, list],
          team_values: Optional[Dict[int, float]] = None, odds: Optional[pd.DataFrame] = None, tag: str = "",
          already_played: Optional[set] = None) -> Path:
    """``already_played``: players whose week-``week`` game was over when this was written
    (e.g. Thursday night before a Sunday write). Their week-``week`` forecast is blanked and
    flagged ``pre_played`` so scoring skips that game instead of crediting hindsight."""
    t = tbl[[c for c in COLS if c in tbl.columns]].copy()
    # expected points per future week (week-by-week, so partial-season scoring is exact)
    for w in range(week, v.FANTASY_LAST_WEEK + 1):
        xa = v._avail_for_week(tbl, w)
        t[f"xp_w{w}"] = tbl["est_ppg"] * xa
        t[f"xa_w{w}"] = xa
    t["pre_played"] = t.index.isin(list(already_played or []))
    t.loc[t["pre_played"], [f"xp_w{week}", f"xa_w{week}"]] = np.nan
    owner = {i: tm for tm, ids in rostered_by_team.items() for i in ids}
    t["team_id"] = t.index.map(owner)
    t = t[t["team_id"].notna() | (t["est_ppg"] >= 3)]
    d = _dir(season)
    f = d / f"wk{week:02d}{tag}_players.csv"
    t.to_csv(f)
    if team_values is not None:
        tv = pd.DataFrame({"team_id": list(team_values), "ros_value": list(team_values.values())})
        if odds is not None:
            tv = tv.merge(odds.reset_index().rename(columns={"index": "team_id"}), on="team_id", how="left")
        tv.to_csv(d / f"wk{week:02d}{tag}_teams.csv", index=False)
    return f


def _spearman(a, b):
    a, b = pd.Series(list(a)), pd.Series(list(b))
    return float(a.rank().corr(b.rank())) if len(a) > 2 else np.nan


def completed_weeks(season: int, weeks) -> list:
    """Weeks whose whole NFL slate is final (nflverse schedule scores), so a Thursday game
    doesn't make a week look scoreable. Falls back to ``weeks`` without a schedule file."""
    f = ROOT / "history" / "nflverse" / "schedules.csv"
    if not f.exists():
        return sorted(weeks)
    sc = pd.read_csv(f)
    sc = sc[sc["season"] == season]
    if sc.empty or "home_score" not in sc:
        return sorted(weeks)
    fin = sc.groupby("week")["home_score"].apply(lambda x: x.notna().all())
    return sorted(w for w in weeks if bool(fin.get(w, False)))


def score(season: int, tag: str = "") -> pd.DataFrame:
    s = v.load_season(season)
    wk = s.weekly[(s.weekly["week"] >= 1) & (s.weekly["week"] <= v.FANTASY_LAST_WEEK)]
    wk = wk[wk["week"].isin(completed_weeks(season, wk["week"].unique()))]
    done = sorted(wk["week"].unique())
    if not done:
        return pd.DataFrame()
    last_done = int(max(done))
    played = wk[wk["played"]].set_index(["espn_id", "week"])["points"]
    out = []
    for f in sorted(_dir(season).glob(f"wk*{tag}_players.csv")):
        W = int(f.name[2:4])
        if W > last_done:
            continue
        t = pd.read_csv(f, index_col=0)
        pre = t["pre_played"].astype(bool).to_numpy() if "pre_played" in t else np.zeros(len(t), bool)
        rec = {"ledger_week": W, "scored_through": last_done}
        # next game (week W, or W+1 if on bye)
        nxt = np.where(t["bye"] == W, W + 1, W)
        y = np.array([(i, w) in played.index for i, w in zip(t.index, nxt)], dtype=float)
        ok = (nxt <= last_done) & ~pre
        p = t["avail_now"].to_numpy(float)
        rec["n_players"] = int(ok.sum())
        rec["next_brier"] = float(np.mean((p[ok] - y[ok]) ** 2))
        rec["next_pred_play"] = float(p[ok].mean())
        rec["next_actual_play"] = float(y[ok].mean())
        pts = np.array([played.get((i, w), np.nan) for i, w in zip(t.index, nxt)])
        m = ok & ~np.isnan(pts) & t["position"].isin(["QB", "RB", "WR", "TE"]).to_numpy()
        tt = t[m].assign(actual=pts[m])
        rec["next_pts_rank_corr"] = float(np.nanmean([_spearman(g["est_ppg"], g["actual"]) for _, g in tt.groupby("position") if len(g) > 5]))
        rec["next_pts_mae"] = float((tt["est_ppg"] - tt["actual"]).abs().mean())
        # ROS so far: sum expected vs actual over weeks W..last_done
        weeks = [w for w in range(W, last_done + 1) if f"xp_w{w}" in t.columns]
        if weeks:
            xp = t[[f"xp_w{w}" for w in weeks]].sum(axis=1)  # NaN (pre-played) weeks count 0
            act = pd.Series([sum(played.get((i, w), 0.0) for w in weeks if not (p_ and w == W))
                             for i, p_ in zip(t.index, pre)], index=t.index)
            sk = t["position"].isin(["QB", "RB", "WR", "TE"])
            rec["ros_weeks"] = len(weeks)
            rec["ros_rank_corr"] = float(np.nanmean([_spearman(xp[g.index], act[g.index]) for _, g in t[sk].groupby("position") if len(g) > 5]))
            rec["ros_mae"] = float((xp[sk] - act[sk]).abs().mean())
            rec["ros_realized_over_pred"] = float(act[sk].sum() / xp[sk].sum()) if xp[sk].sum() else np.nan
            # teams: forecast lineup points need the team file; use roster sums as a proxy check
            tf = f.with_name(f.name.replace("_players", "_teams"))
            if tf.exists() and s.team_weekly is not None and len(s.team_weekly):
                tv = pd.read_csv(tf)
                tw = s.team_weekly[(s.team_weekly["week"] >= W) & (s.team_weekly["week"] <= last_done)].groupby("team_id")["points"].sum()
                tv["actual"] = tv["team_id"].map(tw)
                if "ros_value" in tv and tv["actual"].notna().sum() > 3:
                    rec["team_rank_corr"] = _spearman(tv["ros_value"], tv["actual"])
        out.append(rec)
    res = pd.DataFrame(out)
    res.to_csv(_dir(season) / f"scores{tag}.csv", index=False)
    return res


# ------------------------------------------------------------------ projection sources
SKILL = ("QB", "RB", "WR", "TE")


def _fp_projection_pg(season: int, week: int, fp_file: Path) -> Optional[pd.Series]:
    """FantasyPros ROS projected points (API export) -> per game, espn_id-indexed. Their ROS
    runs through NFL week 18, so games = non-bye weeks from ``week`` to 18."""
    if not fp_file.exists():
        return None
    f = pd.read_csv(fp_file)
    if not {"fpid", "fantasyPoints"} <= set(f.columns):
        return None
    ids = pd.read_csv(ROOT / "history" / "nflverse" / "ids.csv", dtype={"fantasypros_id": str}).dropna(subset=["fantasypros_id", "espn_id"])
    m = ids.assign(k=ids["fantasypros_id"].str.replace(r"\.0$", "", regex=True)).drop_duplicates("k").set_index("k")["espn_id"]
    f["espn_id"] = f["fpid"].astype(str).str.replace(r"\.0$", "", regex=True).map(m)
    f = f.dropna(subset=["espn_id"])
    bye = pd.to_numeric(f.get("byeWeek"), errors="coerce")
    games = (18 - week + 1) - ((bye >= week) & (bye <= 18)).astype(int)
    pg = f["fantasyPoints"].to_numpy(float) / games.clip(lower=1).to_numpy(float)
    return pd.Series(pg, index=f["espn_id"].astype("int64")).groupby(level=0).max()


def write_sources(season: int, week: int, tbl: pd.DataFrame, fp_file: Optional[Path] = None, tag: str = "") -> Path:
    """Per-game rest-of-season rate from each source, so they can be scored against each other
    (and against simple averages) as results come in."""
    fp_file = fp_file or (ROOT / "FantasyPros_ROS_projections.csv")
    t = tbl[tbl["position"].isin(SKILL) & (tbl["est_ppg"] > 0)].copy()
    out = pd.DataFrame(index=t.index)
    out["position"] = t["position"]
    out["model"] = t["est_ppg"]
    out["ours"] = t["ours_ppg"]
    out["espn"] = t["espn_now"].where(t["espn_now"] > 0, t["ours_ppg"])
    ab = (out["ours"] + out["espn"]) / 2
    out["experts_rank"] = np.nan
    if "ecr_rank" in t:
        for pos, g in t[t["ecr_rank"].notna()].groupby("position"):
            vals = np.sort(ab[g.index].to_numpy())[::-1]
            order = g["ecr_rank"].rank(method="first").astype(int).to_numpy() - 1
            out.loc[g.index, "experts_rank"] = vals[order]
    fp = _fp_projection_pg(season, week, fp_file)
    out["fp_projection"] = out.index.map(fp) if fp is not None else np.nan
    out["avg3"] = out[["ours", "espn", "experts_rank"]].mean(axis=1)
    out["avg4"] = out[["ours", "espn", "experts_rank", "fp_projection"]].mean(axis=1)
    out["avg_espn_fp"] = out[["espn", "fp_projection"]].mean(axis=1)
    f = _dir(season) / f"wk{week:02d}{tag}_sources.csv"
    out.to_csv(f)
    return f


SOURCE_COLS = ["model", "ours", "espn", "experts_rank", "fp_projection", "avg3", "avg4", "avg_espn_fp"]


def score_sources(season: int, tag: str = "", min_games: int = 2) -> pd.DataFrame:
    """Each source's per-game rate vs realized points per game played since the ledger week.
    Compared on the same players (those every compared source covers)."""
    s = v.load_season(season)
    wk = s.weekly[(s.weekly["week"] >= 1) & (s.weekly["week"] <= v.FANTASY_LAST_WEEK) & s.weekly["played"]]
    wk = wk[wk["week"].isin(completed_weeks(season, wk["week"].unique()))]
    if wk.empty:
        return pd.DataFrame()
    if wk.empty:
        return pd.DataFrame()
    last_done = int(wk["week"].max())
    out = []
    for f in sorted(_dir(season).glob(f"wk*{tag}_sources.csv")):
        W = int(f.name[2:4])
        if W > last_done:
            continue
        t = pd.read_csv(f, index_col=0)
        fut = wk[(wk["week"] >= W) & (wk["week"] <= last_done)].groupby("espn_id")["points"].agg(["mean", "count"])
        t = t.join(fut, how="inner")
        t = t[t["count"] >= min_games]
        present = [c for c in SOURCE_COLS if c in t and t[c].notna().mean() > 0.5]
        common = t.dropna(subset=present)
        for c in present:
            rho = np.nanmean([_spearman(g[c], g["mean"]) for _, g in common.groupby("position") if len(g) > 5])
            out.append({"ledger_week": W, "through": last_done, "source": c, "n": len(common), "rank_corr": rho,
                        "mae": float((common[c] - common["mean"]).abs().mean()), "bias": float((common[c] - common["mean"]).mean())})
    res = pd.DataFrame(out)
    if len(res):
        res.to_csv(_dir(season) / f"sources_scores{tag}.csv", index=False)
    return res


def calibration_watch(season: int, tag: str = "", lam_pts: float = 4000.0, lam_games: float = 100.0,
                      band: float = 0.15) -> pd.DataFrame:
    """Position-level drift monitor (not applied to forecasts). From every ledger so far:
    realized / expected points and games played / expected games for the weeks already played,
    shrunk toward 1 (lam in expected points / expected games). Backtest (inseason_recal.py):
    auto-applying these hurt cross-position ranking of rostered players, so they are a flag --
    a factor outside 1 +/- ``band`` means the model is off for that position this season."""
    s = v.load_season(season)
    wk = s.weekly[(s.weekly["week"] >= 1) & (s.weekly["week"] <= v.FANTASY_LAST_WEEK)]
    wk = wk[wk["week"].isin(completed_weeks(season, wk["week"].unique()))]
    if wk.empty:
        return pd.DataFrame()
    last_done = int(wk.loc[wk["played"], "week"].max()) if wk["played"].any() else 0
    pts = wk.groupby(["espn_id", "week"])["points"].sum()
    played = set(zip(wk.loc[wk["played"], "espn_id"], wk.loc[wk["played"], "week"]))
    acc = {}
    for f in sorted(_dir(season).glob(f"wk*{tag}_players.csv")):
        W = int(f.name[2:4])
        t = pd.read_csv(f, index_col=0)
        for w in range(W, last_done + 1):
            if f"xp_w{w}" not in t:
                continue
            tw = t[~t["pre_played"].astype(bool)] if (w == W and "pre_played" in t) else t
            for pos, g in tw.groupby("position"):
                a = acc.setdefault(pos, [0.0, 0.0, 0.0, 0.0])
                a[0] += float(g[f"xp_w{w}"].sum())
                a[1] += float(sum(pts.get((i, w), 0.0) for i in g.index))
                if f"xa_w{w}" in g:
                    a[2] += float(g[f"xa_w{w}"].sum())
                    a[3] += float(sum((i, w) in played for i in g.index))
    rows = []
    for pos, (E, R, XG, PG) in acc.items():
        f_pts = 1 + (R / E - 1) * E / (E + lam_pts) if E > 0 else np.nan
        f_g = 1 + (PG / XG - 1) * XG / (XG + lam_games) if XG > 0 else np.nan
        rows.append({"position": pos, "expected_pts": round(E), "raw_pts_ratio": R / E if E else np.nan, "pts_factor": f_pts,
                     "expected_games": round(XG, 1), "raw_games_ratio": PG / XG if XG else np.nan, "games_factor": f_g,
                     "flag": bool((abs(f_pts - 1) > band) if pd.notna(f_pts) else False) or bool((abs(f_g - 1) > band) if pd.notna(f_g) else False)})
    return pd.DataFrame(rows).sort_values("position") if rows else pd.DataFrame()
