"""Fit and validate valuation.py across every past season of this league.

Seasons 2021-2025 (12 teams). Everything reported is leave-one-season-out:
parameters are fit on the other seasons and scored on the held-out one, so
nothing is judged on data it was fit to. 2026 (10 teams) is too young to score
and is never used for fitting.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import valuation as v  # noqa: E402

SEASONS = [2021, 2022, 2023, 2024, 2025]
CHECKPOINTS = list(range(3, 11))
OUT = ROOT / "history" / "backtest" / "multi"


def spearman(a, b) -> float:
    return float(pd.Series(list(a)).rank().corr(pd.Series(list(b)).rank()))


# ------------------------------------------------------------------ player rows


def player_rows(seasons=SEASONS, checkpoints=CHECKPOINTS) -> pd.DataFrame:
    """One row per rostered player per checkpoint, with what happened afterward."""
    base = v.ForecastParams(k={p: 0.0 for p in v.POSITIONS})  # raw pieces; fitting recombines
    rows = []
    for yr in seasons:
        s = v.load_season(yr)
        wk = s.weekly
        for W in checkpoints:
            t = base and v.player_table(s, W, base)
            rost = s.roster_at(W)
            t = t[t.index.isin(set(rost["espn_id"]))]
            fut = wk[(wk["week"] >= W) & (wk["week"] <= v.FANTASY_LAST_WEEK)]
            played = fut[fut["played"]].groupby("espn_id")["points"].agg(["sum", "count"])
            fut_played = fut[fut["played"]].groupby("espn_id")["week"].apply(set).to_dict()
            for pid, r in t.iterrows():
                weeks = [w for w in range(W, v.FANTASY_LAST_WEEK + 1) if w != r["bye"]]
                games = len(weeks)
                fp = played.loc[pid] if pid in played.index else None
                pw = fut_played.get(pid, set())
                rows.append({
                    "season": yr, "week": W, "espn_id": pid, "position": r["position"], "status": r["status"],
                    "prior_ppg_raw": r["prior_ppg"], "obs_ppg": r["obs_ppg"], "n": r["n"],
                    "future_games": games,
                    "bye_now": r["bye"] == W,
                    "played_first": float(bool(weeks) and weeks[0] in pw),
                    "later_games": max(0, games - 1),
                    "played_later": float(sum(1 for w in weeks[1:] if w in pw)),
                    "future_played": float(fp["count"]) if fp is not None else 0.0,
                    "future_points": float(fp["sum"]) if fp is not None else 0.0,
                })
    df = pd.DataFrame(rows)
    proj = []
    for yr in seasons:
        f = v.HIST / str(yr) / "espn_weekly_proj.csv"
        if f.exists():
            proj.append(pd.read_csv(f).assign(season=yr))
    if proj:
        P = pd.concat(proj).rename(columns={"espn_proj": "espn_now"})
        df = df.merge(P, on=["season", "week", "espn_id"], how="left")
        prev = P.assign(week=P["week"] + 1).rename(columns={"espn_now": "espn_prev"})
        df = df.merge(prev, on=["season", "week", "espn_id"], how="left")
        on_bye = df["bye_now"].astype(bool)
        df.loc[on_bye, "espn_now"] = df.loc[on_bye, "espn_prev"]
        df = df.drop(columns=["espn_prev"])
    df["future_ppg"] = np.where(df["future_played"] > 0, df["future_points"] / df["future_played"].clip(lower=1), np.nan)
    return df


# ------------------------------------------------------------------ fitting


import os as _os
EQUAL_BLEND = _os.environ.get("EQUAL_BLEND", "1") == "1"  # adopted 9/25 (simple average beat fitted weights); EQUAL_BLEND=0 for the old fitted blend
USE_LEVELS = False  # level uncertainty tested 9/25 and not adopted (see findings); True to fit it for A/B runs

K_GRID = [0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 50, 1e6]


def fit_params(rows: pd.DataFrame) -> v.ForecastParams:
    p = v.ForecastParams()
    for pos in v.POSITIONS:
        g = rows[rows["position"] == pos]
        gp = g.dropna(subset=["future_ppg"])
        w = gp["future_played"]
        # prior calibration: how preseason ppg compares with what players actually did
        scale = float((gp["future_ppg"] * w).sum() / (gp["prior_ppg_raw"] * w).sum()) if len(gp) else 1.0
        p.prior_scale[pos] = round(scale, 3)
        prior = gp["prior_ppg_raw"] * scale
        best = None
        for k in K_GRID:
            est = (k * prior + gp["n"] * gp["obs_ppg"]) / (k + gp["n"])
            mae = float((np.abs(est - gp["future_ppg"]) * w).sum() / w.sum())
            if best is None or mae < best[0]:
                best = (mae, k)
        p.k[pos] = best[1]
        # blend with ESPN's published projection for the checkpoint week
        if "espn_now" in gp:
            ours = (best[1] * prior + gp["n"] * gp["obs_ppg"]) / (best[1] + gp["n"])
            e = gp["espn_now"]
            best_w = None
            for wgt in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                est = np.where(e > 0, wgt * e + (1 - wgt) * ours, ours)
                mae = float((np.abs(est - gp["future_ppg"]) * w).sum() / w.sum())
                if best_w is None or mae < best_w[0]:
                    best_w = (mae, wgt)
            p.espn_weight[pos] = best_w[1]
            gz = g[(g["espn_now"] == 0) & ~g["bye_now"].astype(bool)]
            p.espn_zero_now[pos] = round(float((gz["played_first"].sum() + 1) / (len(gz) + 20)), 3) if len(gz) else 0.05
        p.availability[pos] = {"now": {}, "later": {}}
        for b in v.STATUS_BUCKETS:
            gb = g[g["status"] == b]
            # a few pseudo-observations at the position's overall rate keep thin cells sane
            prior_now = float(g["played_first"].mean()) if len(g) else 0.8
            prior_later = float(g["played_later"].sum() / max(1, g["later_games"].sum()))
            m = 20.0
            p.availability[pos]["now"][b] = round(float((gb["played_first"].sum() + m * prior_now) / (len(gb) + m)), 3)
            p.availability[pos]["later"][b] = round(float((gb["played_later"].sum() + m * prior_later) / (gb["later_games"].sum() + m)), 3)
    return p


def fit_waiver_scale(seasons, p: v.ForecastParams, checkpoints=(4, 6, 8, 10), top: int = 5) -> Dict[str, float]:
    """realized / forecast rest-of-season points for the best unrostered players at each
    position (the forecaster is fit on rostered players and overrates the waiver pool)."""
    pred = {pos: 0.0 for pos in v.POSITIONS}
    real = {pos: 0.0 for pos in v.POSITIONS}
    for yr in seasons:
        s = v.load_season(yr)
        for W in checkpoints:
            rost = set(s.roster_at(W)["espn_id"])
            t = v.player_table(s, W, p, rostered=rost)
            fut = s.weekly[(s.weekly["week"] >= W) & (s.weekly["week"] <= v.FANTASY_LAST_WEEK)]
            tot = fut.groupby("espn_id")["points"].sum()
            exp = v.expected_ros_points(t, W, v.FANTASY_LAST_WEEK)
            for pos in v.POSITIONS:
                fa = exp[(~t.index.isin(rost)) & (t["position"] == pos)].sort_values(ascending=False).head(top)
                pred[pos] += float(fa.sum())
                real[pos] += float(sum(tot.get(i, 0.0) for i in fa.index))
    return {pos: round(min(1.2, real[pos] / pred[pos]), 3) if pred[pos] > 0 else 1.0 for pos in v.POSITIONS}


def fit_espn_zero_later(seasons, p: v.ForecastParams, checkpoints=range(3, 14),
                        min_ppg: float = 3.0, shrink: float = 30.0) -> Dict[str, Dict[str, float]]:
    """Effective availability after this week for players ESPN projects at 0 in a game week:
    realized points from W on / (v2 per-game estimate x remaining non-bye games after W),
    split by rostered (going into W) vs free agent. Players with estimates under
    ``min_ppg`` are dropped (deep backups ESPN zeroes because they have no role).
    Shrunk toward the rostered active rate with ``shrink`` pseudo-games."""
    num = {}
    den = {}
    for yr in seasons:
        s = v.load_season(yr)
        for W in checkpoints:
            rost = set(s.roster_at(W - 1)["espn_id"])
            try:
                t = v.player_table(s, W, p)
            except Exception:
                continue
            zero = (t["espn_now"] == 0) & (t["bye"] != W) & (t["est_ppg"] >= min_ppg)
            t = t[zero]
            if t.empty:
                continue
            fut = s.weekly[(s.weekly["week"] >= W) & (s.weekly["week"] <= v.FANTASY_LAST_WEEK)]
            tot = fut.groupby("espn_id")["points"].sum()
            later = (v.FANTASY_LAST_WEEK - W) - ((t["bye"] > W) & (t["bye"] <= v.FANTASY_LAST_WEEK)).astype(int)
            for pid, r in t.iterrows():
                g = int(later.loc[pid])
                if g < 2:
                    continue
                key = (r["position"], "rostered" if pid in rost else "free_agent")
                num[key] = num.get(key, 0.0) + float(tot.get(pid, 0.0))
                den[key] = den.get(key, 0.0) + float(r["est_ppg"]) * g
    out: Dict[str, Dict[str, float]] = {}
    for pos in v.POSITIONS:
        base = p.availability[pos]["later"]["active"]
        for grp in ("rostered", "free_agent"):
            n, d = num.get((pos, grp), 0.0), den.get((pos, grp), 0.0)
            # pseudo-observations: `shrink` games of a 10-ppg player at the active rate
            val = (n + shrink * 10.0 * base) / (d + shrink * 10.0)
            out.setdefault(pos, {})[grp] = round(float(min(val, base)), 3)
    return out


def fit_ecr_weight(seasons, grid=np.round(np.arange(0, 1.01, 0.1), 2)) -> Dict[str, float]:
    """Per-position weight on rank-matched FantasyPros ROS consensus, chosen to maximize
    within-week rank corr of rest-of-season ppg (players with >= 3 later games), using the
    usage_rows_* checkpoints (v2 estimates fit without that season)."""
    import nflverse_data as nd
    from backtest import ecr_signal as es

    E = nd.ecr_table().dropna(subset=["espn_id"])
    E["espn_id"] = E["espn_id"].astype("int64")
    x = es.ros_rows(E)
    x = x[x["season"].isin(list(seasons)) & (x["est_ppg"] > 0)].copy()

    x["ecr_ppg"] = np.nan
    for _, g in x.groupby(["season", "week", "position"]):
        vals = np.sort(g["est_ppg"].to_numpy())[::-1]
        order = g["ecr"].rank(method="first").astype(int).to_numpy() - 1
        x.loc[g.index, "ecr_ppg"] = vals[order]
    x = x[x["future_played"] >= 3]
    out = {}
    for pos, g in x.groupby("position"):
        scores = []
        for w in grid:
            pred = (1 - w) * g["est_ppg"] + w * g["ecr_ppg"]
            scores.append(np.mean([spearman(pred[h.index], h["future_ppg"]) for _, h in g.groupby(["season", "week"]) if len(h) > 5]))
        out[pos] = float(grid[int(np.argmax(scores))])
    return out


def fit_level_sd(seasons) -> Dict:
    """Persistent rest-of-season level sd by position -> role -> half, from split-half
    covariance of per-game ratios (backtest/level_uncertainty.py; needs level_rows.pkl)."""
    from backtest import level_uncertainty as lu

    x = pd.read_pickle(OUT / "level_rows.pkl")
    x = x[x["season"].isin(list(seasons))]
    P = lu.group_params(x, ["position", "role", "half"])
    out: Dict = {}
    for (pos, role, half), r in P.iterrows():
        role = {"deep/off": "deep"}.get(role, role)
        out.setdefault(pos, {}).setdefault(role, {})[half] = round(float(r["level_sd"]), 3)
    return out


def fit_all(rows: pd.DataFrame, seasons) -> v.ForecastParams:
    p = fit_params(rows[rows["season"].isin(seasons)])
    try:
        import avail_model as am
        p.avail_model = am.fit_from_backtest_cache(seasons)
    except FileNotFoundError:  # run backtest/availability.py first to build the cache
        p.avail_model = {}
    try:
        p.ecr_weight = fit_ecr_weight(seasons)
    except FileNotFoundError:  # no ECR archive cached (nflverse_data.py --extras)
        p.ecr_weight = {}
    if EQUAL_BLEND:  # simple average of the three sources (ours, ESPN weekly, experts) for QB/RB/WR/TE
        for pos in ("QB", "RB", "WR", "TE"):
            p.espn_weight[pos] = 0.5
        p.ecr_weight = {pos: round(1 / 3, 4) for pos in ("QB", "RB", "WR", "TE")}
    try:
        p.level_sd = fit_level_sd(seasons) if USE_LEVELS else {}
    except FileNotFoundError:
        p.level_sd = {}
    p.espn_zero_later = fit_espn_zero_later(seasons, p)
    p.waiver_scale = fit_waiver_scale(seasons, p)
    return p


def apply(rows: pd.DataFrame, p: v.ForecastParams, mode: str = "fitted") -> pd.Series:
    """Predicted rest-of-season points for each row under a forecasting mode."""
    scale = rows["position"].map(p.prior_scale)
    prior = rows["prior_ppg_raw"] * scale
    if mode == "prior_only":
        est, avail = rows["prior_ppg_raw"], pd.Series(1.0, index=rows.index)
    elif mode == "observed_only":
        est = np.where(rows["n"] > 0, rows["obs_ppg"], rows["prior_ppg_raw"])
        avail = pd.Series(1.0, index=rows.index)
    else:
        k = rows["position"].map(p.k)
        est = (k * prior + rows["n"] * rows["obs_ppg"]) / (k + rows["n"])
        if mode == "fitted" and "espn_now" in rows:
            we = rows["position"].map(p.espn_weight)
            est = np.where(rows["espn_now"] > 0, we * rows["espn_now"] + (1 - we) * est, est)
        now = pd.Series([p.availability[a]["now"][b] for a, b in zip(rows["position"], rows["status"])], index=rows.index)
        if mode == "fitted" and "espn_now" in rows:
            out_now = (rows["espn_now"] == 0) & ~rows["bye_now"].astype(bool)
            now = now.where(~out_now, rows["position"].map(p.espn_zero_now))
        later = pd.Series([p.availability[a]["later"][b] for a, b in zip(rows["position"], rows["status"])], index=rows.index)
        return pd.Series(est, index=rows.index) * (now * (rows["future_games"] > 0) + later * rows["later_games"])
    return pd.Series(est, index=rows.index) * avail * rows["future_games"]


def loso_forecast_report(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for held in SEASONS:
        train, test = rows[rows["season"] != held], rows[rows["season"] == held].copy()
        p = fit_params(train)
        for mode in ("prior_only", "observed_only", "fitted_no_espn", "fitted"):
            test["pred"] = apply(test, p, mode)
            for pos, g in test.groupby("position"):
                rho = np.mean([spearman(gg["pred"], gg["future_points"]) for _, gg in g.groupby("week") if len(gg) > 4])
                out.append({"season": held, "mode": mode, "position": pos,
                            "mae": float(np.abs(g["pred"] - g["future_points"]).mean()),
                            "bias": float((g["pred"] - g["future_points"]).mean()), "rho": rho})
    return pd.DataFrame(out)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rows = player_rows()
    rows.to_csv(OUT / "player_rows.csv", index=False)
    rep = loso_forecast_report(rows)
    rep.to_csv(OUT / "forecast_loso.csv", index=False)
    pd.set_option("display.width", 200)
    print(rep.groupby(["mode", "position"])[["mae", "bias", "rho"]].mean().round(3).unstack("mode").to_string())
    full = fit_all(rows, SEASONS)
    (OUT / "forecast_params_all_seasons.json").write_text(full.to_json())
    print(full.to_json())
