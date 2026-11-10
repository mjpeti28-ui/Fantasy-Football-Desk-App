"""Which weeks 1-9 fantasy stats predict weeks 10-17 points, and where they beat ESPN.

Unit: player-season. Relevant players only: rostered non-IR in the league at the cut week, or in the
startable range by first-half PPG (QB15/RB36/WR42/TE15). Min 4 games before the cut, 3 after.
Target: league-scoring PPG after the cut, and the residual vs ESPN's weekly projections (actual - ESPN,
averaged over games played). History 2021-25; 2026 is test only (no fitting on it).

  python3 backtest/midseason_fantasy.py NFL_DIR [ACT_2026_DIR]
NFL_DIR holds nflverse stats_{y}.parquet, snaps_{y}.parquet and ffopportunity opp_{y}.parquet.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
NFL = Path(sys.argv[1])
ACT26 = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "history/actuals/2026"
HIST = [2021, 2022, 2023, 2024, 2025]
TOP = {"QB": 15, "RB": 36, "WR": 42, "TE": 15}
rng = np.random.default_rng(0)

ids = pd.read_csv(ROOT / "history/nflverse/ids.csv")
e2g = ids.dropna(subset=["espn_id"]).assign(espn_id=lambda d: d.espn_id.astype(int)).drop_duplicates("espn_id").set_index("espn_id")["gsis_id"]
p2g = ids.dropna(subset=["pfr_id"]).drop_duplicates("pfr_id").set_index("pfr_id")["gsis_id"]


def act_dir(y):
    return ACT26 if y == 2026 else ROOT / f"history/actuals/{y}"


def weekly(y):
    """One row per (gsis, week) played, with league points and raw counting stats."""
    st = pd.read_parquet(NFL / f"stats_{y}.parquet")
    st = st[(st.season_type == "REG") & st.position.isin(TOP)]
    tm = st.groupby(["team", "week"])[["carries", "attempts", "sacks_suffered"]].sum().add_suffix("_tm")
    st = st.join(tm, on=["team", "week"])
    op = pd.read_parquet(NFL / f"opp_{y}.parquet")[["player_id", "week", "total_fantasy_points_exp"]]
    op["week"] = op.week.astype(int)
    st = st.merge(op.groupby(["player_id", "week"], as_index=False).sum(), on=["player_id", "week"], how="left")
    sn = pd.read_parquet(NFL / f"snaps_{y}.parquet")
    sn = sn[sn.game_type == "REG"].assign(player_id=lambda d: d.pfr_player_id.map(p2g))
    st = st.merge(sn[["player_id", "week", "offense_pct"]].dropna(), on=["player_id", "week"], how="left")
    pw = pd.read_csv(act_dir(y) / "players_weekly.csv")
    pw = pw[(pw.kind == "actual") & pw.played].assign(player_id=lambda d: d.espn_id.map(e2g))
    st = st.merge(pw[["player_id", "espn_id", "week", "points"]].dropna(subset=["player_id"]), on=["player_id", "week"], how="inner")
    return st


def features(w):
    g = w.groupby("player_id")
    s = g.sum(numeric_only=True)
    f = pd.DataFrame({"games": g.size(), "position": g.position.first(), "name": g.player_display_name.first(),
                      "espn_id": g.espn_id.first()})
    f["ppg"] = s.points / f.games
    f["xfp_g"] = s.total_fantasy_points_exp / f.games
    f["fpoe_g"] = s.fantasy_points_ppr / f.games - f.xfp_g
    f["snap_pct"] = g.offense_pct.mean()
    f["tgt_share"] = g.target_share.mean()
    f["ay_share"] = g.air_yards_share.mean()
    f["carry_share"] = s.carries / s.carries_tm
    f["tgt_g"] = s.targets / f.games
    f["opp_g"] = (s.targets + s.carries) / f.games
    f["yds_tgt"] = s.receiving_yards / s.targets
    f["catch_rate"] = s.receptions / s.targets
    f["adot"] = s.receiving_air_yards / s.targets
    f["ypc"] = s.rushing_yards / s.carries
    f["td_rate"] = (s.receiving_tds + s.rushing_tds) / (s.targets + s.carries)
    f["epa_play"] = (s.receiving_epa.fillna(0) + s.rushing_epa.fillna(0)) / (s.targets + s.carries)
    db = s.attempts + s.sacks_suffered
    f["qb_epa_db"] = s.passing_epa / db
    f["qb_ypa"] = s.passing_yards / s.attempts
    f["qb_td_rate"] = s.passing_tds / s.attempts
    f["qb_rush_g"] = s.carries / f.games
    f["qb_att_g"] = s.attempts / f.games
    return f.replace([np.inf, -np.inf], np.nan)


def outcome(y, w, lo, hi):
    e = pd.read_csv(act_dir(y) / "espn_weekly_proj.csv")
    x = w[(w.week >= lo) & (w.week <= hi)].merge(e, on=["espn_id", "week"], how="left")
    g = x.groupby("player_id")
    return pd.DataFrame({"n_after": g.size(), "ros_ppg": g.points.mean(), "espn_ros": g.espn_proj.mean(),
                         "resid": (x.points - x.espn_proj).groupby(x.player_id).mean()})


def relevant(y, f, cut):
    r = pd.read_csv(act_dir(y) / "rosters_by_week.csv")
    r = r[(r.week == min(cut + 1, r.week.max())) & (r.lineup_slot != "IR")]
    keep = f.espn_id.isin(r.espn_id)
    rank = f.groupby("position").ppg.rank(ascending=False)
    return keep | (rank <= f.position.map(TOP))


def build(y, cut, last):
    w = weekly(y)
    f = features(w[w.week <= cut])
    f = f[f.games >= (4 if cut >= 8 else 3)]
    f = f.join(outcome(y, w, cut + 1, last), how="inner")
    f = f[(f.n_after >= 3) & relevant(y, f, cut)]
    f["season"] = y
    return f


STATS = {"QB": ["ppg", "xfp_g", "fpoe_g", "qb_att_g", "qb_rush_g", "qb_epa_db", "qb_ypa", "qb_td_rate"],
         "RB": ["ppg", "xfp_g", "fpoe_g", "snap_pct", "carry_share", "tgt_share", "opp_g", "ypc", "td_rate", "epa_play"],
         "WR": ["ppg", "xfp_g", "fpoe_g", "snap_pct", "tgt_share", "ay_share", "tgt_g", "adot", "yds_tgt", "catch_rate", "td_rate", "epa_play"],
         "TE": ["ppg", "xfp_g", "fpoe_g", "snap_pct", "tgt_share", "ay_share", "tgt_g", "adot", "yds_tgt", "catch_rate", "td_rate", "epa_play"]}


def spear(a, b):
    m = a.notna() & b.notna()
    return a[m].rank().corr(b[m].rank()) if m.sum() > 8 else np.nan


def boot_ci(d, fn, B=400):
    v = [fn(d.iloc[rng.integers(0, len(d), len(d))]) for _ in range(B)]
    return np.nanpercentile(v, [2.5, 97.5])


def ols(X, yv):
    A = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(A, yv, rcond=None)[0]


def pred(c, X):
    return np.column_stack([np.ones(len(X)), X]) @ c


def loso_gain(d, base, extra, target):
    """MAE(base) - MAE(base+extra) on held-out seasons; >0 = extra helps. Per-season list too."""
    gains = []
    for y in sorted(d.season.unique()):
        tr, te = d[d.season != y], d[d.season == y]
        cols = base + extra
        tr, te = tr.dropna(subset=cols + [target]), te.dropna(subset=cols + [target])
        if len(te) < 5:
            continue
        m0 = np.abs(te[target] - pred(ols(tr[base].values, tr[target].values), te[base].values)).mean() if base else np.abs(te[target] - tr[target].mean()).mean()
        m1 = np.abs(te[target] - pred(ols(tr[cols].values, tr[target].values), te[cols].values)).mean()
        gains.append(m0 - m1)
    return np.mean(gains), sum(g > 0 for g in gains), len(gains)


def report(cut, last, out):
    H = pd.concat([build(y, cut, last) for y in HIST])
    lines = [f"\n## Cut after week {cut}, outcome weeks {cut + 1}-{last}. History n={len(H)} player-seasons\n"]
    for pos, stats in STATS.items():
        d = H[H.position == pos]
        lines.append(f"\n### {pos} (n={len(d)})\n")
        lines.append("| stat | rho vs ROS PPG [95% CI] | adds to PPG (MAE gain, seasons +) | rho vs (actual - ESPN) [95% CI] | beats ESPN (MAE gain on resid, seasons +) |")
        lines.append("|---|---|---|---|---|")
        for s in stats:
            r = spear(d[s], d.ros_ppg)
            ci = boot_ci(d, lambda z: spear(z[s], z.ros_ppg))
            g, k, n = loso_gain(d, [] if s == "ppg" else ["ppg"], [s], "ros_ppg")
            dr = d.dropna(subset=["resid"])
            rr = spear(dr[s], dr.resid)
            cr = boot_ci(dr, lambda z: spear(z[s], z.resid))
            ge, ke, ne = loso_gain(dr, [], [s], "resid")
            lines.append(f"| {s} | {r:.2f} [{ci[0]:.2f}, {ci[1]:.2f}] | {g:+.3f} ({k}/{n}) | {rr:+.2f} [{cr[0]:+.2f}, {cr[1]:+.2f}] | {ge:+.3f} ({ke}/{ne}) |")
    out.write("\n".join(lines) + "\n")
    return H


if __name__ == "__main__":
    o = open(ROOT / "history/backtest/midseason_fantasy.md", "w")
    o.write("# Midseason fantasy stats: weeks 1-9 -> 10-17 (auto-generated tables)\n")
    H9 = report(9, 17, o)
    H9.to_csv(ROOT / "history/backtest/midseason_rows_hist.csv")
    # 2026 check: same question at the cut 5 -> weeks 6-9 (the only 2026 outcome window we have), vs history at the same cut
    H5 = pd.concat([build(y, 5, 9) for y in HIST])
    T5 = build(2026, 5, 9)
    o.write("\n## 2026 test: weeks 1-5 -> 6-9, Spearman rho (history 2021-25 same cut, 5-season range in brackets)\n\n")
    o.write("| pos | stat | hist rho | hist season range | 2026 rho | 2026 n | hist rho vs resid | 2026 rho vs resid |\n|---|---|---|---|---|---|---|---|\n")
    for pos, stats in STATS.items():
        for s in stats:
            h, t = H5[H5.position == pos], T5[T5.position == pos]
            per = [spear(h[h.season == y][s], h[h.season == y].ros_ppg) for y in HIST]
            o.write(f"| {pos} | {s} | {spear(h[s], h.ros_ppg):.2f} | {np.nanmin(per):.2f}..{np.nanmax(per):.2f} | {spear(t[s], t.ros_ppg):.2f} | {len(t)} | "
                    f"{spear(h[s], h.resid):+.2f} | {spear(t[s], t.resid):+.2f} |\n")
    H5.to_csv(ROOT / "history/backtest/midseason_rows_hist_cut5.csv")
    T5.to_csv(ROOT / "history/backtest/midseason_rows_2026_cut5.csv")
    # 2026 weeks 1-9 board (no outcome yet)
    w = weekly(2026)
    F = features(w[w.week <= 9])
    F = F[(F.games >= 4) & relevant(2026, F, 9)]
    e = pd.read_csv(ACT26 / "espn_weekly_proj.csv")
    F["espn_wk10"] = F.espn_id.map(e[e.week == 10].drop_duplicates("espn_id").set_index("espn_id").espn_proj)
    F.to_csv(ROOT / "history/backtest/midseason_2026_board.csv")
    o.close()
    print("done")
