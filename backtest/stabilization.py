"""Stabilization clock: after how many games is each usage/efficiency stat mostly signal?  (2021-25, regular season)

Per player-season, one row per game played (offense snap share > 0). Stats are ratio-of-sums over a set of games.
  split-half reliability at N games: players with >= N games; take their first N games, split odd/even, compute the
    stat on each half, Pearson r across players, Spearman-Brown corrected to N games: rel = 2r / (1 + r).
  N-weeks-in: stat over weeks 1..N vs the same stat over weeks N+1..17 (and vs fantasy ppg over N+1..17), N = 2..8.
  95% CIs: bootstrap whole player-seasons (1000x).
Relevant players: on a roster in Desk's league for >= 3 weeks that season (history/actuals/*/rosters_by_week.csv).
Free data: nflverse weekly player stats (nflreadpy, cached in the main checkout's .cache/nflverse_full), nflverse snaps, the
league's own weekly points. PFF (cache only, no API calls): route rate, slot rate, YPRR from differencing the cached
cumulative receiving/summary windows w01-02..w01-12 (weeks 3-12), ROE from the v2 receiving windows (weeks 4-12).
Hot start: at week N=4/6, a player's ppg over/under his preseason-ish baseline (prior-season ppg) is split into a usage
part (expected points from nflverse opportunity, xfp) and an efficiency part (ppg - xfp); each part's carry-over into
weeks N+1..17 ppg is the regression slope.

Result (2026-10-17, 1,023 rostered player-seasons): games to reliability 0.5 / 0.7
  stable within 2-5 games: snap % (all), carry share (RB), target share (RB 2/4, WR 2/4, TE 2/5), air-yards share
    (WR 3/5, TE 4/8), slot rate (PFF), QB rush share (2/3), RB ppg (2/5)
  mid (5-12): WR aDOT (4/8), WR YPRR (6/9, PFF), WR ppg (4/9), QB ppg (6/12), TE ppg (4/11), QB YPA (9/15)
  never within a season (>16): yards/target, catch rate, TD rate (all positions), RB YPC (13/>16), QB INT rate, ROE
  N weeks in (r of wk 1..N with wk N+1..17 at N=4): target share .66-.69, carry share .68, snap % .61-.66,
    ppg .32-.58; efficiency .02-.28. Hot start at wk 6: ~60-70% of a usage-driven gap vs last season's ppg carries
    over, ~45% of an efficiency-driven gap for WR/RB, ~0-20% for QB/TE.
  Caveat: PFF route_rate is routes / the player's own pass snaps, ~1 for WRs, so its low reliability is range
  restriction, not noise.

  python3 backtest/stabilization.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MAIN = Path.home() / "Developer" / "Fantasy-Football-Desk-App"
sys.path.insert(0, str(ROOT / "backtest"))
import pff_api as api  # noqa: E402
from pff_probe import rows_of  # noqa: E402

SEAS = [2021, 2022, 2023, 2024, 2025]
FULL = MAIN / ".cache" / "nflverse_full" / "player_stats_2021_2025.csv"
PFF = MAIN / ".cache" / "pff"
NS = list(range(2, 17, 2))
BOOTS = 1000
RNG = np.random.default_rng(0)

# stat -> (numerator, denominator); per-game columns built below
STATS = {
    "WR": ["snap_pct", "tgt_share", "ay_share", "adot", "yds_per_tgt", "catch_rate", "td_per_tgt", "ppg", "route_rate", "slot_rate", "yprr", "roe"],
    "TE": ["snap_pct", "tgt_share", "ay_share", "adot", "yds_per_tgt", "catch_rate", "td_per_tgt", "ppg", "route_rate", "slot_rate", "yprr", "roe"],
    "RB": ["snap_pct", "carry_share", "tgt_share", "ypc", "yds_per_tgt", "td_per_opp", "ppg"],
    "QB": ["ypa", "td_per_att", "int_per_att", "rush_share", "ppg"],
}
DEF = {"snap_pct": ("snaps", "team_snaps"), "tgt_share": ("tgt", "team_tgt"), "ay_share": ("ay", "team_ay"),
       "adot": ("ay", "tgt"), "yds_per_tgt": ("rec_yds", "tgt"), "catch_rate": ("rec", "tgt"), "td_per_tgt": ("rec_td", "tgt"),
       "carry_share": ("car", "team_car"), "ypc": ("rush_yds", "car"), "td_per_opp": ("td", "opp"), "ppg": ("pts", "one"),
       "ypa": ("pass_yds", "att"), "td_per_att": ("pass_td", "att"), "int_per_att": ("ints", "att"), "rush_share": ("car", "team_car"),
       "route_rate": ("routes", "pass_plays"), "slot_rate": ("slot", "pass_plays"), "yprr": ("pff_yds", "routes"), "roe": ("roe_tot", "pff_tgt")}


def ids() -> pd.DataFrame:
    return pd.read_csv(ROOT / "history" / "nflverse" / "ids.csv", usecols=["gsis_id", "pfr_id", "espn_id"])


def pbp_games(y: int) -> pd.DataFrame:
    """Per player-game box counts + team totals from nflverse weekly player stats (cached in the main checkout)."""
    if not FULL.exists():
        import nflreadpy as nfl
        FULL.parent.mkdir(parents=True, exist_ok=True)
        nfl.load_player_stats(SEAS).write_csv(str(FULL))
    d = pd.read_csv(FULL, low_memory=False)
    d = d[(d["season"] == y) & (d["season_type"] == "REG")]
    m = {"targets": "tgt", "receptions": "rec", "receiving_yards": "rec_yds", "receiving_air_yards": "ay", "receiving_tds": "rec_td",
         "carries": "car", "rushing_yards": "rush_yds", "rushing_tds": "rush_td", "attempts": "att", "passing_yards": "pass_yds",
         "passing_tds": "pass_td", "passing_interceptions": "ints"}
    g = d[["player_id", "week", "team"] + list(m)].rename(columns={"player_id": "gsis_id", **m}).fillna(0)
    t = g.groupby(["team", "week"])[["tgt", "ay", "car"]].sum().rename(columns=lambda c: "team_" + c).reset_index()
    return g.merge(t, on=["team", "week"], how="left")


def pff_weekly(y: int, pe: dict) -> pd.DataFrame:
    """Per-week PFF receiving counts from differencing cumulative windows w01-(k-1) -> w01-k."""
    cum = {}
    for k in range(2, 13):
        p = PFF / "receiving_summary" / str(y) / f"w01-{k:02d}.json"
        if p.exists():
            d = pd.DataFrame(json.loads(p.read_text())[next(iter(json.loads(p.read_text())))])
            cum[k] = d.set_index("player_id")[["routes", "pass_plays", "slot_snaps", "yards"]].apply(pd.to_numeric, errors="coerce").fillna(0)
    roe = {}
    for k in range(3, 13):
        b = api.cache_file("/v2/nfl/positions/reports/receiving", {"season": y, "weekGroup": "REG", "week": 1, "weekTo": k})
        if b.exists():
            d = pd.DataFrame(rows_of(json.loads(b.read_text())["body"]))
            roe[k] = d.set_index("playerId")[["receptionsOeTotal", "targets"]].apply(pd.to_numeric, errors="coerce").fillna(0)
    out = []
    for k in range(3, 13):
        if k in cum and k - 1 in cum:
            a, b = cum[k], cum[k - 1].reindex(cum[k].index).fillna(0)
            w = (a - b).rename(columns={"slot_snaps": "slot", "yards": "pff_yds"})
            if k in roe and k - 1 in roe:
                ra, rb = roe[k], roe[k - 1].reindex(roe[k].index).fillna(0)
                w = w.join((ra - rb).rename(columns={"receptionsOeTotal": "roe_tot", "targets": "pff_tgt"}))
            out.append(w[w["pass_plays"] > 0].assign(week=k).reset_index().rename(columns={"index": "player_id"}))
    if not out:
        return pd.DataFrame()
    W = pd.concat(out, ignore_index=True)
    W["espn_id"] = W["player_id"].map(pe)
    return W.dropna(subset=["espn_id"]).drop(columns="player_id")


def season_games(y: int, I: pd.DataFrame, pe: dict) -> pd.DataFrame:
    base = ROOT / "history" / "actuals" / str(y)
    ro = pd.read_csv(base / "rosters_by_week.csv")
    nwk = ro.groupby("espn_id")["week"].nunique()
    pos = ro.drop_duplicates("espn_id").set_index("espn_id")["position"]
    rel = set(nwk[nwk >= 3].index)
    sn = pd.read_csv(ROOT / "history" / "nflverse" / str(y) / "snaps.csv")
    sn = sn.merge(I.dropna(subset=["pfr_id"]).drop_duplicates("pfr_id")[["pfr_id", "espn_id"]], left_on="pfr_player_id", right_on="pfr_id")
    ts = sn.groupby(["team", "week"])["offense_snaps"].max().rename("team_snaps")  # a full-game player approximates team snaps
    sn = sn.join(ts, on=["team", "week"]).rename(columns={"offense_snaps": "snaps"})
    sn = sn[(sn["snaps"] > 0) & sn["espn_id"].isin(rel)].groupby(["espn_id", "week"], as_index=False).agg(snaps=("snaps", "max"), team_snaps=("team_snaps", "max"), team=("team", "first"))
    g = pbp_games(y).merge(I.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id")[["gsis_id", "espn_id"]], on="gsis_id").drop(columns=["gsis_id", "team"])
    G = sn.merge(g, on=["espn_id", "week"], how="left")
    pw = pd.read_csv(base / "players_weekly.csv")
    pw = pw[pw["kind"] == "actual"].groupby(["espn_id", "week"])["points"].sum().rename("pts")
    G = G.join(pw, on=["espn_id", "week"])
    pf = pff_weekly(y, pe)
    if len(pf):
        G = G.merge(pf, on=["espn_id", "week"], how="left")
    G = G.fillna({c: 0 for c in G.columns if c not in ("team",)})
    G["one"] = 1.0
    G["td"] = G["rec_td"] + G["rush_td"]
    G["opp"] = G["car"] + G["tgt"]
    G["position"] = G["espn_id"].map(pos).replace({"D/ST": "DST"})
    G = G[G["week"] <= 17]
    return G.assign(season=y, key=G["espn_id"].astype(int).astype(str) + f"_{y}")


def _val(g: pd.DataFrame, s: str) -> pd.Series:
    n, d = DEF[s]
    a = g.groupby("key")[[n, d]].sum()
    return (a[n] / a[d].where(a[d] > 0)).rename(s)


def reliability(G: pd.DataFrame, s: str) -> dict:
    """rel(N) for each N, with player-season bootstrap CI. PFF stats only use games that have PFF rows."""
    n, d = DEF[s]
    g = G[G[d] > 0] if s in ("route_rate", "slot_rate", "yprr", "roe") else G
    g = g.sort_values(["key", "week"]).copy()
    g["i"] = g.groupby("key").cumcount()
    out = {}
    for N in NS:
        cnt = g.groupby("key")["i"].max() + 1
        h = g[g["key"].isin(cnt[cnt >= N].index) & (g["i"] < N)]
        if h["key"].nunique() < 30:
            break
        A = h.assign(half=h["i"] % 2).groupby(["key", "half"])[[n, d]].sum()
        v = (A[n] / A[d].where(A[d] > 0)).unstack().dropna()
        if len(v) < 30:
            break
        x, y = v[0].to_numpy(), v[1].to_numpy()

        def sb(ix):
            r = np.corrcoef(x[ix], y[ix])[0, 1]
            return 2 * r / (1 + r)
        bs = [sb(RNG.integers(0, len(x), len(x))) for _ in range(BOOTS)]
        out[N] = (sb(np.arange(len(x))), *np.nanpercentile(bs, [2.5, 97.5]), len(x))
    return out


def crossing(curve: dict, t: float, idx: int = 0):
    """First N where rel >= t, linearly interpolated between grid points."""
    Ns = sorted(curve)
    for a, b in zip(Ns, Ns[1:]):
        ra, rb = curve[a][idx], curve[b][idx]
        if ra >= t:
            return float(a)
        if rb >= t:
            return a + (t - ra) / (rb - ra) * (b - a)
    return float(Ns[0]) if Ns and curve[Ns[0]][idx] >= t else None


def weeks_in(G: pd.DataFrame, s: str) -> dict:
    out = {}
    for N in range(2, 9):
        early, late = G[G["week"] <= N], G[G["week"] > N]
        ke = early.groupby("key").size()
        kl = late.groupby("key").size()
        keep = ke[ke >= max(2, N - 1)].index.intersection(kl[kl >= 4].index)
        a = _val(early[early["key"].isin(keep)], s)
        b = _val(late[late["key"].isin(keep)], s)
        p = _val(late[late["key"].isin(keep)], "ppg")
        j = pd.concat([a, b.rename("b"), p.rename("p")], axis=1).dropna()
        if len(j) < 30:
            continue
        x, y, z = j.iloc[:, 0].to_numpy(), j["b"].to_numpy(), j["p"].to_numpy()
        bs = []
        for _ in range(300):
            ix = RNG.integers(0, len(x), len(x))
            bs.append(np.corrcoef(x[ix], y[ix])[0, 1])
        out[N] = (np.corrcoef(x, y)[0, 1], *np.percentile(bs, [2.5, 97.5]), np.corrcoef(x, z)[0, 1], len(j))
    return out


def hot_start(G: pd.DataFrame, X: pd.DataFrame) -> None:
    """ppg gap vs prior-season ppg at week N, split into usage (xfp) and efficiency (ppg - xfp) parts."""
    for N in (4, 6):
        rows = []
        for (k, pos), g in G.groupby(["key", "position"]):
            if pos not in STATS:
                continue
            e, l = g[g["week"] <= N], g[g["week"] > N]
            if len(e) < N - 1 or len(l) < 4:
                continue
            rows.append((k, pos, e["pts"].mean(), e["xfp"].mean(), l["pts"].mean()))
        R = pd.DataFrame(rows, columns=["key", "pos", "ppg", "xfp", "ros"])
        R["espn"], R["season"] = R["key"].str.split("_").str[0].astype(int), R["key"].str.split("_").str[1].astype(int)
        R = R.merge(X, on=["espn", "season"], how="inner")
        for pos, g in R.groupby("pos"):
            gap_u, gap_e, y = (g["xfp"] - g["prior"]).to_numpy(), (g["ppg"] - g["xfp"]).to_numpy(), (g["ros"] - g["prior"]).to_numpy()
            M = np.column_stack([np.ones(len(g)), gap_u, gap_e])

            def fit(ix):
                return np.linalg.lstsq(M[ix], y[ix], rcond=None)[0][1:]
            b = fit(np.arange(len(g)))
            bs = np.array([fit(RNG.integers(0, len(g), len(g))) for _ in range(BOOTS)])
            lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
            print(f"hot start wk{N} {pos}: n={len(g):4}  usage part carries {b[0]:.2f} [{lo[0]:.2f},{hi[0]:.2f}]  "
                  f"efficiency part carries {b[1]:.2f} [{lo[1]:.2f},{hi[1]:.2f}]  (1.0 = fully real, 0 = pure noise)")


def main() -> None:
    I = ids()
    pe = {}
    p = PFF / "db_playerids.csv"
    if p.exists():
        di = pd.read_csv(p, usecols=["pff_id", "espn_id"]).dropna()
        pe = {int(a): int(b) for a, b in zip(di["pff_id"], di["espn_id"])}
    G = pd.concat([season_games(y, I, pe) for y in SEAS], ignore_index=True)
    # nflverse expected fantasy points per game for the hot-start split (league ppg vs PPR xfp differ in level; slopes still read as carry-over)
    xs = []
    for y in SEAS:
        op = pd.read_csv(ROOT / "history" / "nflverse" / str(y) / "opportunity.csv").merge(I.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id")[["gsis_id", "espn_id"]], left_on="player_id", right_on="gsis_id")
        xs.append(op.groupby(["espn_id", "week"], as_index=False)["total_fantasy_points_exp"].sum().assign(season=y))
    xf = pd.concat(xs).rename(columns={"total_fantasy_points_exp": "xfp"})
    G = G.merge(xf, on=["espn_id", "week", "season"], how="left")
    G["xfp"] = G["xfp"].fillna(0)
    print("player-games", len(G), "player-seasons", G["key"].nunique(), G.drop_duplicates("key")["position"].value_counts().to_dict())
    print("\nSTABILIZATION (split-half, Spearman-Brown): N games where reliability >= 0.5 / 0.7 [95% CI from the curve's bootstrap bounds]; rel at N=4/8/12")
    for pos, stats in STATS.items():
        g = G[G["position"] == pos]
        for s in stats:
            c = reliability(g, s)
            if not c:
                continue
            f = lambda t: crossing(c, t)
            lo = lambda t: crossing(c, t, 2)  # upper CI bound crosses first -> earliest plausible N
            hi = lambda t: crossing(c, t, 1)
            fmt = lambda v: (" <=2" if v <= 2 else f"{v:4.1f}") if v is not None else f"  >{max(c)}"
            at = " ".join(f"N{N}:{c[N][0]:+.2f}[{c[N][1]:+.2f},{c[N][2]:+.2f}]" for N in (4, 8, 12) if N in c)
            print(f"{pos} {s:12} 0.5 at {fmt(f(0.5))} [{fmt(lo(0.5))},{fmt(hi(0.5))}]  0.7 at {fmt(f(0.7))} [{fmt(lo(0.7))},{fmt(hi(0.7))}]  {at}  (n@4={c[min(c)][3]})")
    print("\nN WEEKS IN: r(stat wk1..N, same stat wk N+1..17) [95% CI] | r with ROS ppg")
    for pos, stats in STATS.items():
        g = G[G["position"] == pos]
        for s in stats:
            w = weeks_in(g, s)
            if w:
                print(f"{pos} {s:12} " + "  ".join(f"N{N}:{v[0]:+.2f}[{v[1]:+.2f},{v[2]:+.2f}]|{v[3]:+.2f}" for N, v in w.items() if N in (2, 4, 6, 8)) + f"  (n@4={w.get(4, (0,) * 5)[4]})")
    # prior-season league ppg as the hot-start baseline
    pr = G.groupby(["espn_id", "season"])["pts"].mean().reset_index()
    pr["season"] += 1
    X = pr.rename(columns={"espn_id": "espn", "pts": "prior"})
    X["espn"] = X["espn"].astype(int)
    print()
    hot_start(G, X)


if __name__ == "__main__":
    main()
