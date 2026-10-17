"""Lead time vs ESPN: when Shifts flags a role change, how much of it has ESPN's weekly projection already priced in?

Events: the Shifts / roles.trends tags (last 2 games vs this season's earlier games, >= 4 games): snap_up / snap_down
(snap % +-15 pts), share_up (target share, or target+carry share for RB, +5 pts with snaps +10), share_spike (+5 pts
without), share_drop (-5 pts, snaps > -10). Only the first week of a run of the same tag counts (the "news" week W).
RB/WR/TE, 2021-25 regular season weeks 4-13; every player with an ESPN projection for his next game (league-wide,
because waiver targets are the point), split into free agents in Desk's league at W and rostered players.
For the next 4 games he plays (h = 1..4):
  real move   = league ppg over those games - his ppg over the earlier (pre-tag) games
  ESPN move h = ESPN's projection for game h - his mean ESPN projection over the earlier games
  priced in   = ESPN move at h=1 / real move (event-weighted ratio of means): 1.0 = ESPN already caught up
  catch-up    = ESPN move h=1 / ESPN move h=4: how much of ESPN's own eventual move is there in week 1
  edge        = mean(points - ESPN projection) over the 4 games, vs untagged player-weeks in the same season-weeks
95% CIs from a player-season bootstrap.

Result (2026-10-17, 3,538 tag events, 6,677 untagged player-weeks): there is no lead time to win. ESPN's very next
weekly projection already prices in the full real move for snap_up (0.95 [0.81,1.10] for free agents), share_up and
snap_down, and then backs off (its week-1 bump is 1.15-1.5x its week-4 level). Flagged free agents beat ESPN by only
+0.15 [-0.06,+0.35] pts/game (snap_up). Where ESPN errs, it over-reacts: after a target-share spike without more
snaps, players score 0.31 [0.07,0.53] pts/game below ESPN vs untagged (rostered: 0.43; share_up rostered: 0.82
[0.31,1.32]). Placebos (500x): vs random untagged weeks of the same player-season p<0.002 for all three; vs untagged
weeks with the same recent-points jump (is it just ESPN chasing hot games?) share_spike p=0.006 (rostered 0.006),
share_up rostered p<0.002 -- those jump-matched weeks sit at -0.01..-0.08, so the effect is specific to share bumps.
With 15 tag x pool cells, share_up rostered survives Bonferroni; share_spike is borderline (0.006 vs 0.0033). ESPN under-moves on share drops (prices 36%), and correctly so: those players also beat ESPN by +0.20.

  python3 backtest/lead_time.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MAIN = Path.home() / "Developer" / "Fantasy-Football-Desk-App"
SEAS = [2021, 2022, 2023, 2024, 2025]
SNAP_T, SHARE_T = 15.0, 5.0  # app/engine/roles.py
H = 4
BOOTS = 1000
RNG = np.random.default_rng(0)


def tag(pos: str, ds: float, dh: float) -> str | None:
    """Same branches as roles.trends (snap tags take precedence, as in shifts.zone)."""
    if ds >= SNAP_T:
        return "snap_up"
    if ds <= -SNAP_T:
        return "snap_down"
    if np.isnan(dh):
        return None
    if dh >= SHARE_T:
        return "share_up" if ds >= 10 else "share_spike"
    if dh <= -SHARE_T and ds > -10:
        return "share_drop"
    return None


def season(y: int) -> pd.DataFrame:
    rw = pd.read_csv(MAIN / "app" / "cache.nosync" / "roles" / str(y) / "roles_weekly.csv", low_memory=False)
    rw = rw[rw["pos"].isin(["RB", "WR", "TE"]) & rw["espn_id"].notna() & (rw["week"] <= 17)].copy()
    rw["espn_id"] = rw["espn_id"].astype(int)
    rw = rw[rw["snap"] > 0].sort_values(["espn_id", "week"]).drop_duplicates(["espn_id", "week"])
    base = ROOT / "history" / "actuals" / str(y)
    pr = pd.read_csv(base / "espn_weekly_proj.csv").groupby(["espn_id", "week"])["espn_proj"].first()
    pw = pd.read_csv(base / "players_weekly.csv")
    pts = pw[pw["kind"] == "actual"].groupby(["espn_id", "week"])["points"].sum()
    ro = pd.read_csv(base / "rosters_by_week.csv")
    rost = set(zip(ro["espn_id"], ro["week"]))
    out = []
    for (e, pos), d in rw.groupby(["espn_id", "pos"]):
        wk = d["week"].to_numpy()
        sn = d["snap"].to_numpy(float)
        num = (d["tgt"] + d["car"]).to_numpy(float) if pos == "RB" else d["tgt"].to_numpy(float)
        den = (d["t_tgt"] + d["t_car"]).to_numpy(float) if pos == "RB" else d["t_tgt"].to_numpy(float)
        sh = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan) * 100
        prev = None
        for i in range(3, len(wk)):
            ds = sn[i - 1:i + 1].mean() - sn[:i - 1].mean()
            dh = np.nanmean(sh[i - 1:i + 1]) - np.nanmean(sh[:i - 1])
            t = tag(pos, ds, dh) or "none"
            W = int(wk[i])
            new = t != prev
            prev = t
            if not (4 <= W <= 13) or (t != "none" and not new):
                continue
            fut = wk[i + 1:i + 1 + H]
            if len(fut) < 2 or pr.get((e, int(fut[0])), 0) <= 0:
                continue
            bw = wk[:i - 1]
            bp = [pr.get((e, int(w))) for w in bw]
            bp = [v for v in bp if v is not None and v > 0]
            if not bp:
                continue
            fp = np.array([pr.get((e, int(w)), np.nan) for w in fut], float)
            fa = np.array([pts.get((e, int(w)), 0.0) for w in fut], float)
            out.append({"season": y, "week": W, "espn_id": e, "pos": pos, "tag": t, "fa": (e, W) not in rost,
                        "base_ppg": float(np.mean([pts.get((e, int(w)), 0.0) for w in bw])),
                        "rec_ppg": float(np.mean([pts.get((e, int(w)), 0.0) for w in wk[i - 1:i + 1]])), "base_proj": float(np.mean(bp)),
                        "fut_ppg": float(fa.mean()), "p1": fp[0], "p4": fp[min(3, len(fp) - 1)], "resid": float(np.nanmean(fa - fp)),
                        "n_fut": len(fut)})
    return pd.DataFrame(out)


def boot(g: pd.DataFrame, f) -> tuple:
    k = (g["espn_id"].astype(str) + "_" + g["season"].astype(str)).to_numpy()
    uk = np.unique(k)
    ix = {u: np.flatnonzero(k == u) for u in uk}
    v = f(g)
    bs = [f(g.iloc[np.concatenate([ix[u] for u in RNG.choice(uk, len(uk))])]) for _ in range(BOOTS)]
    return v, *np.nanpercentile(bs, [2.5, 97.5])


def main() -> None:
    E = pd.concat([season(y) for y in SEAS], ignore_index=True)
    E["real"] = E["fut_ppg"] - E["base_ppg"]
    E["m1"] = E["p1"] - E["base_proj"]
    E["m4"] = E["p4"] - E["base_proj"]
    ctrl = E[E["tag"] == "none"].groupby(["season", "week"])["resid"].mean()
    E["edge"] = E["resid"] - E.set_index(["season", "week"]).index.map(ctrl).to_numpy()
    print("events", E[E["tag"] != "none"].groupby("tag").size().to_dict(), "untagged player-weeks", int((E["tag"] == "none").sum()))
    print("tag          pool      n   real move  ESPN wk1  priced in [95% CI]     ESPN wk1/wk4   edge vs untagged, pts/game [95% CI]")
    for t in ["snap_up", "share_up", "share_spike", "snap_down", "share_drop"]:
        for pool, m in (("all", slice(None)), ("free agent", E["fa"]), ("rostered", ~E["fa"])):
            g = E[(E["tag"] == t) & m] if pool != "all" else E[E["tag"] == t]
            if len(g) < 25:
                continue
            pi = boot(g, lambda d: d["m1"].mean() / d["real"].mean())
            cu = boot(g, lambda d: d["m1"].mean() / d["m4"].mean())
            ed = boot(g, lambda d: d["edge"].mean())
            print(f"{t:12} {pool:10} {len(g):5}  {g['real'].mean():+6.2f}    {g['m1'].mean():+6.2f}   {pi[0]:5.2f} [{pi[1]:5.2f},{pi[2]:5.2f}]   "
                  f"{cu[0]:5.2f} [{cu[1]:4.2f},{cu[2]:4.2f}]   {ed[0]:+5.2f} [{ed[1]:+5.2f},{ed[2]:+5.2f}]")
    print()
    allp = pd.Series(True, index=E.index)
    placebo(E, "share_spike", allp)
    placebo(E, "share_spike", ~E["fa"])
    placebo(E, "share_up", ~E["fa"])


def placebo(E: pd.DataFrame, t: str, pool, n: int = 500) -> None:
    """(a) each event swapped for a random untagged week of the same player-season; (b) each event swapped for a
    random untagged week (any player, same position) with a similar recent-points jump (same jump decile)."""
    g = E[(E["tag"] == t) & pool].copy()
    U = E[E["tag"] == "none"].copy()
    U["ps"] = U["espn_id"].astype(str) + "_" + U["season"].astype(str)
    g["ps"] = g["espn_id"].astype(str) + "_" + g["season"].astype(str)
    obs = g["edge"].mean()
    by_ps = U.groupby("ps")["edge"].apply(np.array)
    ga = g[g["ps"].isin(by_ps.index)]
    pools = [by_ps[k] for k in ga["ps"]]
    a = np.array([np.mean([RNG.choice(x) for x in pools]) for _ in range(n)])
    E2 = pd.concat([g, U])
    E2["jump"] = E2["rec_ppg"] - E2["base_ppg"]
    E2["dec"] = pd.qcut(E2["jump"], 10, labels=False, duplicates="drop")
    gd, Ud = E2.iloc[:len(g)], E2.iloc[len(g):]
    cells = Ud.groupby(["pos", "dec"])["edge"].apply(np.array)
    keys = [k for k in zip(gd["pos"], gd["dec"]) if k in cells.index]
    b = np.array([np.mean([RNG.choice(cells[k]) for k in keys]) for _ in range(n)])
    pa = float((a <= ga["edge"].mean()).mean()) if obs < 0 else float((a >= ga["edge"].mean()).mean())
    pb = float((b <= obs).mean()) if obs < 0 else float((b >= obs).mean())
    print(f"placebo {t:11} n={len(g)}: observed {obs:+.2f} | (a) same player-season untagged weeks: mean {a.mean():+.2f}, "
          f"one-sided p={pa:.3f} (n={len(ga)}) | (b) untagged weeks with the same recent-points jump: mean {b.mean():+.2f}, one-sided p={pb:.3f}")


if __name__ == "__main__":
    main()
