"""Variance strategy: when you're the underdog, does a higher-variance lineup win more?  (2021-25, Desk's league)

Per player-week, a spread estimate from data before that week: sd of his actual/ESPN-projection ratio over his earlier
games this season and last season, shrunk toward his position's ratio sd with K_GAMES pseudo-games; his points sd =
proj x that. Lineup sd assumes independent players (as live.py does).
Per team-week: L0 = the lineup maximising ESPN projections (exact slot assignment, IR excluded). Alternatives are
single swaps of one L0 starter for an eligible bench player that cost <= COST projected points. "Up" picks the swap
that raises lineup variance most, "down" the one that lowers it most.
No head-to-head schedule is stored, so each team-week is scored against every other team's actual score that week
(11 pairs). Pregame p(win) for a pair is the live.py-style simulation of L0 vs the opponent's actually started
lineup (ESPN projections x position ratio quantiles, fit leave-one-season-out). Underdog pairs: p < 0.40 (and < 0.45);
favourite pairs: p > 0.60 (and > 0.55).
Reported: change in realised win rate (alt lineup vs L0, same opponent score) and the points cost per team-week, 95%
CIs bootstrapping season-weeks; placebo = a random swap with cost <= COST (500x), same pairs.
Also: is a player's weekly spread stable? split-half r of the ratio sd (odd vs even games, players with >= 10 games).

Result (2026-10-17): no. A player's weekly spread does not repeat (split-half r of ratio sd: WR +0.02, RB -0.08,
TE +0.05, QB +0.11, all +0.13), so "higher-variance" swaps are just lower-projected players. Underdogs (p<0.40) that
take the max-variance swap within 1 pt of projection lose 4.8 [1.7,7.7] pts of win rate (base 33%); within 2 pts,
5.5 [2.9,8.4]. That is worse than random same-cost swaps (placebo p 0.99-1.00). Favourites going low-variance:
-0.4 to -1.4 pts of win rate, ~random (p 0.46-0.89). Every variant also costs ~0.7-2.1 real points per team-week.

  python3 backtest/variance_strategy.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backtest"))
import scorecard as sc  # noqa: E402

SEAS = sc.SEAS
K_GAMES = 8.0
COSTS = (1.0, 2.0)
SIMS = 4000
RNG = np.random.default_rng(1)


def spreads(data: dict) -> pd.DataFrame:
    """Per (espn_id, season, week): shrunk ratio sd from earlier games (this season + last season)."""
    R = []
    for y, (_, ro, _) in data.items():
        r = ro[(ro["proj"] > 0)].drop_duplicates(["espn_id", "week"])[["espn_id", "week", "position", "pts", "proj"]].copy()
        r["season"] = y
        R.append(r)
    R = pd.concat(R, ignore_index=True)
    R["ratio"] = R["pts"] / R["proj"]
    R = R[R["proj"] >= 2]
    pos_var = R.groupby("position")["ratio"].var()
    out = []
    for e, g in R.sort_values(["season", "week"]).groupby("espn_id"):
        rs, ys, ws = g["ratio"].to_numpy(), g["season"].to_numpy(), g["week"].to_numpy()
        pv = pos_var.get(g["position"].iloc[0], pos_var.mean())
        for i in range(len(g)):
            m = ((ys == ys[i]) & (ws < ws[i])) | (ys == ys[i] - 1)
            x = rs[m]
            n = len(x)
            v = ((n - 1) * x.var(ddof=1) + K_GAMES * pv) / (n - 1 + K_GAMES) if n >= 2 else pv
            out.append((e, ys[i], ws[i], np.sqrt(v)))
    return pd.DataFrame(out, columns=["espn_id", "season", "week", "rsd"])


def stability(data: dict) -> None:
    rows = []
    for y, (_, ro, _) in data.items():
        r = ro[(ro["proj"] >= 5) & ~ro["lineup_slot"].isin(["BE", "IR"])].drop_duplicates(["espn_id", "week"]).sort_values("week")
        for (e, pos), g in r.groupby(["espn_id", "position"]):
            x = (g["pts"] / g["proj"]).to_numpy()
            if len(x) >= 10:
                rows.append((pos, x[0::2].std(ddof=1), x[1::2].std(ddof=1)))
    S = pd.DataFrame(rows, columns=["pos", "a", "b"])
    print("stability of a player's weekly spread (sd of actual/proj, odd vs even games, >= 10 starts): "
          + ", ".join(f"{p} r={np.corrcoef(g['a'], g['b'])[0, 1]:+.2f} (n={len(g)})" for p, g in S.groupby("pos"))
          + f"; all r={np.corrcoef(S['a'], S['b'])[0, 1]:+.2f}")


def lineups(r: pd.DataFrame, slots: list, cost: float):
    """L0 indices by slot, plus candidate swaps (bench idx, starter idx, cost, d_var)."""
    P = r.reset_index(drop=True)
    C = np.full((len(slots), len(P)), 1e6)
    elig = np.zeros((len(slots), len(P)), bool)
    for i, sl in enumerate(slots):
        ok = P["position"].isin(sc.ELIG[sl]).to_numpy()
        elig[i] = ok
        C[i, ok] = -P.loc[ok, "proj"].to_numpy()
    ri, ci = linear_sum_assignment(C)
    keep = C[ri, ci] < 1e5
    ri, ci = ri[keep], ci[keep]
    start = set(ci)
    var = (P["proj"] * P["rsd"]).to_numpy() ** 2
    sw = []
    for s_slot, s in zip(ri, ci):
        for b in range(len(P)):
            if b in start or not elig[s_slot, b]:
                continue
            c = P.at[s, "proj"] - P.at[b, "proj"]
            if c <= cost:
                sw.append((b, s, c, var[b] - var[s]))
    return P, list(ci), sw


def run() -> None:
    data = {y: sc.load(y) for y in SEAS}
    stability(data)
    S = spreads(data)
    rows = []
    for held in SEAS:
        rat = {}
        for y in SEAS:
            if y != held:
                _, ro, _ = data[y]
                st = ro[~ro["lineup_slot"].isin(["BE", "IR"]) & (ro["proj"] > 0) & ro["week"].between(3, 14)]
                for pos, g in st.groupby("position"):
                    rat.setdefault(pos, []).extend((g["pts"] / g["proj"]).tolist())
        Q = {p: np.quantile(v, np.linspace(0.01, 0.99, 50)) for p, v in rat.items()}
        s, ro, tw = data[held]
        slots = [k for k, n in s["lineup_slot_counts"].items() if k in sc.ELIG for _ in range(n)]
        ro = ro.merge(S[S["season"] == held].drop(columns="season"), on=["espn_id", "week"], how="left")
        ro["rsd"] = ro["rsd"].fillna(ro.groupby("position")["rsd"].transform("median")).fillna(1.0)
        act = tw.set_index(["week", "team_id"])["points"]

        def sim(projs, poss):
            tot = np.zeros(SIMS)
            for pj, pos in zip(projs, poss):
                if pj > 0:
                    q = Q.get(pos, Q["WR"])
                    u = RNG.random(SIMS) * (len(q) - 1)
                    lo = np.floor(u).astype(int)
                    tot += pj * (q[lo] + (q[np.minimum(lo + 1, len(q) - 1)] - q[lo]) * (u - lo))
            return tot
        for w, r in ro.groupby("week"):
            opp = {t: sim(g["proj"], g["position"]) for t, g in r[~r["lineup_slot"].isin(["BE", "IR"])].groupby("team_id")}
            for t, g in r[r["lineup_slot"] != "IR"].groupby("team_id"):
                for cost in COSTS:
                    P, L0, sw = lineups(g, slots, cost)
                    pts0 = float(P.loc[L0, "pts"].sum())
                    mine = sim(P.loc[L0, "proj"], P.loc[L0, "position"])
                    up = max(sw, key=lambda x: x[3]) if sw and max(x[3] for x in sw) > 0 else None
                    dn = min(sw, key=lambda x: x[3]) if sw and min(x[3] for x in sw) < 0 else None
                    alt = {k: (pts0 - P.at[v[1], "pts"] + P.at[v[0], "pts"], v[2]) if v else None for k, v in (("up", up), ("down", dn))}
                    rnd = [pts0 - P.at[s_, "pts"] + P.at[b_, "pts"] for b_, s_, _, _ in sw]
                    for o in opp:
                        if o == t or (w, o) not in act:
                            continue
                        p = float((mine > opp[o]).mean())
                        rows.append({"season": held, "week": w, "team": t, "opp_team": o, "cost": cost, "p": p, "opp": float(act[(w, o)]),
                                     "pts0": pts0, "up": alt["up"][0] if alt["up"] else np.nan, "up_cost": alt["up"][1] if alt["up"] else np.nan,
                                     "down": alt["down"][0] if alt["down"] else np.nan, "down_cost": alt["down"][1] if alt["down"] else np.nan,
                                     "rnd": rnd})
    X = pd.DataFrame(rows)
    report(X)


def wboot(d: pd.DataFrame, f, B: int = 1000):
    k = (d["season"].astype(str) + "_" + d["week"].astype(str)).to_numpy()
    uk = np.unique(k)
    ix = {u: np.flatnonzero(k == u) for u in uk}
    bs = [f(d.iloc[np.concatenate([ix[u] for u in RNG.choice(uk, len(uk))])]) for _ in range(B)]
    return f(d), *np.percentile(bs, [2.5, 97.5])


def report(X: pd.DataFrame) -> None:
    win = lambda a, o: (a > o).astype(float) + 0.5 * (a == o)
    print("\ncost  side       pairs   tw   base win  win change [95% CI]           placebo p   pts cost/team-week [95% CI]")
    for cost in COSTS:
        for side, col, sel in (("dog<.40", "up", lambda d: d["p"] < 0.40), ("dog<.45", "up", lambda d: d["p"] < 0.45),
                               ("fav>.60", "down", lambda d: d["p"] > 0.60), ("fav>.55", "down", lambda d: d["p"] > 0.55)):
            d = X[(X["cost"] == cost) & sel(X) & X[col].notna()].reset_index(drop=True)
            if len(d) < 50:
                continue
            ch = wboot(d, lambda z: (win(z[col], z["opp"]) - win(z["pts0"], z["opp"])).mean())
            tw = d.drop_duplicates(["season", "week", "team"])
            pc = wboot(tw, lambda z: (z[col] - z["pts0"]).mean())
            null = []
            for _ in range(500):
                ra = np.array([RNG.choice(v) if len(v) else p0 for v, p0 in zip(d["rnd"], d["pts0"])])
                null.append((win(ra, d["opp"]) - win(d["pts0"], d["opp"])).mean())
            null = np.array(null)
            pv = float((null >= ch[0]).mean())
            print(f"{cost:3.0f}   {side:9} {len(d):6} {len(tw):5}   {win(d['pts0'], d['opp']).mean():.3f}   {ch[0]:+.4f} [{ch[1]:+.4f},{ch[2]:+.4f}]   "
                  f"{pv:.3f} (null mean {null.mean():+.4f})   {pc[0]:+.2f} [{pc[1]:+.2f},{pc[2]:+.2f}]")


if __name__ == "__main__":
    run()
