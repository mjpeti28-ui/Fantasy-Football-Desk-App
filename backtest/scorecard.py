"""Decision scorecard: lineup regret and probability calibration, 2021-25 (Desk's league, regular season).

A. Lineup regret, per team-week: hindsight-optimal lineup points (exact slot assignment over the active roster, IR
   excluded) minus (1) the lineup the manager actually started and (2) the lineup that maximises ESPN's projections
   (what "start ESPN's numbers" would have scored). The league seeds by total points scored, so regret is standings
   currency. Mean per team-week, per manager, and how often the manager beat ESPN's lineup; 95% CIs bootstrap weeks.
B. Calibration (Brier score, Brier skill vs the base rate, reliability bins, logistic calibration slope):
   B1 hit_prob P(start)/P(boom)/P(bust): its own leave-one-season-out predictions (backtest/hit_prob.py).
   B2 pregame matchup win probability as app/engine/live.py computes it: each starter's ESPN projection x an
      independent draw from that position's actual/projection quantiles (fit leave-one-season-out on rostered
      starters), A beats B by simulation. No head-to-head schedule is stored, so every pair of teams in a week is
      scored (calibration does not need the real pairings). CIs bootstrap season-weeks.

Result (2026-10-17; 840 team-weeks, started-lineup points match ESPN team scores in 100%):
  A. managers leave 15.8 [15.0,16.6] pts/week on the bench vs hindsight; ESPN's-projection lineup leaves 14.6
     [13.7,15.7]; managers trail ESPN's lineup by 1.16 [0.43,1.81] pts/week (every manager and season negative;
     team 3: -1.18/wk, -83 pts over 70 weeks), and beat it in only 33% of weeks (27% identical).
  B1. hit_prob (rostered, LOSO): P(start) Brier skill +0.26 [0.21,0.31], ECE 0.021, slope 0.89; P(boom) +0.21, ECE
     0.018 (top bin overconfident); P(bust) +0.01 [-0.00,0.03]: calibrated but no skill over the base rate.
  B2. pregame win prob (live.py method, 4,618 team pairs): Brier 0.227, skill +0.09 [0.06,0.12] vs base rate,
     slope 0.92, spread ratio 1.04 -- well calibrated; bins within ~0.04 except the n=24 lowest bin.

  python3 backtest/scorecard.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backtest"))
import hit_prob as hp  # noqa: E402

SEAS = [2021, 2022, 2023, 2024, 2025]
ELIG = {"QB": {"QB"}, "RB": {"RB"}, "WR": {"WR"}, "TE": {"TE"}, "DST": {"DST", "D/ST"}, "K": {"K"}, "RB/WR": {"RB", "WR"},
        "WR/TE": {"WR", "TE"}, "FLEX": {"RB", "WR", "TE"}}
SIMS = 4000
RNG = np.random.default_rng(0)
MY_TEAM = 3


def load(y: int):
    b = ROOT / "history" / "actuals" / str(y)
    s = json.loads((b / "settings.json").read_text())
    ro = pd.read_csv(b / "rosters_by_week.csv")
    pw = pd.read_csv(b / "players_weekly.csv")
    pts = pw[pw["kind"] == "actual"].groupby(["espn_id", "week"])["points"].sum()
    pr = pd.read_csv(b / "espn_weekly_proj.csv").groupby(["espn_id", "week"])["espn_proj"].first()
    ro = ro[ro["week"] <= s["regular_season_weeks"]].copy()
    ro["pts"] = [pts.get((e, w), 0.0) for e, w in zip(ro["espn_id"], ro["week"])]
    ro["proj"] = [pr.get((e, w), 0.0) for e, w in zip(ro["espn_id"], ro["week"])]
    tw = pd.read_csv(b / "team_weekly.csv")
    return s, ro, tw


def best(r: pd.DataFrame, slots: list, col: str) -> float:
    """Exact best lineup by `col`; returns its realised points."""
    P = r.reset_index(drop=True)
    if not len(P):
        return 0.0
    C = np.full((len(slots), len(P)), 1e6)
    for i, sl in enumerate(slots):
        ok = P["position"].isin(ELIG[sl]).to_numpy()
        C[i, ok] = -P.loc[ok, col].to_numpy()
    ri, ci = linear_sum_assignment(C)
    keep = C[ri, ci] < 1e5
    return float(P.loc[ci[keep], "pts"].sum())


def regret() -> pd.DataFrame:
    rows = []
    for y in SEAS:
        s, ro, tw = load(y)
        slots = [k for k, n in s["lineup_slot_counts"].items() if k in ELIG for _ in range(n)]
        for (w, t), r in ro.groupby(["week", "team_id"]):
            act = r[~r["lineup_slot"].isin(["BE", "IR"])]["pts"].sum()
            pool = r[r["lineup_slot"] != "IR"]
            rows.append({"season": y, "week": w, "team": t, "actual": act, "opt": best(pool, slots, "pts"), "espn": best(pool, slots, "proj")})
    R = pd.DataFrame(rows)
    R["reg_actual"] = R["opt"] - R["actual"]
    R["reg_espn"] = R["opt"] - R["espn"]
    R["vs_espn"] = R["actual"] - R["espn"]
    return R


def wboot(R: pd.DataFrame, f, B: int = 1000):
    k = (R["season"].astype(str) + "_" + R["week"].astype(str)).to_numpy()
    uk = np.unique(k)
    ix = {u: np.flatnonzero(k == u) for u in uk}
    bs = [f(R.iloc[np.concatenate([ix[u] for u in RNG.choice(uk, len(uk))])]) for _ in range(B)]
    return f(R), *np.percentile(bs, [2.5, 97.5])


def brier(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    b = np.mean((p - y) ** 2)
    return b, 1 - b / np.mean((y.mean() - y) ** 2)


def cal_slope(y, p):
    from sklearn.linear_model import LogisticRegression
    p = np.asarray(p, float)
    z = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4))).reshape(-1, 1)
    return float(LogisticRegression(C=1e6).fit(z, y).coef_[0, 0])


def hit_calibration() -> None:
    d = hp.load()
    for task in hp.TASKS:
        d["p_" + task], d["y_" + task] = np.nan, np.nan
    for held in sorted(d["season"].unique()):
        tr, te = d["season"] != held, d["season"] == held
        ln = hp.lines(d[tr])
        c = hp.fit(d[tr], ln)
        for task in hp.TASKS:
            d.loc[te, "p_" + task] = hp.predict(c[task], d.loc[te, "position"], d.loc[te, "est_ppg"])
            d.loc[te, "y_" + task] = hp.target(d[te], task, ln)
    d = d[d["rostered"].astype(bool)] if "rostered" in d else d
    for task in hp.TASKS:
        g = d.dropna(subset=["p_" + task, "y_" + task])
        y, p = g["y_" + task].to_numpy(), g["p_" + task].to_numpy()
        b, bss = brier(y, p)
        ece, worst, _ = hp.reliability(g["y_" + task], g["p_" + task])
        k = (g["espn_id"].astype(str) + "_" + g["season"].astype(str)).to_numpy()
        uk = np.unique(k)
        ix = {u: np.flatnonzero(k == u) for u in uk}
        bs = []
        for _ in range(500):
            ii = np.concatenate([ix[u] for u in RNG.choice(uk, len(uk))])
            bs.append(brier(y[ii], p[ii])[1])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"  P({task:5}) rostered n={len(g):5} base={y.mean():.3f} Brier={b:.3f} skill vs base rate={bss:+.3f} [{lo:+.3f},{hi:+.3f}] "
              f"ECE={ece:.3f} worst bin off {worst:.3f} slope={cal_slope(y, p):.2f}")


def winprob() -> pd.DataFrame:
    data = {y: load(y) for y in SEAS}
    out = []
    for held in SEAS:
        rat = {}
        for y in SEAS:
            if y == held:
                continue
            _, ro, _ = data[y]
            st = ro[~ro["lineup_slot"].isin(["BE", "IR"]) & (ro["proj"] > 0) & ro["week"].between(3, 14)]
            for pos, g in st.groupby("position"):
                rat.setdefault(pos, []).extend((g["pts"] / g["proj"]).tolist())
        Q = {p: np.quantile(v, np.linspace(0.01, 0.99, 50)) for p, v in rat.items()}
        s, ro, tw = data[held]
        act = tw.set_index(["week", "team_id"])["points"]
        for w, r in ro.groupby("week"):
            st = r[~r["lineup_slot"].isin(["BE", "IR"])]
            draws = {}
            for t, g in st.groupby("team_id"):
                tot = np.zeros(SIMS)
                for pos, pj in zip(g["position"], g["proj"]):
                    if pj <= 0:
                        continue
                    q = Q.get(pos, Q["WR"])
                    u = RNG.random(SIMS) * (len(q) - 1)
                    lo = np.floor(u).astype(int)
                    tot += pj * (q[lo] + (q[np.minimum(lo + 1, len(q) - 1)] - q[lo]) * (u - lo))
                draws[t] = tot
            ts = sorted(draws)
            for i, a in enumerate(ts):
                for b in ts[i + 1:]:
                    if (w, a) not in act or (w, b) not in act or act[(w, a)] == act[(w, b)]:
                        continue
                    p = float((draws[a] > draws[b]).mean())
                    sd = float(np.std(draws[a] - draws[b]))
                    out.append({"season": held, "week": w, "p": p, "y": float(act[(w, a)] > act[(w, b)]), "sd": sd,
                                "dproj": float(draws[a].mean() - draws[b].mean()), "dact": float(act[(w, a)] - act[(w, b)])})
    return pd.DataFrame(out)


def main() -> None:
    R = regret()
    tw = pd.concat([pd.read_csv(ROOT / "history" / "actuals" / str(y) / "team_weekly.csv").assign(season=y) for y in SEAS])
    chk = R.merge(tw, left_on=["season", "week", "team"], right_on=["season", "week", "team_id"])
    print(f"check: our started-lineup points match ESPN team scores within 0.5 in {(abs(chk['actual'] - chk['points']) < 0.5).mean():.0%} of {len(chk)} team-weeks")
    print(f"A. LINEUP REGRET ({len(R)} team-weeks; points vs hindsight-best lineup, per team-week)")
    for col, lab in (("reg_actual", "managers' actual lineups"), ("reg_espn", "start-ESPN's-projections lineup")):
        v = wboot(R, lambda d: d[col].mean())
        print(f"  {lab:32} regret {v[0]:5.1f} [{v[1]:.1f},{v[2]:.1f}] pts/week")
    v = wboot(R, lambda d: d["vs_espn"].mean())
    print(f"  managers minus ESPN lineup: {v[0]:+.2f} [{v[1]:+.2f},{v[2]:+.2f}] pts/week; managers beat ESPN's lineup in "
          f"{(R['vs_espn'] > 0).mean():.0%} of team-weeks, tie {(R['vs_espn'] == 0).mean():.0%}")
    by = R.groupby("team").agg(weeks=("week", "size"), regret=("reg_actual", "mean"), vs_espn=("vs_espn", "mean")).round(2)
    print("  by manager (team id; ids persist across seasons):\n" + by.sort_values("vs_espn").to_string())
    print("  by season, mean managers minus ESPN lineup:", R.groupby("season")["vs_espn"].mean().round(2).to_dict())
    m = R[R["team"] == MY_TEAM]
    print(f"  team {MY_TEAM}: regret {m['reg_actual'].mean():.1f}/wk, vs ESPN lineup {m['vs_espn'].mean():+.2f}/wk "
          f"({m['vs_espn'].sum():+.0f} pts over {len(m)} weeks)")
    print("\nB1. hit_prob calibration (leave-one-season-out, rostered checkpoint rows)")
    hit_calibration()
    W = winprob()
    b, bss = brier(W["y"], W["p"])
    v = wboot(W, lambda d: brier(d["y"], d["p"])[1], 500)
    print(f"\nB2. pregame win probability (live.py method), {len(W)} team pairs: Brier={b:.3f} skill vs coin flip "
          f"{1 - b / 0.25:+.3f}; skill vs base rate {v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}]; calibration slope {cal_slope(W['y'], W['p']):.2f} (1 = right spread)")
    W["bin"] = pd.cut(W["p"], np.linspace(0, 1, 11))
    t = W.groupby("bin", observed=True).agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean")).round(3)
    print(t.to_string())
    k = float(np.std(W["dact"] - W["dproj"]) / W["sd"].mean())
    print(f"  spread check: sd(actual margin - projected margin) / simulated margin sd = {k:.2f} (1 = the sim's spread is right)")


if __name__ == "__main__":
    main()
