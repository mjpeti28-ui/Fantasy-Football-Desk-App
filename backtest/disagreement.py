"""Is there an edge hiding where Desk's weekly number disagrees with ESPN's?

Rows from disagreement_rows.py (LOSO 2021-25). gap = desk - espn, resid = actual - espn.
slope of resid on gap: 0 = Desk's disagreement is noise, 1 = Desk is right, in between = some info.
All CIs are player-season cluster bootstraps; p-values are player-season sign-flip tests
(flip each cluster's sum of gap*resid), Holm-corrected over every slice tested.
  python3 backtest/disagreement.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import valuation as v  # noqa: E402
from backtest.start_sit import choose  # noqa: E402

OUT = ROOT / "history" / "backtest" / "multi"
RNG = np.random.default_rng(7)
B = 2000
START = {"QB": 15, "RB": 36, "WR": 42, "TE": 15}  # "startable range" by ESPN rank that week


def slope_test(d: pd.DataFrame, x="gap", y="resid", key="ps"):
    """No-intercept slope of y on x, cluster bootstrap CI, cluster sign-flip p."""
    g = d.assign(xy=d[x] * d[y], xx=d[x] ** 2).groupby(key)[["xy", "xx"]].sum()
    xy, xx = g["xy"].to_numpy(), g["xx"].to_numpy()
    b = xy.sum() / xx.sum()
    idx = RNG.integers(0, len(g), (B, len(g)))
    bs = xy[idx].sum(1) / xx[idx].sum(1)
    flips = RNG.choice([-1, 1], (B, len(g)))
    p = (np.abs((flips * xy).sum(1)) >= abs(xy.sum())).mean()
    return b, *np.percentile(bs, [2.5, 97.5]), max(p, 1 / B), len(g)


def mean_test(vals: np.ndarray, clusters: np.ndarray):
    df = pd.DataFrame({"v": vals, "c": clusters}).groupby("c")["v"].agg(["sum", "count"])
    s, n = df["sum"].to_numpy(), df["count"].to_numpy()
    m = s.sum() / n.sum()
    idx = RNG.integers(0, len(df), (B, len(df)))
    bs = s[idx].sum(1) / n[idx].sum(1)
    flips = RNG.choice([-1, 1], (B, len(df)))
    p = (np.abs((flips * s).sum(1)) >= abs(s.sum())).mean()
    return m, *np.percentile(bs, [2.5, 97.5]), max(p, 1 / B), len(df)


def holm(ps):
    ps = np.asarray(ps)
    o = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0
    for r, i in enumerate(o):
        run = max(run, min(1, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def lineup_points(d: pd.DataFrame, cols) -> pd.DataFrame:
    """Actual points of the lineup each score column would set, per team-week."""
    slots = {y: v.load_season(y).slots for y in d["season"].unique()}
    out = []
    for (y, w, t), g in d[d["team_id"] > 0].groupby(["season", "week", "team_id"]):
        act = dict(zip(g["espn_id"], g["actual"]))
        cands = list(zip(g["espn_id"].astype(int), g["position"].replace({"D/ST": "DST"})))
        rec = {"season": y, "week": w, "team": t}
        for c in cols:
            rec[c] = sum(act[i] for i in choose([(i, p, s) for (i, p), s in zip(cands, g[c])], slots[y]))
        out.append(rec)
    return pd.DataFrame(out)


def lineup_flips(d: pd.DataFrame) -> pd.DataFrame:
    slots = {y: v.load_season(y).slots for y in d["season"].unique()}
    out = []
    for (y, w, t), g in d[d["team_id"] > 0].groupby(["season", "week", "team_id"]):
        act = dict(zip(g["espn_id"], g["actual"]))
        cands = list(zip(g["espn_id"].astype(int), g["position"].replace({"D/ST": "DST"})))
        pe = choose([(i, p, s) for (i, p), s in zip(cands, g["espn"])], slots[y])
        pd_ = choose([(i, p, s) for (i, p), s in zip(cands, g["desk"])], slots[y])
        out.append({"season": y, "week": w, "team": t, "changed": set(pe) != set(pd_),
                    "n_swaps": len(set(pd_) - set(pe)),
                    "espn": sum(act[i] for i in pe), "desk": sum(act[i] for i in pd_)})
    return pd.DataFrame(out)


if __name__ == "__main__":
    raw = pd.read_csv(OUT / "disagreement_rows.csv")
    d = raw[raw["position"].isin(START) & (raw["espn"] > 0)].copy()
    d["rank"] = d.groupby(["season", "week", "position"])["espn"].rank(ascending=False, method="first")
    d["relevant"] = (d["team_id"] > 0) | (d["rank"] <= d["position"].map(START))
    d = d[d["relevant"]].copy()
    d["gap"], d["resid"] = d["desk"] - d["espn"], d["actual"] - d["espn"]
    d["gap_level"] = d["est_ppg"] - d["espn"]              # Desk's level vs ESPN
    d["gap_avail"] = d["desk"] - d["est_ppg"]              # Desk's availability discount
    d["gap_ours"] = d["ours_ppg"] * d["avail_now"] - d["espn"]
    d["ps"] = d["season"].astype(str) + "_" + d["espn_id"].astype(str)
    d["espn_move"] = (d["espn"] - d["espn_prev"]).abs()
    q90 = d["gap"].abs().quantile(0.9)
    mv75 = d["espn_move"].quantile(0.75)
    print(f"relevant rows {len(d)}, player-seasons {d['ps'].nunique()}, |gap| p90 = {q90:.2f}, espn move p75 = {mv75:.2f}")
    for c in ("espn", "desk"):
        print(f"  MAE {c}: {(d['actual'] - d[c]).abs().mean():.3f}")

    # --- A. does the disagreement carry information? (slices)
    slices = {
        "all relevant": d,
        **{f"pos {p}": d[d["position"] == p] for p in START},
        "Desk higher": d[d["gap"] > 0], "Desk lower": d[d["gap"] < 0],
        "big gaps (top 10% |gap|)": d[d["gap"].abs() >= q90],
        "rostered starters": d[~d["slot"].isin(["BE", "FA"])],
        "bench": d[d["slot"] == "BE"], "free agents": d[d["slot"] == "FA"],
        "early (wk 1-4)": d[d["week"] <= 4], "late (wk 10-17)": d[d["week"] >= 10],
        "few games (n<=3)": d[d["n"] <= 3], "established (n>=6)": d[d["n"] >= 6],
        "missed last game": d[d["status"] == "missed_last"],
        "ESPN moved a lot (top 25%)": d[d["espn_move"] >= mv75],
        "ESPN barely moved": d[d["espn_move"] < d["espn_move"].quantile(0.25)],
        "has ECR": d[d["ecr_rank"].notna()], "no ECR": d[d["ecr_rank"].isna()],
    }
    res = []
    for name, s in slices.items():
        b, lo, hi, p, k = slope_test(s)
        res.append({"slice": name, "rows": len(s), "clusters": k, "slope": b, "lo": lo, "hi": hi, "p": p,
                    "mean_gap": s["gap"].mean(), "mean_resid": s["resid"].mean()})
    for comp in ("gap_level", "gap_avail", "gap_ours"):
        b, lo, hi, p, k = slope_test(d, x=comp)
        res.append({"slice": f"component {comp}", "rows": len(d), "clusters": k, "slope": b, "lo": lo, "hi": hi, "p": p,
                    "mean_gap": d[comp].mean(), "mean_resid": d["resid"].mean()})
    for y, s in d.groupby("season"):
        b, lo, hi, p, k = slope_test(s)
        res.append({"slice": f"season {y}", "rows": len(s), "clusters": k, "slope": b, "lo": lo, "hi": hi, "p": p,
                    "mean_gap": s["gap"].mean(), "mean_resid": s["resid"].mean()})
    R = pd.DataFrame(res)
    R["p_holm"] = holm(R["p"])
    pd.set_option("display.width", 200)
    print("\nA. slope of (actual - ESPN) on (Desk - ESPN)")
    print(R.round(3).to_string(index=False))

    # --- B. big disagreements: mean (actual - ESPN) when Desk is far above / below
    print("\nB. top-10% disagreements: mean actual - ESPN (pts), cluster CI")
    for name, s in (("Desk >> ESPN", d[d["gap"] >= q90]), ("Desk << ESPN", d[d["gap"] <= -q90])):
        m, lo, hi, p, k = mean_test(s["resid"].to_numpy(), s["ps"].to_numpy())
        print(f"  {name}: n={len(s)} clusters={k} mean gap={s['gap'].mean():+.2f} resid={m:+.2f} [{lo:+.2f},{hi:+.2f}] p={p:.3f}")
    # who wins the big disagreements, Desk or ESPN (absolute error)
    big = d[d["gap"].abs() >= q90]
    m, lo, hi, p, k = mean_test(((big["actual"] - big["espn"]).abs() - (big["actual"] - big["desk"]).abs()).to_numpy(), big["ps"].to_numpy())
    print(f"  big gaps, ESPN abs err - Desk abs err: {m:+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.3f} (+ = Desk closer)")

    # --- C. start/sit: lineups set by Desk vs by ESPN
    L = lineup_flips(raw)
    L["diff"] = L["desk"] - L["espn"]
    L["ts"] = L["season"].astype(str) + "_" + L["team"].astype(str)
    ch = L[L["changed"]]
    m, lo, hi, p, k = mean_test(L["diff"].to_numpy(), L["ts"].to_numpy())
    print(f"\nC. lineups: {len(L)} team-weeks, Desk lineup differs in {L['changed'].mean():.1%} (avg {ch['n_swaps'].mean():.2f} swaps)")
    print(f"  Desk - ESPN per team-week: {m:+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.3f}")
    m, lo, hi, p, k = mean_test(ch["diff"].to_numpy(), ch["ts"].to_numpy())
    print(f"  in changed weeks only: {m:+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.3f}; Desk wins {(ch['diff'] > 0).mean():.1%}, ties {(ch['diff'] == 0).mean():.1%}")
    print("  per season (changed weeks):", ch.groupby("season")["diff"].mean().round(2).to_dict())

    # --- D. rest-of-season: does the gap predict the next 4 games better than this week?
    a = raw[["season", "espn_id", "week", "actual", "played"]]
    nxt = []
    for k_ in range(0, 4):
        t = a.assign(week=a["week"] - k_)
        nxt.append(t)
    nx = pd.concat(nxt)
    nx = nx[nx["played"]].groupby(["season", "espn_id", "week"])["actual"].agg(["mean", "count"]).reset_index()
    r4 = d.merge(nx, on=["season", "espn_id", "week"], how="inner")
    r4 = r4[r4["count"] >= 2]
    r4["resid4"] = r4["mean"] - r4["espn"]
    r4["gap4"] = r4["est_ppg"] - r4["espn"]
    b, lo, hi, p, k = slope_test(r4, x="gap4", y="resid4")
    print(f"\nD. next-4-games ppg (games played) - ESPN this week, on Desk level - ESPN: slope {b:.3f} [{lo:.3f},{hi:.3f}] p={p:.4f} (n={len(r4)}, clusters={k})")
    for pos, s in r4.groupby("position"):
        b, lo, hi, p, k = slope_test(s, x="gap4", y="resid4")
        print(f"  {pos}: {b:.3f} [{lo:.3f},{hi:.3f}] p={p:.4f}")
    # D2. same, against ESPN's level (mean of its last 3 published weekly numbers, no look-ahead)
    lvl = raw.sort_values("week").groupby(["season", "espn_id"])["espn"].transform(lambda x: x.where(x > 0).rolling(3, min_periods=1).mean())
    r4 = r4.merge(raw.assign(espn_lvl=lvl)[["season", "espn_id", "week", "espn_lvl"]], on=["season", "espn_id", "week"])
    r4["resid4l"], r4["gap4l"] = r4["mean"] - r4["espn_lvl"], r4["est_ppg"] - r4["espn_lvl"]
    b, lo, hi, p, k = slope_test(r4, x="gap4l", y="resid4l")
    print(f"  vs ESPN level (3-wk avg of its weekly numbers): slope {b:.3f} [{lo:.3f},{hi:.3f}] p={p:.4f}")
    for c in ("espn", "espn_lvl", "est_ppg"):
        print(f"    next-4 MAE {c}: {(r4['mean'] - r4[c]).abs().mean():.3f}")

    # --- E. rankings only: demean gap and resid within season-week-position (level bias removed)
    print("\nE. within position-week (demeaned) slope -- what matters for start/sit ordering")
    grp = ["season", "week", "position"]
    d["gap_c"] = d["gap"] - d.groupby(grp)["gap"].transform("mean")
    d["resid_c"] = d["resid"] - d.groupby(grp)["resid"].transform("mean")
    for comp in ("gap_level", "gap_avail", "gap_ours"):
        d[comp + "_c"] = d[comp] - d.groupby(grp)[comp].transform("mean")
    rows_e = []
    for name, s_ in [("all relevant", d), *[(f"pos {p}", d[d["position"] == p]) for p in START],
                     ("rostered starters", d[~d["slot"].isin(["BE", "FA"])]), ("missed last game", d[d["status"] == "missed_last"])]:
        rows_e.append((name, *slope_test(s_, x="gap_c", y="resid_c")))
    for comp in ("gap_level_c", "gap_avail_c", "gap_ours_c"):
        rows_e.append((f"component {comp}", *slope_test(d, x=comp, y="resid_c")))
    E = pd.DataFrame(rows_e, columns=["slice", "slope", "lo", "hi", "p", "clusters"])
    E["p_holm"] = holm(E["p"])
    print(E.round(3).to_string(index=False))

    # --- F. lineups by ESPN + c * (Desk - ESPN); c picked leave-one-season-out
    grid = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    raw["gap"] = raw["desk"] - raw["espn"]
    for c in grid:
        raw[f"c{c}"] = raw["espn"] + c * raw["gap"]
    LP = lineup_points(raw, [f"c{c}" for c in grid])
    LP["ts"] = LP["season"].astype(str) + "_" + LP["team"].astype(str)
    print("\nF. lineup pts/team-week vs pure ESPN by c:", {c: round((LP[f"c{c}"] - LP["c0.0"]).mean(), 3) for c in grid})
    pick = {}
    for y in LP["season"].unique():
        o = LP[LP["season"] != y]
        pick[y] = max(grid, key=lambda c: o[f"c{c}"].mean())
    LP["loso"] = [r[f"c{pick[r['season']]}"] for _, r in LP.iterrows()]
    m, lo, hi, p, k = mean_test((LP["loso"] - LP["c0.0"]).to_numpy(), LP["ts"].to_numpy())
    print(f"  LOSO-picked c {pick}: {m:+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.3f} pts/team-week")
    for c in (0.2, 0.3):
        m, lo, hi, p, k = mean_test((LP[f"c{c}"] - LP["c0.0"]).to_numpy(), LP["ts"].to_numpy())
        print(f"  fixed c={c}: {m:+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.3f}; by season", (LP[f"c{c}"] - LP["c0.0"]).groupby(LP["season"]).mean().round(2).to_dict())
    R.to_csv(OUT / "disagreement_slices.csv", index=False)
