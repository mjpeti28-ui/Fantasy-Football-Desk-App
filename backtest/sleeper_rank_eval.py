"""Ranking usefulness beyond Spearman: noise ceiling, disagreement tests, pairwise calibration,
weekly rank churn, top/bottom-3 hit rate, and playoff-odds calibration. Clean Sleeper set.

    .venv/bin/python backtest/sleeper_rank_eval.py      # writes ~/Developer/sleeper-data/rank_eval.md
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).parent))
import sleeper_crawl as sc  # noqa: E402
import sleeper_experiments as se  # noqa: E402
import sleeper_power as sp  # noqa: E402

BLEND_A = {4: 0.70, 6: 0.40, 8: 0.55, 10: 0.40}  # R5 weights on strength (in-sample)
ACT_W = 0.09  # z-weight of first-half activity (key test coef)
CANDS = ("record", "pf_g", "allplay", "strength", "blend", "blend_act", "orig3")
SIMS = 2000
RNG = np.random.default_rng(1)


def a_for(w):
    return BLEND_A[min(BLEND_A, key=lambda c: abs(c - w))]


def zs(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x * 0


def ci(v, reps=2000):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    b = v[RNG.integers(0, len(v), (reps, len(v)))].mean(1)
    return f"{v.mean():.3f} [{np.percentile(b, 2.5):.3f},{np.percentile(b, 97.5):.3f}]"


def sp_corr(a, b):
    return np.corrcoef(rankdata(a), rankdata(b))[0, 1]


def pairwise(x, y):
    i, j = np.triu_indices(len(y), 1)
    dx, dy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
    ok = (dx != 0) & (dy != 0)
    return np.mean(dx[ok] == dy[ok]) if ok.any() else np.nan


def topk(x, y, k, low=False):
    s = 1 if low else -1
    return len(set(np.argsort(s * x)[:k]) & set(np.argsort(s * y)[:k])) / k


def league_eval(m, lg, P, S, info, avail, acc):
    s = m["season"]
    rows = pd.DataFrame(se.league(m, lg, P, S, info, avail, {})[0])
    reg_end = min((m["playoff_week_start"] or 15) - 1, 17)
    slots = [x for x in lg["roster_positions"] if x in sp.SLOT_ELIG]
    mu = {w: sc.get(f"{sc.BASE}/league/{m['league_id']}/matchups/{w}", offline=True) or [] for w in range(1, reg_end + 1)}
    score = {w: {e["roster_id"]: sum(S[s][w].get(p, (0, 0))[0] for p in e["starters"] or [] if p not in ("0", None))
                 for e in mu[w]} for w in mu}
    rids = sorted(score[1])
    idx = {r: i for i, r in enumerate(rids)}
    n = len(rids)
    pairs = {}
    for w in mu:
        g = collections.defaultdict(list)
        for e in mu[w]:
            if e.get("matchup_id") is not None:
                g[e["matchup_id"]].append(idx[e["roster_id"]])
        pairs[w] = [tuple(v) for v in g.values() if len(v) == 2]
    Sc = np.array([[score[w][r] for r in rids] for w in range(1, reg_end + 1)])  # (weeks, teams)
    H = np.zeros_like(Sc)  # h2h wins per week
    for w in range(1, reg_end + 1):
        for a, b in pairs[w]:
            H[w - 1, a] = 1.0 if Sc[w - 1, a] > Sc[w - 1, b] else 0.5 if Sc[w - 1, a] == Sc[w - 1, b] else 0.0
            H[w - 1, b] = 1.0 - H[w - 1, a]
    AP = np.array([(rankdata(row) - 1) / (n - 1) for row in Sc])
    final_rank = np.lexsort((-Sc.sum(0), -H.sum(0)))
    made = np.zeros(n)
    made[final_rank[: m["playoff_teams"] or 6]] = 1
    sd_week = Sc.std(0).mean()

    # ---- weekly candidate scores (for churn), weeks 3..reg_end-2
    wk = {}
    for w in range(3, reg_end - 1):
        roster = {e["roster_id"]: e["players"] or [] for e in mu[w]}
        V = {**info[s][w]["projmean"], **{p: v for p, (v, _) in P[s][w + 1].items()}}
        st = np.array([se.lineup(roster[r], V, slots)[0] for r in rids])
        pf = Sc[:w].mean(0)
        a = a_for(w)
        wk[w] = {"record": H[:w].mean(0), "pf_g": pf, "allplay": AP[:w].mean(0), "strength": st,
                 "blend": a * zs(st) + (1 - a) * zs(pf), "_zst": zs(st), "_zpf": zs(pf), "_a": a}
    for w in wk:
        if w - 1 in wk:
            for c in ("record", "pf_g", "allplay", "strength", "blend"):
                acc["churn", c].append(np.abs(rankdata(-wk[w][c]) - rankdata(-wk[w - 1][c])).mean())
            a = wk[w]["_a"]
            dst = a * (wk[w]["_zst"] - wk[w - 1]["_zst"])
            dpf = (1 - a) * (wk[w]["_zpf"] - wk[w - 1]["_zpf"])
            acc["var_st"].append(np.var(dst))
            acc["var_pf"].append(np.var(dpf))
            # do score-only moves predict ROS beyond the current blend?
            ros = Sc[w:].mean(0)
            resid = zs(ros) - np.polyval(np.polyfit(wk[w]["blend"], zs(ros), 1), wk[w]["blend"])
            acc["scoremove_vs_resid"].append(sp_corr(dpf, resid))
            acc["rostermove_vs_resid"].append(sp_corr(dst, resid))

    # ---- checkpoint metrics
    for W, g in rows.groupby("cp"):
        g = g.set_index("rid").loc[rids]
        a = BLEND_A[W]
        X = {"record": H[:W].mean(0), "pf_g": g.pf_g.values, "allplay": AP[:W].mean(0),
             "strength": g.next1_fill.values, "orig3": g.orig3.values}
        X["blend"] = a * zs(X["strength"]) + (1 - a) * zs(X["pf_g"])
        X["blend_act"] = X["blend"] + ACT_W * zs(g.fh_S5_activity.fillna(0).values)
        blend_pts = a * X["strength"] + (1 - a) * X["pf_g"]
        y = Sc[W:].mean(0)
        ros_w = H[W:].sum(0)
        # ceiling: truth = ROS mean; realization = resample each team's own ROS weeks
        R = Sc[W:]
        draws = R[RNG.integers(0, len(R), (300, len(R))), :].mean(1)
        acc["ceil", "spearman"].append(np.mean([sp_corr(y, d) for d in draws]))
        acc["ceil", "pairwise"].append(np.mean([pairwise(y, d) for d in draws[:100]]))
        acc["ceil", "top3"].append(np.mean([topk(y, d, 3) for d in draws]))
        acc["ceil", "bottom3"].append(np.mean([topk(y, d, 3, True) for d in draws]))
        for c in CANDS:
            x = X[c]
            acc["spearman", c].append(sp_corr(x, y))
            acc["pairwise", c].append(pairwise(x, y))
            acc["top3", c].append(topk(x, y, 3))
            acc["bottom3", c].append(topk(x, y, 3, True))
        # disagreement: where candidate and reference order a pair differently
        i, j = np.triu_indices(n, 1)
        for c in ("pf_g", "allplay", "strength", "blend", "blend_act"):
            for ref in ("record", "pf_g"):
                if c == ref:
                    continue
                dc, dr = np.sign(X[c][i] - X[c][j]), np.sign(X[ref][i] - X[ref][j])
                dis = (dc != 0) & (dr != 0) & (dc != dr)
                if dis.any():
                    pick_hi = np.where(dc[dis] > 0, i[dis], j[dis])
                    pick_lo = np.where(dc[dis] > 0, j[dis], i[dis])
                    acc["dis", c, ref, "pf"].append((np.sum(y[pick_hi] > y[pick_lo]), dis.sum(), m["league_id"]))
                    acc["dis", c, ref, "w"].append((np.sum(ros_w[pick_hi] > ros_w[pick_lo]),
                                                    np.sum(ros_w[pick_hi] != ros_w[pick_lo]), m["league_id"]))
        for ii, jj in zip(i, j):
            if y[ii] != y[jj]:
                acc["gap"].append((s, W, blend_pts[ii] - blend_pts[jj], float(y[ii] > y[jj]), m["league_id"]))
        # playoff odds Monte Carlo
        idx_ = np.arange(n)
        for mdl, k, base in (("coinflip", 0.0, None), ("pf_g", 0.45, "pf_g"), ("blend", 0.50, "blend"),
                             ("blend_overconf", 1.0, "blend")):
            z = np.zeros(n) if base is None else zs(X[base])
            mu_t = X["pf_g"].mean() + k * z * y.std()
            wins = np.tile(H[:W].sum(0), (SIMS, 1))
            pf = np.tile(Sc[:W].sum(0), (SIMS, 1))
            for w in range(W + 1, reg_end + 1):
                sim = mu_t + RNG.normal(0, sd_week, (SIMS, n))
                pf += sim
                for a_, b_ in pairs[w]:
                    wins[:, a_] += sim[:, a_] > sim[:, b_]
                    wins[:, b_] += sim[:, b_] > sim[:, a_]
            top = np.argsort(-(wins * 1e6 + pf), axis=1)[:, : m["playoff_teams"] or 6]
            prob = np.array([(top == t).any(1).mean() for t in idx_])
            acc["calib"] += [(mdl, W, p, mm) for p, mm in zip(prob, made)]


def main():
    seasons = list(sc.SEASONS)
    P = {s: {w: se.load("projections", s, w) for w in range(1, 18)} for s in seasons}
    S = {s: {w: se.load("stats", s, w) for w in range(1, 18)} for s in seasons}
    info, avail = se.player_info(P, S), se.availability(S)
    objs = sp.league_objects()
    kept = json.loads((sc.DATA / "kept_leagues.json").read_text())
    qual = {m["league_id"]: sc.analyze_league(m) for m in kept}
    clean = [m for m in kept if qual[m["league_id"]]["complete"] and m["type"] in (0, 1, 2)
             and 8 <= m["teams"] <= 14 and qual[m["league_id"]]["inactive_strict"] == 0]
    acc = collections.defaultdict(list)
    for k, m in enumerate(clean):
        league_eval(m, objs[m["league_id"]], P, S, info, avail, acc)
        if k % 200 == 0:
            print(f"{k}/{len(clean)}", flush=True)

    L = ["# Ranking usefulness (clean set)\n",
         f"{len(clean)} league-seasons; checkpoints {sp.CHECKPOINTS}. Target ROS PF/g (PPR rescored). "
         f"strength = leak-free next-week lineup projection (byes filled); blend weights {BLEND_A} (in-sample, R5); "
         f"blend_act = blend + {ACT_W} z(first-half adds/wk); orig3 = leaky near-oracle (reference). "
         "CIs: bootstrap over league-checkpoints (per-league for disagreement).\n",
         "## 1/3/5. Accuracy vs noise ceiling\n",
         "Ceiling = true ROS mean vs a resampled ROS realization from each team's own weeks.\n",
         "| metric | ceiling | " + " | ".join(CANDS) + " |", "|---|---|" + "---|" * len(CANDS)]
    for met in ("spearman", "pairwise", "top3", "bottom3"):
        L.append(f"| {met} | {np.nanmean(acc['ceil', met]):.3f} | " +
                 " | ".join(f"{np.nanmean(acc[met, c]):.3f}" for c in CANDS) + " |")
    L.append("| spearman % of ceiling | 100% | " + " | ".join(
        f"{np.nanmean(acc['spearman', c]) / np.nanmean(acc['ceil', 'spearman']):.0%}" for c in CANDS) + " |")
    L.append("\nSpearman 95% CI: " + "; ".join(f"{c} {ci(acc['spearman', c])}" for c in CANDS) + "\n")

    L += ["## 2. Disagreement test: when candidate and reference order a pair differently, candidate's pick wins\n",
          "| candidate | vs | n pairs | higher ROS PF/g | more ROS h2h wins |", "|---|---|---|---|---|"]
    for key in sorted({k[1:3] for k in acc if k[0] == "dis"}):
        out = []
        for t in ("pf", "w"):
            d = pd.DataFrame(acc[("dis",) + key + (t,)], columns=["hit", "n", "lg"]).groupby("lg").sum()
            boots = []
            for _ in range(1000):
                b = d.iloc[RNG.integers(0, len(d), len(d))]
                boots.append(b.hit.sum() / b.n.sum())
            out.append((d.n.sum(), f"{d.hit.sum() / d.n.sum():.3f} [{np.percentile(boots, 2.5):.3f},"
                                   f"{np.percentile(boots, 97.5):.3f}]"))
        L.append(f"| {key[0]} | {key[1]} | {out[0][0]} | {out[0][1]} | {out[1][1]} |")

    gp = pd.DataFrame(acc["gap"], columns=["season", "cp", "gap", "win", "lg"])
    def fit(tr):  # 1-param logistic P = sigmoid(b*gap), Newton
        b = 0.05
        for _ in range(25):
            p = 1 / (1 + np.exp(-b * tr.gap))
            b += np.sum((tr.win - p) * tr.gap) / np.sum(p * (1 - p) * tr.gap ** 2)
        return b

    gp["p"] = 0.5
    for s in gp.season.unique():  # LOSO by season
        gp.loc[gp.season == s, "p"] = 1 / (1 + np.exp(-fit(gp[gp.season != s]) * gp.loc[gp.season == s, "gap"]))
    b_all = fit(gp)
    L += ["\n## 3. Pairwise calibration: P(A beats B in ROS PF/g) from blend gap in pts/wk (LOSO logistic)\n",
          f"Brier {np.mean((gp.p - gp.win) ** 2):.4f} (coin flip 0.25); slope b={b_all:.3f}/pt. "
          "Gap for P=60/70/80%: " + ", ".join(f"{q:.0%}: {np.log(q / (1 - q)) / b_all:.1f} pts/wk" for q in (.6, .7, .8)) + "\n",
          "| decile | mean gap | predicted | actual | n |", "|---|---|---|---|---|"]
    gp["dec"] = pd.qcut(gp.p, 10, labels=False, duplicates="drop")
    for d, x in gp.groupby("dec"):
        L.append(f"| {d} | {x.gap.mean():+.1f} | {x.p.mean():.3f} | {x.win.mean():.3f} | {len(x)} |")

    L += ["\n## 4. Weekly rank churn (mean |rank change| week to week)\n",
          ", ".join(f"{c} {np.mean(acc['churn', c]):.2f}" for c in ("record", "pf_g", "allplay", "strength", "blend")),
          f"\nBlend score change variance from roster/projection moves {np.mean(acc['var_st']):.3f} vs from "
          f"last week's score {np.mean(acc['var_pf']):.3f} (z units; not additive if correlated).",
          "Do the moves predict ROS beyond the current blend (Spearman with residual)? score-driven "
          f"{ci(acc['scoremove_vs_resid'])}; roster-driven {ci(acc['rostermove_vs_resid'])}\n"]

    cd = pd.DataFrame(acc["calib"], columns=["model", "cp", "p", "made"])
    L += ["## Playoff-odds calibration (Monte Carlo over real remaining schedule, banked wins)\n",
          "| model | Brier | log-loss | 0-10% | 10-30% | 30-50% | 50-70% | 70-90% | 90-100% |", "|---|---|---|---|---|---|---|---|---|"]
    bins = [0, .1, .3, .5, .7, .9, 1.0001]
    for mdl, d in cd.groupby("model", sort=False):
        pc = d.p.clip(0.005, 0.995)
        ll = -np.mean(d.made * np.log(pc) + (1 - d.made) * np.log(1 - pc))
        cells = [f"{b.p.mean():.2f}->{b.made.mean():.2f}" if len(b) else "-"
                 for b in (d[(d.p >= lo) & (d.p < hi)] for lo, hi in zip(bins[:-1], bins[1:]))]
        L.append(f"| {mdl} | {np.mean((d.p - d.made) ** 2):.4f} | {ll:.4f} | " + " | ".join(cells) + " |")
    L.append("\nBins: mean predicted -> actual rate. made = top seeds by rescored h2h wins then PF (not the real bracket).")
    text = "\n".join(L) + "\n"
    (sc.DATA / "rank_eval.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
