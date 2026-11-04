"""Power-ranking experiments applied to Max's ESPN league (team 3) for face validity.

A: strength smoothing (1/2/3 weeks) on the clean Sleeper set.
B: port to ESPN 2021-25 (next-week ESPN projection lineup, byes filled), blend with PF/g, calibration.
C: per-season wk6/wk10 tables + biggest blend-vs-record disagreements.
D: 2026 power ranking through the latest completed week.

Reads the main checkout's history/actuals read-only (HIST env to override).
    .venv/bin/python backtest/power_my_league.py      # writes ~/Developer/sleeper-data/my_league.md
"""
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).parent))
import sleeper_crawl as sc  # noqa: E402
import sleeper_experiments as se  # noqa: E402
import sleeper_power as sp  # noqa: E402

HIST = Path(os.environ.get("HIST", Path.home() / "Developer" / "Fantasy-Football-Desk-App" / "history" / "actuals"))
BLEND_A = {4: 0.70, 6: 0.40, 8: 0.55, 10: 0.40}
LOGIT_B = 0.074  # Sleeper-fit P(A beats B ROS) = sigmoid(b * blend gap pts/wk)
TIER_GAP = 5.5
ME = 3
ESPN_SLOT = {"QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE", "K": "K", "DST": "DEF",
             "FLEX": "FLEX", "RB/WR": "WRRB_FLEX", "WR/TE": "REC_FLEX"}
RNG = np.random.default_rng(2)


def a_for(w):
    return BLEND_A[min(BLEND_A, key=lambda c: abs(c - w))]


def zs(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / x.std() if x.std() > 0 else x * 0


def sp_corr(a, b):
    return np.corrcoef(rankdata(a), rankdata(b))[0, 1]


def ci(v, reps=2000):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    b = v[RNG.integers(0, len(v), (reps, len(v)))].mean(1)
    return f"{v.mean():.3f} [{np.percentile(b, 2.5):.3f},{np.percentile(b, 97.5):.3f}]"


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ---------------- A: smoothing on Sleeper ----------------

def exp_a(L):
    seasons = list(sc.SEASONS)
    P = {s: {w: se.load("projections", s, w) for w in range(1, 18)} for s in seasons}
    S = {s: {w: se.load("stats", s, w) for w in range(1, 18)} for s in seasons}
    info = se.player_info(P, S)
    objs = sp.league_objects()
    kept = json.loads((sc.DATA / "kept_leagues.json").read_text())
    qual = {m["league_id"]: sc.analyze_league(m) for m in kept}
    clean = [m for m in kept if qual[m["league_id"]]["complete"] and m["type"] in (0, 1, 2)
             and 8 <= m["teams"] <= 14 and qual[m["league_id"]]["inactive_strict"] == 0]
    acc = collections.defaultdict(list)
    for m in clean:
        s, lg = m["season"], objs[m["league_id"]]
        reg_end = min((m["playoff_week_start"] or 15) - 1, 17)
        slots = [x for x in lg["roster_positions"] if x in sp.SLOT_ELIG]
        mu = {w: sc.get(f"{sc.BASE}/league/{m['league_id']}/matchups/{w}", offline=True) or []
              for w in range(1, reg_end + 1)}
        rids = sorted(e["roster_id"] for e in mu[1])
        Sc = np.array([[next(sum(S[s][w].get(p, (0, 0))[0] for p in e["starters"] or [] if p not in ("0", None))
                             for e in mu[w] if e["roster_id"] == r) for r in rids] for w in range(1, reg_end + 1)])
        st = {}
        for w in range(1, reg_end - 1):
            roster = {e["roster_id"]: e["players"] or [] for e in mu[w]}
            V = {**info[s][w]["projmean"], **{p: v for p, (v, _) in P[s][w + 1].items()}}
            st[w] = np.array([se.lineup(roster[r], V, slots)[0] for r in rids])
        prev = {}
        for W in range(3, reg_end - 1):
            pf, ros = Sc[:W].mean(0), Sc[W:].mean(0)
            for k in (1, 2, 3):
                sm = np.mean([st[W - j] for j in range(k)], 0)
                bl = a_for(W) * zs(sm) + (1 - a_for(W)) * zs(pf)
                for name, x in (("strength", sm), ("blend", bl)):
                    rk = rankdata(-x)
                    if (name, k) in prev:
                        acc["churn", name, k].append(np.abs(rk - prev[name, k]).mean())
                    prev[name, k] = rk
                    if W in sp.CHECKPOINTS:
                        i, j = np.triu_indices(len(rids), 1)
                        dx, dy = np.sign(x[i] - x[j]), np.sign(ros[i] - ros[j])
                        ok = (dx != 0) & (dy != 0)
                        acc["spearman", name, k].append(sp_corr(x, ros))
                        acc["pairwise", name, k].append(np.mean(dx[ok] == dy[ok]))
                        acc["top3", name, k].append(len(set(np.argsort(-x)[:3]) & set(np.argsort(-ros)[:3])) / 3)
    L += ["## A. Strength smoothing (clean Sleeper set, n=%d leagues)\n" % len(clean),
          "Smoothed strength at week W = mean of leak-free strength from weeks W-k+1..W (each week's own roster).\n",
          "| version | Spearman | pairwise | top-3 | weekly churn |", "|---|---|---|---|---|"]
    best = max((1, 2, 3), key=lambda k: np.mean(acc["spearman", "blend", k]))
    for name in ("strength", "blend"):
        for k in (1, 2, 3):
            L.append(f"| {name} k={k} | {ci(acc['spearman', name, k])} | {np.mean(acc['pairwise', name, k]):.3f} | "
                     f"{np.mean(acc['top3', name, k]):.3f} | {np.mean(acc['churn', name, k]):.2f} |")
    L.append(f"\nBest blend by Spearman: k={best}. Used for ESPN below.\n")
    return best


# ---------------- ESPN league ----------------

def load_season(y):
    d = HIST / str(y)
    st = json.loads((d / "settings.json").read_text())
    slots = [ESPN_SLOT[k] for k, n in st["lineup_slot_counts"].items() if k in ESPN_SLOT for _ in range(n)]
    tw = pd.read_csv(d / "team_weekly.csv")
    mt = pd.read_csv(d / "matchups.csv")
    rw = pd.read_csv(d / "rosters_by_week.csv")
    pj = pd.read_csv(d / "espn_weekly_proj.csv")
    for pid, pos in zip(rw.espn_id, rw.position):
        sp.POS.setdefault(f"e{pid}", set()).add("DEF" if pos == "DST" else pos)
    proj = {w: {f"e{i}": v for i, v in zip(g.espn_id, g.espn_proj)} for w, g in pj.groupby("week")}
    roster = {(w, t): [f"e{i}" for i in g.espn_id] for (w, t), g in rw.groupby(["week", "team_id"])}
    reg = st["regular_season_weeks"]
    played = tw[(tw.week <= reg) & (tw.points > 0)].week.max()
    teams = sorted(tw.team_id.unique())
    Sc = tw[tw.week <= played].pivot(index="week", columns="team_id", values="points").loc[:, teams].values
    H = np.zeros_like(Sc)
    ix = {t: i for i, t in enumerate(teams)}
    for w, g in mt[mt.week <= played].groupby("week"):
        for h, a in zip(g.home_id, g.away_id):
            i, j = ix[h], ix[a]
            si, sj = Sc[w - 1, i], Sc[w - 1, j]
            H[w - 1, i] = 1.0 if si > sj else 0.5 if si == sj else 0.0
            H[w - 1, j] = 1.0 - H[w - 1, i]
    psum, pn, pm = collections.Counter(), collections.Counter(), {}
    for w in range(1, max(proj) + 1):
        for p, v in proj.get(w, {}).items():
            psum[p] += v
            pn[p] += 1
        pm[w] = {p: psum[p] / pn[p] for p in pn}

    def strength(W, roster_week=None):
        rk = roster_week or W
        V = {**pm[W], **proj.get(W + 1, {})}
        return np.array([se.lineup(roster.get((rk, t), []), V, slots)[0] for t in teams])
    return {"st": st, "teams": teams, "Sc": Sc, "H": H, "reg": reg, "played": played, "strength": strength,
            "names": st["team_names"], "rw": rw, "pj": pj}


def team_table(d, W, k, roster_week=None):
    """Blend etc. at checkpoint W (weeks 1..W played)."""
    Sc, H, teams = d["Sc"], d["H"], d["teams"]
    sm = np.mean([d["strength"](W - j, roster_week if j == 0 else None) for j in range(k)], 0)
    pf = Sc[:W].mean(0)
    a = a_for(W)
    ap = np.array([(rankdata(r) - 1) / (len(teams) - 1) for r in Sc[:W]]).sum(0)
    t = pd.DataFrame({"team": teams, "name": [d["names"][str(x)].strip() for x in teams],
                      "W": H[:W].sum(0), "L": W - H[:W].sum(0), "pf_g": pf, "strength": sm,
                      "blend": a * sm + (1 - a) * pf, "allplay_w": ap})
    t["luck"] = t.W - t.allplay_w
    t = t.sort_values("blend", ascending=False).reset_index(drop=True)
    t["rank"] = np.arange(1, len(t) + 1)
    t["tier"] = 1 + np.concatenate([[0], np.cumsum(-np.diff(t.blend.values) >= TIER_GAP)])
    t["rec_rank"] = rankdata(-(t.W.values * 1e4 + t.pf_g.values), method="ordinal").astype(int)
    if W < Sc.shape[0]:
        t["ros_pf_g"] = [Sc[W:, teams.index(x)].mean() for x in t.team]
    return t


def exp_b(L, k):
    fs = pd.read_csv(HIST / "final_standings.csv")
    acc, gaps, C = collections.defaultdict(list), [], []
    for y in range(2021, 2026):
        d = load_season(y)
        for W in sp.CHECKPOINTS:
            t = team_table(d, W, k)
            y_ = t.ros_pf_g.values
            for c in ("rec", "pf_g", "strength", "blend"):
                x = -t.rec_rank.values if c == "rec" else t[c].values
                acc["spearman", c].append(sp_corr(x, y_))
                i, j = np.triu_indices(len(t), 1)
                dx, dy = np.sign(x[i] - x[j]), np.sign(y_[i] - y_[j])
                ok = (dx != 0) & (dy != 0)
                acc["pairwise", c].append(np.mean(dx[ok] == dy[ok]))
            i, j = np.triu_indices(len(t), 1)
            for a_, b_ in zip(i, j):
                if y_[a_] != y_[b_]:
                    gaps.append((t.blend[a_] - t.blend[b_], float(y_[a_] > y_[b_])))
            if W in (6, 10):
                f = fs[fs.season == y].set_index("team_id")
                t["seed"], t["final"] = t.team.map(f.seed), t.team.map(f.final)
                C.append((y, W, t))
    L += ["## B. ESPN port, 2021-25 (12 teams x 5 seasons x 4 checkpoints; tiny n, cannot confirm anything)\n",
          "| predictor | Spearman vs ROS PF/g | pairwise |", "|---|---|---|"]
    for c in ("rec", "pf_g", "strength", "blend"):
        L.append(f"| {c} | {ci(acc['spearman', c])} | {np.mean(acc['pairwise', c]):.3f} |")
    g = pd.DataFrame(gaps, columns=["gap", "win"])
    g["absgap"], g["hit"] = g.gap.abs(), np.where(g.gap >= 0, g.win, 1 - g.win)
    g["bucket"] = pd.cut(g.absgap, [0, 5.5, 11.4, 18.7, 1e9], right=False,
                         labels=["0-5.5", "5.5-11.4", "11.4-18.7", "18.7+"])
    L += ["\nCalibration of the Sleeper-fit gap->odds curve on this league (pairs are correlated):\n",
          "| blend gap pts/wk | n pairs | Sleeper-predicted | actual (higher-blend team has higher ROS PF/g) |",
          "|---|---|---|---|"]
    for bkt, x in g.groupby("bucket", observed=True):
        L.append(f"| {bkt} | {len(x)} | {sigmoid(LOGIT_B * x.absgap).mean():.2f} | {x.hit.mean():.2f} |")
    L.append("")
    return C


def exp_c(L, C):
    L += ["## C. Face validity by season (wk 6 and wk 10)\n",
          "Biggest calls: blend rank vs record rank differ by >= 3. Right = closer to ROS PF/g rank.\n"]
    for y, W, t in C:
        t = t.copy()
        t["ros_rank"] = rankdata(-t.ros_pf_g.values, method="ordinal").astype(int)
        L.append(f"### {y} wk {W}\n")
        L.append("| blend rank (tier) | team | record | rec rank | ROS PF/g rank | seed | final |")
        L.append("|---|---|---|---|---|---|---|")
        for _, r in t.iterrows():
            L.append(f"| {r['rank']} (T{r.tier}) | {r['name']}{' *' if r.team == ME else ''} | {r.W:g}-{r.L:g} | "
                     f"{r.rec_rank} | {r.ros_rank} | {r.seed} | {r.final} |")
        dis = t[(t["rank"] - t.rec_rank).abs() >= 3].copy()
        dis["gap"] = (dis["rank"] - dis.rec_rank).abs()
        calls = []
        for _, r in dis.sort_values("gap", ascending=False).head(3).iterrows():
            who = "blend" if abs(r["rank"] - r.ros_rank) < abs(r.rec_rank - r.ros_rank) else \
                "record" if abs(r["rank"] - r.ros_rank) > abs(r.rec_rank - r.ros_rank) else "tie"
            calls.append(f"{r['name']}: blend #{r['rank']} vs record #{r.rec_rank} -> ROS #{r.ros_rank}, seed {r.seed} "
                         f"({who + ' right' if who != 'tie' else 'tie'})")
        L.append("\nBig calls: " + ("; ".join(calls) if calls else "none") + "\n")


def exp_d(L, k):
    d = load_season(2026)
    W = int(d["played"])
    last_roster = int(d["rw"].week.max())
    now = team_table(d, W, k, roster_week=last_roster)
    prev = team_table(d, W - 1, k).set_index("team")
    me = now[now.team == ME].iloc[0]
    rw = d["rw"]
    adds = collections.Counter()
    for t in d["teams"]:
        for w in range(2, last_roster + 1):
            a_ = set(rw[(rw.week == w) & (rw.team_id == t)].espn_id) - set(rw[(rw.week == w - 1) & (rw.team_id == t)].espn_id)
            adds[t] += len(a_)
    a = a_for(W)
    L += [f"## D. 2026 power ranking through week {W} (history/actuals/2026; roster as of week {last_roster}, "
          f"strength = ESPN week {W + 1} projection lineup, smoothed k={k}; blend = {a:.2f} strength + "
          f"{1 - a:.2f} PF/g)\n",
          "| rank (tier) | team | record | PF/g | strength | blend | luck (W - all-play W) | chg vs wk %d | driver | "
          "P(Stinkers higher ROS PF/g) | adds/wk |" % (W - 1),
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in now.iterrows():
        p = prev.loc[r.team]
        chg = int(p["rank"] - r["rank"])
        dst, dpf = a * (r.strength - p.strength), (1 - a) * (r.pf_g - p.pf_g)
        driver = "-" if chg == 0 else ("roster/proj" if abs(dst) > abs(dpf) else "scoring")
        pb = "-" if r.team == ME else f"{sigmoid(LOGIT_B * (me.blend - r.blend)):.0%}"
        L.append(f"| {r['rank']} (T{r.tier}) | {r['name']}{' *' if r.team == ME else ''} | {r.W:g}-{r.L:g} | "
                 f"{r.pf_g:.1f} | {r.strength:.1f} | {r.blend:.1f} | {r.luck:+.1f} | {chg:+d} | {driver} | {pb} | "
                 f"{adds[r.team] / (last_roster - 1):.1f} |")
    L.append("\nadds/wk inferred from week-to-week roster diffs (includes trades). Tier break = gap >= 5.5 pts/wk "
             "(~60% pairwise). P(...) is the Sleeper-fit pairwise curve, not head-to-head game odds.\n")


def proj_accuracy(L):
    out = []
    for pos in ("QB", "RB", "WR", "TE", "K", "DST"):
        x, y = [], []
        for yr in range(2021, 2026):
            pw = pd.read_csv(HIST / str(yr) / "players_weekly.csv")
            pw = pw[(pw.kind == "actual") & (pw.position == pos)]
            pj = pd.read_csv(HIST / str(yr) / "espn_weekly_proj.csv")
            m = pj.merge(pw, on=["espn_id", "week"])
            m = m[m.espn_proj >= 1]
            x += list(m.espn_proj)
            y += list(m.points)
        out.append(f"{pos} {np.corrcoef(x, y)[0, 1]:.2f} (n={len(x)})")
    L.append("## ESPN projection accuracy 2021-25 (Pearson, proj >= 1, players with an actual row; Sleeper "
             "numbers counted missing as 0, so not identical footing)\n\n" + ", ".join(out) +
             "\n\nSleeper (earlier): QB .39, RB .65, WR .61, TE .58, K .12, DEF .23\n")


def main():
    L = ["# Power-ranking experiments applied to my ESPN league\n"]
    k = exp_a(L)
    C = exp_b(L, k)
    exp_d(L, k)
    exp_c(L, C)
    proj_accuracy(L)
    text = "\n".join(L) + "\n"
    (sc.DATA / "my_league.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
