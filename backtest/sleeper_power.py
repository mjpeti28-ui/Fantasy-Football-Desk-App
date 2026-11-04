"""Pilot backtest: does roster strength predict rest-of-season results better than
record / points-for? Uses leagues cached by sleeper_crawl.py.

Everything is rescored to full PPR from Sleeper's weekly stat lines (stats pts_ppr per
starter). "Made playoffs" uses the league's own scoring/standings.

    .venv/bin/python backtest/sleeper_power.py
"""
import collections
import glob
import gzip
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, rankdata

sys.path.insert(0, str(Path(__file__).parent))
import sleeper_crawl as sc  # noqa: E402

CHECKPOINTS = (4, 6, 8, 10)
LOOKAHEAD = 3  # avg best-lineup projection over the next 3 weeks
PREDICTORS = ("record", "pf_g", "allplay", "strength", "blend")
TARGETS = ("ros_allplay", "ros_pf_g", "ros_h2h", "made_playoffs")
FAIR_TARGETS = ("final_wins", "made_playoffs")
DIFFS = (("strength", "pf_g"), ("strength", "allplay"), ("blend", "strength"))
SLOT_ELIG = {
    "QB": {"QB"}, "RB": {"RB"}, "WR": {"WR"}, "TE": {"TE"}, "K": {"K"}, "DEF": {"DEF"},
    "DL": {"DL"}, "LB": {"LB"}, "DB": {"DB"},
    "FLEX": {"RB", "WR", "TE"}, "WRRB_FLEX": {"WR", "RB"}, "REC_FLEX": {"WR", "TE"},
    "SUPER_FLEX": {"QB", "RB", "WR", "TE"}, "IDP_FLEX": {"DL", "LB", "DB"},
}
POS = {}  # player_id -> set(fantasy_positions)


def weekly(kind, season, week):
    """{player_id: pts_ppr} from Sleeper's projections or stats endpoint (cached)."""
    rows = sc.get(f"https://api.sleeper.app/{kind}/nfl/{season}/{week}?season_type=regular") or []
    out = {}
    for r in rows:
        pts = (r.get("stats") or {}).get("pts_ppr")
        pl = r.get("player") or {}
        POS.setdefault(r["player_id"], set()).update(pl.get("fantasy_positions") or [pl.get("position")])
        if pts is not None:
            out[r["player_id"]] = pts
    return out


def best_lineup(players, proj, slots):
    left = sorted(players, key=lambda p: -proj.get(p, 0.0))
    total = 0.0
    for slot in sorted(slots, key=lambda s: len(SLOT_ELIG[s])):
        for p in left:
            if POS.get(p, set()) & SLOT_ELIG[slot]:
                total += proj.get(p, 0.0)
                left.remove(p)
                break
    return total


def league_objects():
    objs = {}
    for f in glob.glob(str(sc.CACHE / "*leagues_nfl_*")):
        for lg in json.load(gzip.open(f, "rt"))["body"] or []:
            objs[lg["league_id"]] = lg
    return objs


def allplay(scores):
    """{rid: score} -> {rid: share of other teams beaten (ties 0.5)}."""
    v = list(scores.values())
    n = len(v) - 1
    return {r: (sum(s > o for o in v) + 0.5 * (sum(s == o for o in v) - 1)) / n for r, s in scores.items()}


def league_rows(m, lg, proj, act):
    season, lid = m["season"], m["league_id"]
    reg_end = (m["playoff_week_start"] or 15) - 1
    slots = [s for s in lg["roster_positions"] if s in SLOT_ELIG]
    weeks = {w: sc.get(f"{sc.BASE}/league/{lid}/matchups/{w}", offline=True) or [] for w in range(1, reg_end + 1)}
    score, lscore, h2h, lh2h, ap = {}, {}, {}, {}, {}
    for w, mu in weeks.items():
        a = act[season][w]
        score[w] = {e["roster_id"]: sum(a.get(p, 0.0) for p in e["starters"] or [] if p not in ("0", None))
                    for e in mu}
        lscore[w] = {e["roster_id"]: e.get("points") or 0.0 for e in mu}
        ap[w] = allplay(score[w])
        games = collections.defaultdict(list)
        for e in mu:
            if e.get("matchup_id") is not None:
                games[e["matchup_id"]].append(e["roster_id"])
        for res, sc_ in ((h2h, score[w]), (lh2h, lscore[w])):
            res[w] = {}
            for g in games.values():
                if len(g) == 2:
                    x, y = g
                    res[w][x] = 1.0 if sc_[x] > sc_[y] else 0.5 if sc_[x] == sc_[y] else 0.0
                    res[w][y] = 1.0 - res[w][x]
    rids = sorted(score[1])
    # made playoffs: league wins then league PF over the regular season
    lw = {r: sum(lh2h[w].get(r, 0) for w in weeks) for r in rids}
    lpf = {r: sum(lscore[w][r] for w in weeks) for r in rids}
    seeds = sorted(rids, key=lambda r: (-lw[r], -lpf[r]))[: m["playoff_teams"] or 6]
    out = []
    for W in CHECKPOINTS:
        if W > reg_end - 2:
            continue
        past, ros = range(1, W + 1), range(W + 1, reg_end + 1)
        roster = {e["roster_id"]: e["players"] or [] for e in weeks[W]}
        fut = [k for k in range(W + 1, W + 1 + LOOKAHEAD) if k <= 17]
        mean = lambda d, r, wk: float(np.mean([d[w][r] for w in wk if r in d[w]] or [np.nan]))
        for r in rids:
            out.append({
                "league": lid, "season": season, "cp": W, "rid": r,
                "record": mean(h2h, r, past), "pf_g": mean(score, r, past), "allplay": mean(ap, r, past),
                "strength": float(np.mean([best_lineup(roster[r], proj[season][k], slots) for k in fut])),
                "ros_allplay": mean(ap, r, ros), "ros_pf_g": mean(score, r, ros),
                "ros_h2h": sum(h2h[w].get(r, 0) for w in ros), "made_playoffs": float(r in seeds),
                "banked": sum(h2h[w].get(r, 0) for w in past), "n_ros": sum(r in h2h[w] for w in ros),
                "final_wins": sum(h2h[w].get(r, 0) for w in weeks),
            })
    return out


def z(x):
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x * 0


def rcorr(x, y):
    if x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def pct(x):
    return (rankdata(x) - 1) / max(len(x) - 1, 1)


def group(rows):
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["league"], r["cp"])].append(r)
    return {k: {c: np.array([r[c] for r in v]) for c in v[0] if c not in ("league",)} for k, v in g.items()}


def fit_blend(groups):
    """Leave-one-season-out weight on strength (vs PF/game), maximizing ROS PF/game Spearman."""
    grid = np.linspace(0, 1, 21)
    keys = list(groups)
    seas = np.array([groups[k]["season"][0] for k in keys])
    M = np.array([[rcorr(a * z(groups[k]["strength"]) + (1 - a) * z(groups[k]["pf_g"]), groups[k]["ros_pf_g"])
                   for a in grid] for k in keys])
    return {int(s): float(grid[np.nanargmax(np.nanmean(M[seas != s], 0))]) for s in np.unique(seas)}


def corr_table(groups, blend_a):
    """{(league, cp): {(pred, target): spearman}}; w_* = banked wins + ROS games * within-league pct(pred)."""
    res = {}
    for key, c in groups.items():
        a = blend_a[int(c["season"][0])]
        c["blend"] = a * z(c["strength"]) + (1 - a) * z(c["pf_g"])
        d = {(p, t): rcorr(c[p], c[t]) for p in PREDICTORS for t in TARGETS}
        for p in PREDICTORS:
            w = c["banked"] + c["n_ros"] * pct(c[p])
            for t in FAIR_TARGETS:
                d[("w_" + p, t)] = rcorr(w, c[t])
        res[key] = d
    return res


def league_matrix(ct, combos, keep):
    """Per-league mean correlation over the kept (league, cp) keys -> (leagues, combos) array."""
    by = collections.defaultdict(list)
    for k, d in ct.items():
        if keep(k):
            by[k[0]].append([d.get(c, np.nan) for c in combos])
    return np.array([np.nanmean(v, 0) for v in by.values()])


def boot(M, rng, reps=2000):
    idx = rng.integers(0, len(M), (reps, len(M)))
    b = np.nanmean(M[idx], 1)
    return np.nanmean(M, 0), np.percentile(b, 2.5, 0), np.percentile(b, 97.5, 0)


def summarize(ct, rng, meta):
    L = []
    combos = [(p, t) for p in PREDICTORS for t in TARGETS]
    M = league_matrix(ct, combos, lambda k: True)
    m, lo, hi = boot(M, rng)
    L += [f"Pooled over checkpoints (n={len(M)} leagues), Spearman [95% CI]:\n",
          "| predictor | " + " | ".join(TARGETS) + " |", "|---|" + "---|" * len(TARGETS)]
    for i, p in enumerate(PREDICTORS):
        j = i * len(TARGETS)
        L.append(f"| {p} | " + " | ".join(f"{m[j + q]:.2f} [{lo[j + q]:.2f},{hi[j + q]:.2f}]"
                                          for q in range(len(TARGETS))) + " |")
    L.append("\nPaired differences (league bootstrap), Spearman diff [95% CI]:\n")
    cols = [("pooled", lambda k: True)] + [(f"wk{c}", lambda k, c=c: k[1] == c) for c in CHECKPOINTS]
    L += ["| comparison | target | " + " | ".join(c for c, _ in cols) + " |", "|---|---|" + "---|" * len(cols)]
    for a, b in DIFFS:
        for t in ("ros_pf_g", "ros_allplay"):
            cells = []
            for _, keep in cols:
                M = league_matrix(ct, [(a, t), (b, t)], keep)
                mm, l2, h2 = boot(M[:, :1] - M[:, 1:], rng)
                cells.append(f"{mm[0]:+.3f} [{l2[0]:+.3f},{h2[0]:+.3f}]")
            L.append(f"| {a} - {b} | {t} | " + " | ".join(cells) + " |")
    fc = [("w_" + p, t) for p in PREDICTORS for t in FAIR_TARGETS]
    M = league_matrix(ct, fc, lambda k: True)
    m, lo, hi = boot(M, rng)
    L += ["\nFair playoff comparison: predictor = banked wins + ROS games x within-league pct(predictor), pooled:\n",
          "| predictor | " + " | ".join(FAIR_TARGETS) + " |", "|---|" + "---|" * len(FAIR_TARGETS)]
    for i, p in enumerate(PREDICTORS):
        j = i * len(FAIR_TARGETS)
        L.append(f"| {p} | " + " | ".join(f"{m[j + q]:.2f} [{lo[j + q]:.2f},{hi[j + q]:.2f}]"
                                          for q in range(len(FAIR_TARGETS))) + " |")
    L += ["\nBy season, ROS PF/g (pooled checkpoints):\n", "| season | n lg | pf_g | allplay | strength | blend | strength - pf_g |",
          "|---|---|---|---|---|---|---|"]
    for s in sorted({v for v in meta.values()}):
        keep = lambda k, s=s: meta[k[0]] == s
        M = league_matrix(ct, [(p, "ros_pf_g") for p in ("pf_g", "allplay", "strength", "blend")], keep)
        m = np.nanmean(M, 0)
        mm, l2, h2 = boot(M[:, 2:3] - M[:, :1], rng)
        L.append(f"| {s} | {len(M)} | " + " | ".join(f"{x:.2f}" for x in m) + f" | {mm[0]:+.3f} [{l2[0]:+.3f},{h2[0]:+.3f}] |")
    return L


def sanity(seasons):
    L = ["## Sanity: Sleeper weekly projection vs actual (PPR, players projected >= 1 pt; missing actual = 0)\n",
         "| pos | n | pearson r |", "|---|---|---|"]
    pairs = collections.defaultdict(lambda: ([], []))
    for s in seasons:
        for w in sc.WEEKS:
            pr, ac = weekly("projections", s, w), weekly("stats", s, w)
            for pid, v in pr.items():
                pos = next(iter(POS.get(pid) & {"QB", "RB", "WR", "TE", "K", "DEF"}), None)
                if v >= 1 and pos:
                    pairs[pos][0].append(v)
                    pairs[pos][1].append(ac.get(pid, 0.0))
    for pos in ("QB", "RB", "WR", "TE", "K", "DEF"):
        x, y = pairs[pos]
        L.append(f"| {pos} | {len(x)} | {pearsonr(x, y)[0]:.2f} |")
    return L


def main():
    rng = np.random.default_rng(0)
    seasons = list(sc.SEASONS)
    proj = {s: {w: weekly("projections", s, w) for w in sc.WEEKS} for s in seasons}
    act = {s: {w: weekly("stats", s, w) for w in sc.WEEKS} for s in seasons}
    objs = league_objects()
    kept = json.loads((sc.DATA / "kept_leagues.json").read_text())
    qual = {m["league_id"]: sc.analyze_league(m) for m in kept}
    base = [m for m in kept if qual[m["league_id"]]["complete"] and m["type"] in (0, 1, 2) and m["teams"] >= 6]
    sets = {"clean (0 strict-inactive)": [m for m in base if qual[m["league_id"]]["inactive_strict"] == 0],
            "loose (<=1 strict-inactive)": [m for m in base if qual[m["league_id"]]["inactive_strict"] <= 1]}
    L = ["# Sleeper power-ranking backtest (PILOT, small n)\n",
         "Spearman within league-checkpoint, averaged; 95% CI bootstraps leagues. "
         f"Strength = best legal lineup's Sleeper-projected PPR, avg next {LOOKAHEAD} wks, roster at checkpoint. "
         "Blend = a*z(strength)+(1-a)*z(PF/g), a fit leave-one-season-out on ROS PF/g.\n"]
    for name, ms in sets.items():
        rows = [r for m in ms for r in league_rows(m, objs[m["league_id"]], proj, act)]
        groups = group(rows)
        blend_a = fit_blend(groups)
        ct = corr_table(groups, blend_a)
        L.append(f"## {name}: {len(ms)} league-seasons, {len({(r['league'], r['rid']) for r in rows})} teams\n")
        L.append(f"Blend weight on strength by held-out season: {blend_a}\n")
        L.append("Leagues per checkpoint: " + ", ".join(
            f"wk{c}: {len({r['league'] for r in rows if r['cp'] == c})}" for c in CHECKPOINTS) + "\n")
        L += summarize(ct, rng, {m["league_id"]: m["season"] for m in ms})
        L.append("")
    L.extend(sanity(seasons))
    text = "\n".join(L) + "\n"
    (sc.DATA / "power_backtest.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
