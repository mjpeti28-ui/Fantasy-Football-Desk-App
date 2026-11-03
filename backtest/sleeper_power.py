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
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import sleeper_crawl as sc  # noqa: E402

CHECKPOINTS = (4, 6, 8, 10)
LOOKAHEAD = 3  # avg best-lineup projection over the next 3 weeks
PREDICTORS = ("record", "pf_g", "allplay", "strength", "blend")
TARGETS = ("ros_allplay", "ros_pf_g", "ros_h2h", "made_playoffs")
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
            })
    return out


def z(x):
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x * 0


def corr_table(rows, blend_a):
    """{(league, cp): {(pred, target): spearman}} with blend from per-season weight."""
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["league"], r["cp"])].append(r)
    res = {}
    for key, g in groups.items():
        a = blend_a[g[0]["season"]]
        col = {k: np.array([r[k] for r in g]) for k in PREDICTORS[:-1] + TARGETS}
        col["blend"] = a * z(col["strength"]) + (1 - a) * z(col["pf_g"])
        res[key] = {}
        for p in PREDICTORS:
            for t in TARGETS:
                if col[p].std() > 0 and col[t].std() > 0:
                    res[key][(p, t)] = spearmanr(col[p], col[t])[0]
    return res


def fit_blend(rows):
    """Leave-one-season-out weight on strength (vs PF/game), maximizing ROS PF/game Spearman."""
    seasons = sorted({r["season"] for r in rows})
    grid = np.linspace(0, 1, 21)
    out = {}
    for s in seasons:
        train = [r for r in rows if r["season"] != s]
        scores = []
        for a in grid:
            ct = corr_table(train, collections.defaultdict(lambda: a))
            scores.append(np.mean([v[("blend", "ros_pf_g")] for v in ct.values() if ("blend", "ros_pf_g") in v]))
        out[s] = float(grid[int(np.argmax(scores))])
    return out


def summarize(ct, cps, rng, reps=1000):
    keys = [k for k in ct if k[1] in cps]
    leagues = sorted({k[0] for k in keys})
    by_lg = collections.defaultdict(list)
    for k in keys:
        by_lg[k[0]].append(ct[k])
    out = {}
    for p in PREDICTORS:
        for t in TARGETS:
            per = {lg: [d[(p, t)] for d in by_lg[lg] if (p, t) in d] for lg in leagues}
            flat = lambda lgs: np.mean([v for lg in lgs for v in per[lg]])
            boots = [flat(rng.choice(leagues, len(leagues))) for _ in range(reps)]
            out[(p, t)] = (flat(leagues), *np.percentile(boots, [2.5, 97.5]))
    return out, len(leagues)


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
        blend_a = fit_blend(rows)
        ct = corr_table(rows, blend_a)
        L.append(f"## {name}: {len(ms)} league-seasons, {len({(r['league'], r['rid']) for r in rows})} teams\n")
        L.append(f"Blend weight on strength by held-out season: {blend_a}\n")
        L.append("Teams per checkpoint: " + ", ".join(
            f"wk{c}: {sum(r['cp'] == c for r in rows)} ({len({r['league'] for r in rows if r['cp'] == c})} lg)"
            for c in CHECKPOINTS) + "\n")
        for label, cps in [("all checkpoints pooled", CHECKPOINTS)] + [(f"week {c}", (c,)) for c in CHECKPOINTS]:
            summ, nl = summarize(ct, cps, rng)
            L.append(f"### {label} (n={nl} leagues)\n")
            L.append("| predictor | " + " | ".join(TARGETS) + " |")
            L.append("|---|" + "---|" * len(TARGETS))
            for p in PREDICTORS:
                L.append(f"| {p} | " + " | ".join(
                    f"{summ[(p, t)][0]:.2f} [{summ[(p, t)][1]:.2f},{summ[(p, t)][2]:.2f}]" for t in TARGETS) + " |")
            L.append("")
    L.extend(sanity(seasons))
    text = "\n".join(L) + "\n"
    (sc.DATA / "power_backtest.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
