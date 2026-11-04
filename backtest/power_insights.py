"""In-season power-ranking insights, each tested on the clean Sleeper set, then applied to Max's 2026 ESPN league.

1 luck ledger, 2 fragility, 3 trajectory, 4 size of real moves, 5 playoff structure (real brackets),
6 championship leverage for the 2026 league. No Desk changes; reads history/actuals read-only.

    .venv/bin/python backtest/power_insights.py      # writes ~/Developer/sleeper-data/insights.md
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).parent))
import power_my_league as pm  # noqa: E402
import sleeper_crawl as sc  # noqa: E402
import sleeper_experiments as se  # noqa: E402
import sleeper_power as sp  # noqa: E402

CPS = (6, 8, 10)
SIMS = 4000
SHRINK = 0.5  # weight on blend z when turning it into expected pts (rank_eval: full confidence is worse than coin flip)
RNG = np.random.default_rng(3)
zs, sp_corr, a_for = pm.zs, pm.sp_corr, pm.a_for


def ci(v, reps=2000, f="+.3f"):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    b = v[RNG.integers(0, len(v), (reps, len(v)))].mean(1)
    return f"{v.mean():{f}} [{np.percentile(b, 2.5):{f}},{np.percentile(b, 97.5):{f}}] (n={len(v)})"


def h2h(Sc, pairs, weeks, n):
    H, opp = np.zeros((len(weeks), n)), np.full((len(weeks), n), -1)
    for k, w in enumerate(weeks):
        for a, b in pairs.get(w, []):
            H[k, a] = 1.0 if Sc[w - 1, a] > Sc[w - 1, b] else 0.5 if Sc[w - 1, a] == Sc[w - 1, b] else 0.0
            H[k, b] = 1.0 - H[k, a]
            opp[k, a], opp[k, b] = b, a
    return H, opp


def allplay_w(rows):
    n = rows.shape[1]
    return np.array([(rankdata(r) - 1) / (n - 1) for r in rows]).sum(0)


def opp_over(Sc, opp, weeks, base):
    """Mean of (opponent's score vs you minus opponent's own average `base`)."""
    n = Sc.shape[1]
    return np.array([np.mean([Sc[w - 1, opp[k, i]] - base[opp[k, i]] for k, w in enumerate(weeks)
                              if opp[k, i] >= 0] or [0]) for i in range(n)])


def resid(y, x):
    return zs(y) - np.polyval(np.polyfit(x, zs(y), 1), x)


def fragility(roster, V, slots, avail):
    """(worst single-player loss share, that player, expected weekly loss pts from position absence rates)."""
    tot, chosen, left = se.lineup(roster, V, slots)
    if not chosen or tot <= 0:
        return 0.0, None, 0.0
    top = max(chosen, key=lambda c: V.get(c[0], 0))[0]
    worst = 1 - se.lineup([p for p in roster if p != top], V, slots)[0] / tot
    exp_loss = 0.0
    for p, slot in chosen:
        sub = max((V.get(q, 0) for q in left if sp.POS.get(q, set()) & sp.SLOT_ELIG[slot]), default=0.0)
        exp_loss += (1 - avail.get(se.pos_of(p), 0.9)) * (V.get(p, 0) - sub)
    return worst, top, exp_loss


def move_delta(after, recv, sent, V, slots):
    before = [p for p in after if p not in recv] + sent
    return se.lineup(after, V, slots)[0] - se.lineup(before, V, slots)[0]


# ======================= Sleeper tests =======================

def sleeper_tests(L):
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
    po_rows = []
    for m in clean:
        s, lid, lg = m["season"], m["league_id"], objs[m["league_id"]]
        reg_end = min((m["playoff_week_start"] or 15) - 1, 17)
        slots = [x for x in lg["roster_positions"] if x in sp.SLOT_ELIG]
        mu = {w: sc.get(f"{sc.BASE}/league/{lid}/matchups/{w}", offline=True) or [] for w in range(1, 18)}
        rids = sorted(e["roster_id"] for e in mu[1])
        ix = {r: i for i, r in enumerate(rids)}
        n = len(rids)
        Sc, Pj, Lc, lsc = (np.zeros((17, n)) for _ in range(4))
        pairs = {}
        for w in range(1, 18):
            g = collections.defaultdict(list)
            a = {p: v for p, (v, _) in S[s][w].items()}
            pr = {p: v for p, (v, _) in P[s][w].items()}
            for e in mu[w]:
                if e["roster_id"] not in ix:
                    continue
                i = ix[e["roster_id"]]
                stt = [p for p in e["starters"] or [] if p not in ("0", None)]
                Sc[w - 1, i] = sum(a.get(p, 0) for p in stt)
                lsc[w - 1, i] = e.get("points") or 0
                if w <= reg_end:
                    Pj[w - 1, i] = sum(pr.get(p, 0) for p in stt)
                    Lc[w - 1, i] = Sc[w - 1, i] - sum(a.get(p, 0) for p, _ in se.lineup(e["players"] or [], pr, slots)[1])
                if e.get("matchup_id") is not None:
                    g[e["matchup_id"]].append(i)
            pairs[w] = [tuple(v) for v in g.values() if len(v) == 2]
        st, fr = {}, {}
        for w in range(1, reg_end):
            roster = {e["roster_id"]: e["players"] or [] for e in mu[w]}
            V = {**info[s][w]["projmean"], **{p: v for p, (v, _) in P[s][w + 1].items()}}
            st[w] = np.array([se.lineup(roster[r], V, slots)[0] for r in rids])
            if w in CPS:
                fr[w] = [fragility(roster[r], V, slots, avail) for r in rids]
        # --- 4. size of real moves (trades and waiver weeks): projected lineup change at that week
        for w in range(1, reg_end - 1):
            txs = [t for t in sc.get(f"{sc.BASE}/league/{lid}/transactions/{w}", offline=True) or []
                   if t.get("status") == "complete"]
            after_r = {e["roster_id"]: e["players"] or [] for e in mu[w + 1]}
            V = info[s][w]["projmean"]
            wv = collections.defaultdict(lambda: ([], []))
            for t in txs:
                for r in t.get("roster_ids") or []:
                    recv = [p for p, x in (t.get("adds") or {}).items() if x == r]
                    sent = [p for p, x in (t.get("drops") or {}).items() if x == r]
                    if t["type"] == "trade" and r in after_r:
                        acc["trade_delta"].append(move_delta(after_r[r], recv, sent, V, slots))
                    elif t["type"] in ("waiver", "free_agent"):
                        wv[r][0].extend(recv)
                        wv[r][1].extend(sent)
            for r, (recv, sent) in wv.items():
                if r in after_r:
                    acc["waiver_delta"].append(move_delta(after_r[r], recv, sent, V, slots))
        # --- checkpoint tests
        for W in CPS:
            if W > reg_end - 3:
                continue
            past, ros = list(range(1, W + 1)), list(range(W + 1, reg_end + 1))
            Hp, op = h2h(Sc, pairs, past, n)
            Hr, orr = h2h(Sc, pairs, ros, n)
            sm = np.mean([st[W - j] for j in range(3)], 0)
            pf = Sc[:W].mean(0)
            blend = a_for(W) * zs(sm) + (1 - a_for(W)) * zs(pf)
            y = Sc[W:reg_end].mean(0)
            r = resid(y, blend)
            acc["sched_luck"].append(sp_corr(Hp.sum(0) - allplay_w(Sc[:W]), Hr.sum(0) - allplay_w(Sc[W:reg_end])))
            acc["over_perf"].append(sp_corr((Sc[:W] - Pj[:W]).mean(0), (Sc[W:reg_end] - Pj[W:reg_end]).mean(0)))
            acc["opp_over"].append(sp_corr(opp_over(Sc, op, past, pf), opp_over(Sc, orr, ros, Sc[W:reg_end].mean(0))))
            acc["lineup_cost"].append(sp_corr(Lc[:W].mean(0), Lc[W:reg_end].mean(0)))
            acc["over_vs_res"].append(sp_corr((Sc[:W] - Pj[:W]).mean(0), r))
            acc["worst_vs_res"].append(sp_corr([f[0] for f in fr[W]], r))
            acc["exploss_vs_res"].append(sp_corr([f[2] for f in fr[W]], r))
            acc["worst_vs_sd"].append(sp_corr([f[0] for f in fr[W]], Sc[W:reg_end].std(0)))
            if W - 5 >= 1:
                sm_prev = np.mean([st[W - 3 - j] for j in range(3)], 0)
                acc["traj_vs_res"].append(sp_corr(sm - sm_prev, r))
        # --- 5. playoffs with real brackets
        if reg_end < 11:
            continue
        br = sc.get(f"{sc.BASE}/league/{lid}/winners_bracket") or []
        final = next((b for b in br if b.get("p") == 1 and b.get("w")), None)
        if final is None:
            continue
        lw = h2h(lsc, pairs, list(range(1, reg_end + 1)), n)[0].sum(0)
        seed = np.empty(n, int)
        seed[np.lexsort((-lsc[:reg_end].sum(0), -lw))] = np.arange(1, n + 1)
        r1 = {t for b in br if b.get("r") == 1 for t in (b.get("t1"), b.get("t2"))}
        po_teams = {t for b in br for t in (b.get("t1"), b.get("t2")) if isinstance(t, int)}
        sm10 = np.mean([st[10 - j] for j in range(3)], 0)
        bl10 = a_for(10) * zs(sm10) + (1 - a_for(10)) * zs(Sc[:10].mean(0))
        pw_ = [w - 1 for w in range(reg_end + 1, 18)]
        for t in po_teams:
            if t in ix and pw_:
                i = ix[t]
                po_rows.append({"league": lid, "rid": t, "seed": seed[i], "bye": t not in r1, "blend10": bl10[i],
                                "po_ppg": Sc[pw_, i].mean(), "reg_ppg": Sc[:reg_end, i].mean(),
                                "title": float(final["w"] == t)})
    po = pd.DataFrame(po_rows)
    po["brank"] = po.groupby("league").blend10.rank(ascending=False, method="first")
    corr_po = [sp_corr(g.blend10, g.po_ppg) for _, g in po.groupby("league") if len(g) > 2]
    corr_reg = [sp_corr(g.reg_ppg, g.po_ppg) for _, g in po.groupby("league") if len(g) > 2]
    td, wd = np.array(acc["trade_delta"]), np.array(acc["waiver_delta"])
    good_trade = float(np.median(td[td > 0]))
    q = lambda x: f"median {np.median(x):+.1f}, p75 {np.percentile(x, 75):+.1f}, p90 {np.percentile(x, 90):+.1f}"
    L += [f"## Sleeper tests (clean set, {len(clean)} leagues; Spearman within league, league bootstrap)\n",
          "### 1. Luck ledger: does each component persist (season so far vs ROS)?\n",
          f"- schedule luck (wins minus all-play wins): {ci(acc['sched_luck'])}",
          f"- own over/under-performance vs projected starters: {ci(acc['over_perf'])}",
          f"- opponents' over-performance vs you: {ci(acc['opp_over'])}",
          f"- lineup cost vs projection-best lineup: {ci(acc['lineup_cost'])}",
          f"- over-performance vs ROS PF/g beyond blend: {ci(acc['over_vs_res'])}\n",
          "### 2. Fragility vs ROS PF/g beyond blend (negative = fragile teams underperform)\n",
          f"- worst single-player loss share: {ci(acc['worst_vs_res'])}; vs ROS weekly SD: {ci(acc['worst_vs_sd'])}",
          f"- expected weekly loss from position absence rates: {ci(acc['exploss_vs_res'])}",
          "- absence rates used, P(plays a later week): " +
          ", ".join(f"{k} {v:.2f}" for k, v in sorted((k, v) for k, v in avail.items() if k)) +
          " (byes counted as misses, so overstated by ~0.1)\n",
          "### 3. Trajectory: 3-week change in smoothed strength vs ROS PF/g beyond blend\n",
          f"- {ci(acc['traj_vs_res'])}\n",
          "### 4. Size of real moves (change in projected starting-lineup pts/wk at the time)\n",
          f"- trades, per side (n={len(td)}): {q(td)}; positive sides median {good_trade:+.1f} = 'typical good trade'",
          f"- a team's waiver/FA week (n={len(wd)}): {q(wd)}\n",
          f"### 5. Playoffs with real brackets ({po.league.nunique()} leagues, {len(po)} playoff teams)\n",
          f"- wk-10 blend vs playoff-week PPG (within league): {ci(corr_po)}; reg-season PPG vs playoff PPG: {ci(corr_reg)}",
          "- title rate by wk-10 blend rank among playoff teams: " + ", ".join(
              f"#{int(k)} {v.title.mean():.0%} (n={len(v)})" for k, v in po[po.brank <= 6].groupby("brank")),
          "- title rate, bye vs no bye, within blend-rank band:"]
    for lo, hi in ((1, 2), (3, 4), (5, 8)):
        b = po[(po.brank >= lo) & (po.brank <= hi)]
        L.append(f"  - blend rank {lo}-{hi}: " + ", ".join(
            f"{'bye' if k else 'no bye'} {v.title.mean():.0%} (n={len(v)})" for k, v in b.groupby("bye")))
    L.append("")
    return avail, good_trade


# ======================= 2026 league =======================

def league_2026(L, avail, good_trade):
    d = pm.load_season(2026)
    W = int(d["played"])
    last_roster = int(d["rw"].week.max())
    teams, n = d["teams"], len(d["teams"])
    ix = {t: i for i, t in enumerate(teams)}
    names = {t: d["names"][str(t)].strip() for t in teams}
    Sc, H, stj = d["Sc"], d["H"], d["st"]
    reg, n_po = stj["regular_season_weeks"], stj["playoff_teams"]
    mt = pd.read_csv(pm.HIST / "2026" / "matchups.csv")
    pairs = {w: [(ix[h], ix[a]) for h, a in zip(g.home_id, g.away_id)] for w, g in mt.groupby("week")}
    tbl = pm.team_table(d, W, 3, roster_week=last_roster).set_index("team").loc[teams]
    blend_z = a_for(W) * zs(tbl.strength) + (1 - a_for(W)) * zs(tbl.pf_g)
    slots = [pm.ESPN_SLOT[s] for s, c in stj["lineup_slot_counts"].items() if s in pm.ESPN_SLOT for _ in range(c)]
    pj, rw = d["pj"], d["rw"]
    proj = {w: {f"e{i}": v for i, v in zip(g.espn_id, g.espn_proj)} for w, g in pj.groupby("week")}
    pmean = pj[pj.week <= W].groupby("espn_id").espn_proj.mean()
    V = {**{f"e{i}": v for i, v in pmean.items()}, **proj.get(W + 1, {})}
    pname = {f"e{i}": nm for i, nm in zip(rw.espn_id, rw["name"])}
    roster = {(w, t): [f"e{i}" for i in g.espn_id] for (w, t), g in rw.groupby(["week", "team_id"])}
    pw = pd.read_csv(pm.HIST / "2026" / "players_weekly.csv")
    act = {w: {f"e{i}": v for i, v in zip(g.espn_id, g.points)} for w, g in pw[pw.kind == "actual"].groupby("week")}
    me = ix[pm.ME]
    order = np.argsort(-tbl.blend.values)

    # 1. ledger
    Hp, op = h2h(Sc, pairs, list(range(1, W + 1)), n)
    ap = allplay_w(Sc[:W])
    projected, lcost = np.zeros(n), np.zeros(n)
    for (w, t), g in rw[rw.week <= W].groupby(["week", "team_id"]):
        stt = [f"e{i}" for i, sl in zip(g.espn_id, g.lineup_slot) if sl not in ("BE", "IR")]
        projected[ix[t]] += sum(proj.get(w, {}).get(p, 0) for p in stt) / W
        best = se.lineup(roster[(w, t)], proj.get(w, {}), slots)[1]
        lcost[ix[t]] += (sum(act.get(w, {}).get(p, 0) for p in stt) -
                         sum(act.get(w, {}).get(p, 0) for p, _ in best)) / W
    pf = Sc[:W].mean(0)
    oo = opp_over(Sc, op, list(range(1, W + 1)), pf)
    L += [f"## 2026 league through week {W} (history/actuals/2026; roster week {last_roster})\n",
          "### 1. Luck ledger\n",
          "wins = all-play expected wins + schedule luck. PF/g = projected (your starters' ESPN projections) + "
          "over/under. opp over = opponents vs their own average when facing you. lineup cost = pts/wk vs the "
          "projection-best lineup from your own roster (scored on actuals).\n",
          "| team | wins | = all-play W | + sched luck | PF/g | = projected | + over/under | opp over vs you | lineup cost |",
          "|---|---|---|---|---|---|---|---|---|"]
    for i in order:
        t = teams[i]
        L.append(f"| {names[t]}{' *' if i == me else ''} | {Hp.sum(0)[i]:g} | {ap[i]:.1f} | {Hp.sum(0)[i] - ap[i]:+.1f} | "
                 f"{pf[i]:.1f} | {projected[i]:.1f} | {pf[i] - projected[i]:+.1f} | {oo[i]:+.1f} | {lcost[i]:+.1f} |")

    # 2-3. fragility, trajectory
    L += ["\n### 2-3. Fragility and trajectory\n",
          "worst loss = lineup share lost if the top projected starter is out; exp. loss = pts/wk expected from "
          "position absence rates; trajectory = smoothed strength now minus 3 weeks earlier, with the biggest "
          "roster changes behind it (+ added, - gone; current projection).\n",
          "| team | strength | worst loss (who) | exp. loss pts/wk | trajectory | main moves (last 3 wks) |",
          "|---|---|---|---|---|---|"]
    sm_prev = np.mean([d["strength"](W - 3 - j) for j in range(3)], 0)
    for i in order:
        t = teams[i]
        worst, top, el = fragility(roster[(last_roster, t)], V, slots, avail)
        old = set(roster.get((last_roster - 3, t), []))
        new = set(roster[(last_roster, t)])
        mv = sorted([("+", p) for p in new - old] + [("-", p) for p in old - new], key=lambda x: -V.get(x[1], 0))[:3]
        L.append(f"| {names[t]}{' *' if i == me else ''} | {tbl.strength[t]:.1f} | {worst:.0%} ({pname.get(top, '?')}) | "
                 f"{el:.1f} | {tbl.strength[t] - sm_prev[i]:+.1f} | "
                 f"{', '.join(f'{s}{pname.get(p, p)} ({V.get(p, 0):.1f})' for s, p in mv) or '-'} |")

    # 4. gaps in trades
    L += [f"\n### 4. Stinkers' gap to each team: blend pts/wk and 'typical good trades' ({good_trade:+.1f} pts/wk each)\n",
          "| team | blend | gap (pts/wk) | good trades to close |", "|---|---|---|---|"]
    for i in order:
        if i == me:
            continue
        t = teams[i]
        gap = tbl.blend[t] - tbl.blend[pm.ME]
        L.append(f"| {names[t]} | {tbl.blend[t]:.1f} | {gap:+.1f} | {abs(gap) / good_trade:.1f} "
                 f"({'they lead' if gap > 0 else 'you lead'}) |")

    # 6. championship leverage
    sd_week = Sc[:W].std(0).mean()
    base_mu = Sc[:W].mean() + SHRINK * blend_z * pf.std()
    rem = list(range(W + 1, reg + 1))

    def season(mu_t, force=None):
        wins = np.tile(H[:W].sum(0), (SIMS, 1)).astype(float)
        pts = np.tile(Sc[:W].sum(0), (SIMS, 1)).astype(float)
        for w in rem:
            s_ = mu_t + RNG.normal(0, sd_week, (SIMS, n))
            if force and force[0] == w:
                _, team, win = force
                o = next(b if a == team else a for a, b in pairs[w] if team in (a, b))
                hi, lo = (team, o) if win else (o, team)
                s_[:, hi] = np.maximum(s_[:, hi], s_[:, lo] + 0.01)
            pts += s_
            for a, b in pairs[w]:
                wins[:, a] += s_[:, a] > s_[:, b]
                wins[:, b] += s_[:, b] > s_[:, a]
        seeds = np.argsort(-(wins * 1e6 + pts), axis=1)[:, :n_po]  # record then PF (assumed)
        alive = seeds
        while alive.shape[1] > 1:  # fixed bracket: best seed vs worst remaining slot each round
            k = alive.shape[1]
            a_, b_ = alive[:, : k // 2], alive[:, ::-1][:, : k // 2]
            sa = mu_t[a_] + RNG.normal(0, sd_week, a_.shape)
            sb = mu_t[b_] + RNG.normal(0, sd_week, b_.shape)
            alive = np.where(sa > sb, a_, b_)
        title = (alive[:, 0][:, None] == np.arange(n)).mean(0)
        made = np.stack([(seeds == i).any(1) for i in range(n)], 1).mean(0)
        return title, made
    title, made = season(base_mu)
    bump = base_mu.copy()
    bump[me] += 1
    title_b = season(bump)[0][me]
    lev = np.array([season(base_mu, (w, me, True))[0][me] - season(base_mu, (w, me, False))[0][me] for w in rem])
    L += [f"\n### 6. Championship leverage (sim weeks {W + 1}-{reg} + {n_po}-team bracket; blend means shrunk "
          f"{SHRINK}; weekly sd {sd_week:.1f}; {SIMS} sims)\n",
          "| team | blend rank | playoff odds | title odds |", "|---|---|---|---|"]
    for rk, i in enumerate(order, 1):
        L.append(f"| {names[teams[i]]}{' *' if i == me else ''} | {rk} | {made[i]:.0%} | {title[i]:.1%} |")
    L += [f"\nStinkers title odds {title[me]:.1%}; +1 pt/wk rest of season -> {title_b:.1%} "
          f"({(title_b - title[me]) * 100:+.1f} pts of title odds per pt/wk).\n",
          "Week leverage for Stinkers (title odds if win minus if lose; 1.0 = average remaining week):\n",
          "| week | opponent | title odds swing | leverage index |", "|---|---|---|---|"]
    for w, lv in zip(rem, lev):
        o = next(b if a == me else a for a, b in pairs[w] if me in (a, b))
        L.append(f"| {w} | {names[teams[o]]} | {lv * 100:+.1f} pts | {lv / lev.mean():.2f}"
                 f"{' (high)' if lv / lev.mean() > 1.3 and abs(lv) > 0.015 else ''} |")
    L.append(f"\nUnsure: seeding assumed record then PF (settings.json says {stj['seeding']}); bracket assumed "
             "fixed 1v8/2v7/3v6/4v5 with best-remaining vs worst-remaining, no reseeding; MC noise ~±0.7 pts on swings.\n")


def main():
    L = ["# In-season power-ranking insights\n"]
    avail, good_trade = sleeper_tests(L)
    league_2026(L, avail, good_trade)
    text = "\n".join(L) + "\n"
    (sc.DATA / "insights.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
