"""Experiments on the Sleeper crawl: leak-free roster strength variants and manager-skill measures.

Part 0 leakage check, R1-R5 strength variants, S1-S5 skill reliability, and the key test
(does skill predict ROS PF/g beyond strength). Writes ~/Developer/sleeper-data/experiments.md.

    .venv/bin/python backtest/sleeper_experiments.py
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import sleeper_crawl as sc  # noqa: E402
import sleeper_power as sp  # noqa: E402

CPS = sp.CHECKPOINTS
SKILL = ("QB", "RB", "WR", "TE", "K", "DEF")
K_SHRINK = 3.0  # games of prior weight when shrinking ppg
VARIANTS = ("orig3", "orig3_fill", "next1", "next1_fill", "projmean", "projmean_w1", "blend_c", "ppg_only")
RNG = np.random.default_rng(0)


def load(kind, s, w):
    rows = sc.get(f"https://api.sleeper.app/{kind}/nfl/{s}/{w}?season_type=regular") or []
    out = {}
    for r in rows:
        st, pl = r.get("stats") or {}, r.get("player") or {}
        sp.POS.setdefault(r["player_id"], set()).update(pl.get("fantasy_positions") or [pl.get("position")])
        if st.get("pts_ppr") is not None:
            out[r["player_id"]] = (st["pts_ppr"], st.get("gp") or 0)
    return out


def pos_of(p):
    return next((x for x in SKILL if x in sp.POS.get(p, ())), None)


def lineup(players, val, slots):
    left = sorted(players, key=lambda p: -val.get(p, 0.0))
    total, chosen = 0.0, []
    for slot in sorted(slots, key=lambda s: len(sp.SLOT_ELIG[s])):
        for p in left:
            if sp.POS.get(p, set()) & sp.SLOT_ELIG[slot]:
                total += val.get(p, 0.0)
                chosen.append((p, slot))
                left.remove(p)
                break
    return total, chosen, left


def player_info(P, S):
    """Per season, per week w: value dicts using only info known after week w."""
    info = {}
    for s in P:
        info[s] = {}
        psum, pn, asum, an = (collections.Counter() for _ in range(4))
        for w in range(1, 18):
            for p, (v, _) in P[s][w].items():
                psum[p] += v
                pn[p] += 1
            for p, (v, gp) in S[s][w].items():
                if gp > 0:
                    asum[p] += v
                    an[p] += 1
            pm = {p: psum[p] / pn[p] for p in pn}
            ppg = {p: asum[p] / an[p] for p in an}
            prior = collections.defaultdict(list)
            for p, v in ppg.items():
                prior[pos_of(p)].append(v)
            prior = {k: float(np.mean(v)) for k, v in prior.items()}
            blend = {p: (an[p] * ppg.get(p, 0) + K_SHRINK * pm[p]) / (an[p] + K_SHRINK) for p in pm}
            free = {p: (an[p] * ppg[p] + K_SHRINK * prior.get(pos_of(p), 0)) / (an[p] + K_SHRINK) for p in ppg}
            info[s][w] = {"projmean": pm, "blend_c": blend, "ppg_only": free}
    return info


def availability(S):
    """P(plays a later week | fantasy-relevant early), per position; byes excluded."""
    hit, tot = collections.Counter(), collections.Counter()
    for s in S:
        early = collections.Counter(p for w in range(1, 9) for p, (v, gp) in S[s][w].items() if gp > 0 and v >= 8)
        for p, n in early.items():
            if n < 4:
                continue
            for w in range(9, 18):
                row = S[s][w].get(p)
                tot[pos_of(p)] += 1  # missing row = did not play (bye counted as a miss; ~1/9 bias)
                hit[pos_of(p)] += bool(row and row[1] > 0)
    return {k: hit[k] / tot[k] for k in tot if tot[k]}


def draft_picks(m):
    drafts = sc.get(f"{sc.BASE}/league/{m['league_id']}/drafts") or []
    d = next((d for d in drafts if d.get("status") == "complete" and str(d.get("season")) == str(m["season"])), None)
    return (sc.get(f"{sc.BASE}/draft/{d['draft_id']}/picks") or []) if d else []


def league(m, lg, P, S, info, avail, repl_cache):
    s, lid = m["season"], m["league_id"]
    reg_end = min((m["playoff_week_start"] or 15) - 1, 17)
    slots = [x for x in lg["roster_positions"] if x in sp.SLOT_ELIG]
    owners = {int(k): (v or [None])[0] for k, v in (m.get("owners") or {}).items()}
    act = {w: {p: v for p, (v, _) in S[s][w].items()} for w in range(1, 18)}
    prj = {w: {p: v for p, (v, _) in P[s][w].items()} for w in range(1, 18)}
    mu = {w: sc.get(f"{sc.BASE}/league/{lid}/matchups/{w}", offline=True) or [] for w in range(1, reg_end + 1)}
    tx = [t for w in range(1, 18) for t in sc.get(f"{sc.BASE}/league/{lid}/transactions/{w}", offline=True) or []
          if t.get("status") == "complete"]
    score = {w: {e["roster_id"]: sum(act[w].get(p, 0) for p in e["starters"] or [] if p not in ("0", None))
                 for e in mu[w]} for w in mu}
    rids = sorted(score[1])
    pickups = collections.defaultdict(dict)  # rid -> {pid: week added}
    adds_wk = collections.Counter()
    trades = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for t in tx:
        wk = t.get("leg") or 0
        for p, r in (t.get("adds") or {}).items():
            if t["type"] in ("waiver", "free_agent"):
                pickups[r].setdefault(p, wk)
                adds_wk[(r, wk)] += 1
    # weekly skill
    weekly = []
    for w, entries in mu.items():
        rostered = {p for e in entries for p in e["players"] or []}
        repl = {}
        for e in entries:
            r = e["roster_id"]
            starters = [p for p in e["starters"] or [] if p not in ("0", None)]
            roster = e["players"] or []
            chosen = lineup(roster, prj[w], slots)[1]
            dec = score[w][r] - sum(act[w].get(p, 0) for p, _ in chosen)
            hind = score[w][r] - lineup(roster, act[w], slots)[0]
            wv = 0.0
            for p in starters:
                if pickups[r].get(p, 99) <= w:
                    ps = pos_of(p)
                    if (ps, w) not in repl:
                        key = (s, w, ps)
                        if key not in repl_cache:
                            repl_cache[key] = sorted((q for q in prj[w] if pos_of(q) == ps), key=lambda q: -prj[w][q])
                        q = next((q for q in repl_cache[key] if q not in rostered), None)
                        repl[(ps, w)] = act[w].get(q, 0.0) if q else 0.0
                    wv += act[w].get(p, 0) - repl[(ps, w)]
            weekly.append({"league": lid, "rid": r, "w": w, "S1_dec": dec, "S1_hind": hind,
                           "S2_waiver": wv, "S5_activity": adds_wk[(r, w)]})
    weekly = pd.DataFrame(weekly)
    # trades
    for t in tx:
        wk = t.get("leg") or 0
        if t["type"] != "trade" or wk >= reg_end or wk < 1:
            continue
        ros = range(wk + 1, reg_end + 1)
        for r in t.get("roster_ids") or []:
            recv = [p for p, x in (t.get("adds") or {}).items() if x == r]
            sent = [p for p, x in (t.get("drops") or {}).items() if x == r]
            gain = sum(act[w].get(p, 0) for w in ros for p in recv) - sum(act[w].get(p, 0) for w in ros for p in sent)
            after = next((e["players"] or [] for e in mu.get(wk + 1, []) if e["roster_id"] == r), None)
            if after is None:
                continue
            before = [p for p in after if p not in recv] + sent
            V = info[s][wk]["projmean"]
            trades[r][0] += gain / len(ros)
            trades[r][1] += lineup(after, V, slots)[0] - lineup(before, V, slots)[0]
            trades[r][2] += 1
    # draft (redraft only)
    picks = []
    if m["type"] == 0:
        for pk in draft_picks(m):
            if pk.get("player_id") and pk.get("roster_id") in owners:
                picks.append({"league": lid, "season": s, "teams": m["teams"], "rid": pk["roster_id"],
                              "pick_no": pk["pick_no"], "round": pk.get("round"),
                              "value": sum(act[w].get(pk["player_id"], 0) for w in range(1, reg_end + 1))})
    # checkpoint rows
    rows = []
    for W in CPS:
        if W > reg_end - 2:
            continue
        roster = {e["roster_id"]: e["players"] or [] for e in mu[W]}
        ros = range(W + 1, reg_end + 1)
        I = info[s][W]
        vals = {"projmean": I["projmean"], "blend_c": I["blend_c"], "ppg_only": I["ppg_only"],
                "projmean_w1": info[s][W + 1]["projmean"] if W + 1 <= 17 else I["projmean"]}
        # projmean_w1 uses proj means through w+1 (w+1 projection is published pre-kickoff); stats part unused
        for r in rids:
            R = roster[r]
            row = {"league": lid, "season": s, "cp": W, "rid": r, "user": owners.get(r),
                   "pf_g": np.mean([score[w][r] for w in range(1, W + 1)]),
                   "ros_pf_g": np.mean([score[w][r] for w in ros]),
                   "orig3": np.mean([lineup(R, prj[k], slots)[0] for k in range(W + 1, min(W + 3, 17) + 1)]),
                   "next1": lineup(R, prj[W + 1], slots)[0]}
            # *_fill: players with no projection row that week (bye/absent) get their mean projection so far
            fill = lambda k: {**I["projmean"], **prj[k]}
            row["next1_fill"] = lineup(R, fill(W + 1), slots)[0]
            row["orig3_fill"] = np.mean([lineup(R, fill(k), slots)[0] for k in range(W + 1, min(W + 3, 17) + 1)])
            for k, V in vals.items():
                row[k] = lineup(R, V, slots)[0]
            # R2/R3 on the projmean_w1 values (leak-free baseline candidate; recomputed below if another wins)
            V = vals["projmean_w1"]
            tot, chosen, left = lineup(R, V, slots)
            e = 0.0
            for p, slot in chosen:
                sub = max((V.get(q, 0) for q in left if sp.POS.get(q, set()) & sp.SLOT_ELIG[slot]), default=0.0)
                a = avail.get(pos_of(p), 0.9)
                e += a * V.get(p, 0) + (1 - a) * sub
            row["expected"] = e
            row["bench"] = sum(sorted((V.get(q, 0) for q in left if sp.POS.get(q, set()) & {"RB", "WR", "TE"}),
                                      reverse=True)[:3])
            sv = sorted((V.get(p, 0) for p, _ in chosen), reverse=True)
            row["top2_share"] = sum(sv[:2]) / tot if tot else 0
            for ps in ("QB", "RB", "WR", "TE"):
                row["pos_" + ps] = sum(V.get(p, 0) for p, _ in chosen if pos_of(p) == ps)
            wk = weekly[(weekly.rid == r) & (weekly.w <= W)]
            for c in ("S1_dec", "S2_waiver", "S5_activity"):
                row["fh_" + c] = wk[c].mean()
            rows.append(row)
    team = {r: {"league": lid, "season": s, "rid": r, "user": owners.get(r), "teams": m["teams"],
                "S3_trade": trades[r][0], "S3_trade_proj": trades[r][1], "n_trades": trades[r][2]} for r in rids}
    return rows, weekly, team, picks


# ---------- evaluation helpers ----------

def zg(df, cols, by="gid"):
    g = df.groupby(by)
    for c in cols:
        s = g[c].transform("std")
        df["z_" + c] = ((df[c] - g[c].transform("mean")) / s.where(s > 0)).fillna(0.0)


def wspear(df, pred, target="ros_pf_g"):
    """Within league-cp Spearman, averaged to one value per league."""
    g = df.groupby("gid")
    rp, rt = g[pred].rank(), g[target].rank()
    dp = rp - rp.groupby(df.gid).transform("mean")
    dt = rt - rt.groupby(df.gid).transform("mean")
    num = (dp * dt).groupby(df.gid).sum()
    den = np.sqrt((dp ** 2).groupby(df.gid).sum() * (dt ** 2).groupby(df.gid).sum())
    c = (num / den.where(den > 0)).rename("c")
    lg = df.groupby("gid").league.first()
    return c.groupby(lg).mean()


def ci(vals, reps=2000):
    v = np.asarray(vals, float)
    v = v[~np.isnan(v)]
    b = v[RNG.integers(0, len(v), (reps, len(v)))].mean(1)
    return v.mean(), *np.percentile(b, [2.5, 97.5])


def f3(t, sign=False):
    m, lo, hi = t
    return f"{m:+.3f} [{lo:+.3f},{hi:+.3f}]" if sign else f"{m:.3f} [{lo:.3f},{hi:.3f}]"


def loso(df, feats, y="z_ros_pf_g"):
    pred = pd.Series(0.0, index=df.index)
    for s in df.season.unique():
        tr = df.season != s
        b = np.linalg.lstsq(df.loc[tr, feats].values, df.loc[tr, y].values, rcond=None)[0]
        pred[~tr] = df.loc[~tr, feats].values @ b
    return pred


def incr(df, base, extra):
    """Paired league-bootstrap Spearman gain of LOSO OLS (base+extra) over (base); plus pooled coef CI."""
    zg(df, base + extra + ["ros_pf_g"])
    zb, ze = ["z_" + c for c in base], ["z_" + c for c in base + extra]
    df["_pb"], df["_pe"] = loso(df, zb), loso(df, ze)
    d = (wspear(df, "_pe") - wspear(df, "_pb")).values
    leagues = df.league.unique()
    idx = {lg: np.where(df.league.values == lg)[0] for lg in leagues}
    X, Y = df[ze].values, df["z_ros_pf_g"].values
    coef = np.linalg.lstsq(X, Y, rcond=None)[0][-1]
    bs = []
    for _ in range(300):
        ii = np.concatenate([idx[lg] for lg in RNG.choice(leagues, len(leagues))])
        bs.append(np.linalg.lstsq(X[ii], Y[ii], rcond=None)[0][-1])
    return ci(d), (coef, *np.percentile(bs, [2.5, 97.5]))


def pts_per_sd(df, base, extra):
    """ROS PF/g points per 1 SD of `extra`, within-league demeaned OLS controlling for base."""
    cols = base + [extra, "ros_pf_g"]
    g = df.groupby("gid")[cols]
    dm = df[cols] - g.transform("mean")
    dm[extra] = dm[extra] / df[extra].std()
    return np.linalg.lstsq(dm[base + [extra]].values, dm["ros_pf_g"].values, rcond=None)[0][-1]


def reliab(a, b, groups):
    """Pearson r of paired arrays with bootstrap over `groups` (league or user)."""
    a, b, groups = np.asarray(a, float), np.asarray(b, float), np.asarray(groups)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b, groups = a[ok], b[ok], groups[ok]
    if len(a) < 10:
        return None
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    bs = []
    for _ in range(500):
        ii = np.concatenate([idx[g] for g in RNG.choice(ug, len(ug))])
        bs.append(np.corrcoef(a[ii], b[ii])[0, 1])
    return np.corrcoef(a, b)[0, 1], *np.nanpercentile(bs, [2.5, 97.5]), len(ug)


def main():
    seasons = list(sc.SEASONS)
    P = {s: {w: load("projections", s, w) for w in range(1, 18)} for s in seasons}
    S = {s: {w: load("stats", s, w) for w in range(1, 18)} for s in seasons}
    info, avail = player_info(P, S), availability(S)
    objs = sp.league_objects()
    kept = json.loads((sc.DATA / "kept_leagues.json").read_text())
    qual = {m["league_id"]: sc.analyze_league(m) for m in kept}
    loose = [m for m in kept if qual[m["league_id"]]["complete"] and m["type"] in (0, 1, 2)
             and 8 <= m["teams"] <= 14 and qual[m["league_id"]]["inactive_strict"] <= 1]
    clean_ids = {m["league_id"] for m in loose if qual[m["league_id"]]["inactive_strict"] == 0}
    rows, weekly, teams, picks, cache = [], [], [], [], {}
    for i, m in enumerate(loose):
        r, wk, tm, pk = league(m, objs[m["league_id"]], P, S, info, avail, cache)
        rows += r
        weekly.append(wk)
        teams += list(tm.values())
        picks += pk
        if i % 200 == 0:
            print(f"{i}/{len(loose)} leagues", flush=True)
    df = pd.DataFrame(rows)
    df["gid"] = df.league + "_" + df.cp.astype(str)
    df["clean"] = df.league.isin(clean_ids)
    weekly = pd.concat(weekly)
    teams = pd.DataFrame(teams)
    L = ["# Sleeper experiments: roster strength + manager skill\n",
         f"Loose set {len(loose)} league-seasons ({len(clean_ids)} clean). Target ROS PF/g (PPR rescored). "
         "Spearman within league-checkpoint, averaged per league; 95% CI = league bootstrap unless noted. "
         f"Availability by pos (R2): {{{', '.join(f'{k}: {v:.2f}' for k, v in sorted((k, v) for k, v in avail.items() if k))}}}\n"]

    # ---- Part 0 + R1 + R4
    L += ["## Part 0 / R1 / R4: strength variants (clean; loose diff vs PF/g in last col)\n",
          "| variant | Spearman | minus PF/g | wk4 minus PF/g | wk10 minus PF/g | loose minus PF/g |", "|---|---|---|---|---|---|"]
    C, Lo = df[df.clean].copy(), df.copy()
    base_pf = {k: wspear(d, "pf_g") for k, d in (("c", C), ("l", Lo))}
    best, best_v = None, -1
    for v in ("pf_g",) + VARIANTS:
        sc_ = wspear(C, v)
        cells = [f3(ci(sc_))]
        for sub in (C, C[C.cp == 4], C[C.cp == 10]):
            cells.append(f3(ci((wspear(sub, v) - wspear(sub, "pf_g")).values), True))
        cells.append(f3(ci((wspear(Lo, v) - base_pf["l"]).values), True))
        L.append(f"| {v} | " + " | ".join(cells) + " |")
        if v not in ("pf_g", "orig3", "orig3_fill") and sc_.mean() > best_v:
            best, best_v = v, sc_.mean()
    L.append(f"\norig3 = next-3-week projections (possible leak); *_fill = bye/absent players filled with mean projection so far. Leak-free baseline chosen: **{best}**. "
             "R2/R3 below use projmean_w1 values regardless.\n")

    # ---- R2 depth
    L += ["## R2 depth (clean)\n", "| measure | value |", "|---|---|"]
    L.append(f"| Spearman expected-lineup | {f3(ci(wspear(C, 'expected')))} |")
    L.append(f"| expected minus starters-only (projmean_w1) | "
             f"{f3(ci((wspear(C, 'expected') - wspear(C, 'projmean_w1')).values), True)} |")
    L.append(f"| bench (top-3 RB/WR/TE) alone | {f3(ci(wspear(C, 'bench')))} |")
    g, c = incr(C.copy(), [best], ["bench"])
    L.append(f"| bench incremental over {best}: Spearman gain / z-coef | {f3(g, True)} / {f3(c, True)} |\n")

    # ---- R3 shape
    C2 = C.copy()
    zg(C2, ["pos_QB", "pos_RB", "pos_WR", "pos_TE"])
    C2["balance"] = -C2[["z_pos_QB", "z_pos_RB", "z_pos_WR", "z_pos_TE"]].std(axis=1)
    L += ["## R3 shape (clean): LOSO-OLS Spearman gain and pooled z-coef\n",
          "| feature | base | Spearman gain | coef |", "|---|---|---|---|"]
    for feat in ("top2_share", "balance"):
        for base in ([best], [best, "pf_g"]):
            g, c = incr(C2.copy(), base, [feat])
            L.append(f"| {feat} | {'+'.join(base)} | {f3(g, True)} | {f3(c, True)} |")
    L.append("")

    # ---- R5 blend weight by checkpoint
    L += [f"## R5 blend weight on {best} (vs PF/g) by checkpoint (clean)\n", "| cp | a | Spearman at a |", "|---|---|---|"]
    grid = np.linspace(0, 1, 21)
    for W in CPS:
        sub = C[C.cp == W].copy()
        zg(sub, [best, "pf_g"])
        res = []
        for a in grid:
            sub["_b"] = a * sub["z_" + best] + (1 - a) * sub["z_pf_g"]
            res.append(wspear(sub, "_b").mean())
        L.append(f"| {W} | {grid[int(np.argmax(res))]:.2f} | {max(res):.3f} |")
    L.append("")

    # ---- skill: user-season tables
    weekly = weekly.merge(teams[["league", "rid", "user", "season"]], on=["league", "rid"])
    wk_metrics = ["S1_dec", "S1_hind", "S2_waiver", "S5_activity"]
    team_season = weekly.groupby(["league", "rid", "user", "season"])[wk_metrics].mean().reset_index()
    team_season = team_season.merge(teams[["league", "rid", "S3_trade", "S3_trade_proj", "n_trades"]], on=["league", "rid"])
    pk = pd.DataFrame(picks)
    if len(pk):
        exp = pk.groupby(["teams", "pick_no"]).value.transform("mean")
        cnt = pk.groupby(["teams", "pick_no"]).value.transform("size")
        pk["resid"] = pk.value - np.where(cnt >= 30, exp, pk.groupby("pick_no").value.transform("mean"))
        s4 = pk[pk["round"] <= 10].groupby(["league", "rid"]).resid.mean().rename("S4_draft").reset_index()
        team_season = team_season.merge(s4, on=["league", "rid"], how="left")
    else:
        team_season["S4_draft"] = np.nan
    team_season = team_season[team_season.user.notna()]
    skill_cols = wk_metrics + ["S3_trade", "S3_trade_proj", "S4_draft"]
    us = team_season.groupby(["user", "season"])[skill_cols].mean().reset_index()

    L += ["## S1-S5 reliability: Pearson r [95% CI] (n groups)\n",
          "Split-half = odd vs even weeks within team-season (boot by league); YoY = same user, season t vs t+1 "
          "(boot by user); cross-league = same user, two leagues, same season (boot by user).\n",
          "| metric | mean/team-wk | split-half | YoY | cross-league same season |", "|---|---|---|---|---|"]
    fmtr = lambda t: "n/a" if t is None else f"{t[0]:.2f} [{t[1]:.2f},{t[2]:.2f}] (n={t[3]})"
    wkm = weekly.assign(odd=weekly.w % 2).groupby(["league", "rid", "odd"])[wk_metrics].mean().unstack("odd")
    nxt = us.assign(season=us.season - 1)
    yoy = us.merge(nxt, on=["user", "season"], suffixes=("", "_n"))
    ts2 = team_season.sort_values("league").groupby(["user", "season"]).head(2)
    pairs = ts2.groupby(["user", "season"]).filter(lambda x: len(x) == 2)
    first, second = pairs.iloc[::2], pairs.iloc[1::2]
    for c in skill_cols:
        sh = reliab(wkm[(c, 0)], wkm[(c, 1)], wkm.index.get_level_values(0)) if c in wk_metrics else None
        yy = reliab(yoy[c], yoy[c + "_n"], yoy.user)
        cl = reliab(first[c].values, second[c].values, first.user.values)
        L.append(f"| {c} | {team_season[c].mean():.2f} | {fmtr(sh)} | {fmtr(yy)} | {fmtr(cl)} |")
    L.append("\nS1_dec = chosen starters minus Sleeper-projection-best lineup (actual pts). S1_hind = vs hindsight "
             "optimum. S2 = started in-season pickups' pts above best-projected free agent at that pos. "
             "S3 = trade ROS ppg gained minus sent (all weeks, not usage-weighted); S3_trade_proj = projected "
             "lineup change at trade time. S4 = redraft picks rd 1-10, season pts minus pick-slot mean. "
             "S5 = adds per week.\n")

    # ---- residuals for prior "beat projection"
    C3 = df.copy()
    zg(C3, [best, "pf_g", "ros_pf_g"])
    b = np.linalg.lstsq(C3[["z_" + best, "z_pf_g"]].values, C3.z_ros_pf_g.values, rcond=None)[0]
    C3["resid"] = C3.z_ros_pf_g - C3[["z_" + best, "z_pf_g"]].values @ b
    res_us = C3.groupby(["user", "season"]).resid.mean().rename("prior_resid").reset_index()
    prior = us.merge(res_us, on=["user", "season"], how="outer").assign(season=lambda x: x.season + 1)
    prior.columns = ["user", "season"] + ["prior_" + c if not c.startswith("prior") else c
                                          for c in prior.columns[2:]]

    # ---- key test
    L += [f"## Key test: does skill predict ROS PF/g beyond leak-free strength ({best})?\n",
          "Prior = same user's previous season (any league). First-half = same team, weeks <= checkpoint. "
          "Gain = LOSO-OLS within-league Spearman gain (paired league bootstrap); coef in within-league z units; "
          "pts = ROS PF/g per 1 SD of skill, within-league demeaned.\n",
          "| skill | set | n teams | base | Spearman gain | coef | pts/wk per SD |", "|---|---|---|---|---|---|---|"]
    tests = [("prior_" + c, "prior") for c in skill_cols] + [("prior_resid", "prior")] + \
            [("fh_" + c, "first-half") for c in ("S1_dec", "S2_waiver", "S5_activity")]
    full = df.merge(prior, on=["user", "season"], how="left")
    for feat, kind in tests:
        for setname, d in (("clean", full[full.clean]), ("loose", full)):
            d = d[d[feat].notna()].copy()
            d = d[d.groupby("gid").gid.transform("size") >= 6]
            if d.league.nunique() < 30:
                L.append(f"| {feat} | {setname} | {len(d)} | - | too few | | |")
                continue
            for base in ([best], [best, "pf_g"]):
                if setname == "loose" and len(base) == 1:
                    continue
                g, c = incr(d.copy(), base, [feat])
                L.append(f"| {feat} | {setname} | {len(d)} | {'+'.join(base)} | {f3(g, True)} | {f3(c, True)} | "
                         f"{pts_per_sd(d, base, feat):+.2f} |")
    text = "\n".join(L) + "\n"
    (sc.DATA / "experiments.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
