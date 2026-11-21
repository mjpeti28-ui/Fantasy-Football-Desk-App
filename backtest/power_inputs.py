"""Does any new input improve Desk Power (strength + PF/g blend)?

Leak-free setup from power_insights: strength at week w = best lineup of the week-w roster valued at
{projmean through w, Sleeper week w+1 projection}, smoothed over weeks W-2..W. Nothing after W+1 is used.
Target ROS PF/g. Each test = baseline [strength, PF/g] + candidate, z within league-checkpoint, OLS weights
fit leave-one-season-out separately per checkpoint. Diffs vs baseline with league-bootstrap 95% CIs.

    .venv/bin/python backtest/power_inputs.py      # writes ~/Developer/sleeper-data/power_inputs.md
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import power_my_league as pm  # noqa: E402
import sleeper_crawl as sc  # noqa: E402
import sleeper_experiments as se  # noqa: E402
import sleeper_power as sp  # noqa: E402

CPS = (4, 6, 8, 10)
ELITE_N = {"QB": 12, "RB": 24, "WR": 24, "TE": 12}
BENCH_W = 0.15
RNG = np.random.default_rng(5)
LEDGER = pm.HIST.parent / "ledger"
WEEKLY = ("st", "vorp", "vorp_full", "scar", "elite")


# ---------------- per-week roster features ----------------

def by_pos(V):
    out = collections.defaultdict(list)
    for p, v in sorted(V.items(), key=lambda x: -x[1]):
        out[se.pos_of(p)].append(p)
    return out


def team_feats(rosters, V, slots, ranked):
    """rosters {team: [pid]} -> {team: {st, vorp, vorp_full, scar, elite}} for one league-week."""
    rostered = {p for r in rosters.values() for p in r}
    repl = {}
    for ps, lst in ranked.items():
        q = next((q for q in lst if q not in rostered), None)
        repl[ps] = V.get(q, 0.0) if q else 0.0
    rp = lambda p: repl.get(se.pos_of(p), 0.0)
    lu = {t: se.lineup(r, V, slots) for t, r in rosters.items()}
    sd = {}
    for ps in ("QB", "RB", "WR", "TE", "K", "DEF"):
        v = [V.get(p, 0.0) for t in lu for p, _ in lu[t][1] if se.pos_of(p) == ps]
        if len(v) > 1:
            sd[ps] = np.std(v)
    msd = np.mean([x for x in sd.values() if x > 0]) if sd else 1.0
    elite = set()
    for ps, n in ELITE_N.items():
        elite |= set(sorted((p for p in rostered if se.pos_of(p) == ps), key=lambda p: -V.get(p, 0.0))[:n])
    etot = sum(V.get(p, 0.0) for p in elite) or 1.0
    out = {}
    for t, (tot, chosen, left) in lu.items():
        vorp = sum(V.get(p, 0.0) - rp(p) for p, _ in chosen)
        out[t] = {"st": tot, "vorp": vorp,
                  "vorp_full": vorp + BENCH_W * sum(max(0.0, V.get(q, 0.0) - rp(q)) for q in left),
                  "scar": sum((V.get(p, 0.0) - rp(p)) * sd.get(se.pos_of(p), msd) / msd for p, _ in chosen),
                  "elite": sum(V.get(p, 0.0) for p in rosters[t] if p in elite) / etot}
    return out


# ---------------- Sleeper ----------------

def sleeper_rows():
    seasons = list(sc.SEASONS)
    P = {s: {w: se.load("projections", s, w) for w in range(1, 18)} for s in seasons}
    S = {s: {w: se.load("stats", s, w) for w in range(1, 18)} for s in seasons}
    info = se.player_info(P, S)
    objs = sp.league_objects()
    kept = json.loads((sc.DATA / "kept_leagues.json").read_text())
    qual = {m["league_id"]: sc.analyze_league(m) for m in kept}
    loose = [m for m in kept if qual[m["league_id"]]["complete"] and m["type"] in (0, 1, 2)
             and 8 <= m["teams"] <= 14 and qual[m["league_id"]]["inactive_strict"] <= 1]
    clean = [m for m in loose if qual[m["league_id"]]["inactive_strict"] == 0]
    if len(sys.argv) > 1:  # quick smoke run on the first N clean leagues
        clean = clean[:int(sys.argv[1])]
    txc = lambda lid, w: [t for t in sc.get(f"{sc.BASE}/league/{lid}/transactions/{w}", offline=True) or []
                          if t.get("status") == "complete"]
    # prior-season activity: adds/wk over the regular season, per user-season, any loose league
    ua = collections.defaultdict(list)
    for m in loose:
        reg_end = min((m["playoff_week_start"] or 15) - 1, 17)
        owners = {int(k): (v or [None])[0] for k, v in (m.get("owners") or {}).items()}
        c = collections.Counter(r for w in range(1, reg_end + 1) for t in txc(m["league_id"], w)
                                if t["type"] in ("waiver", "free_agent") for r in (t.get("adds") or {}).values())
        for r, u in owners.items():
            if u:
                ua[(u, m["season"])].append(c[r] / reg_end)
    prior_act = {k: np.mean(v) for k, v in ua.items()}
    ranked_c = {}
    rows = []
    for li, m in enumerate(clean):
        s, lid, lg = m["season"], m["league_id"], objs[m["league_id"]]
        reg_end = min((m["playoff_week_start"] or 15) - 1, 17)
        slots = [x for x in lg["roster_positions"] if x in sp.SLOT_ELIG]
        owners = {int(k): (v or [None])[0] for k, v in (m.get("owners") or {}).items()}
        mu = {w: sc.get(f"{sc.BASE}/league/{lid}/matchups/{w}", offline=True) or [] for w in range(1, reg_end + 1)}
        rids = sorted(e["roster_id"] for e in mu[1])
        ix = {r: i for i, r in enumerate(rids)}
        n = len(rids)
        Sc, Pj = np.zeros((reg_end, n)), np.zeros((reg_end, n))
        for w in range(1, reg_end + 1):
            a = {p: v for p, (v, _) in S[s][w].items()}
            pr = {p: v for p, (v, _) in P[s][w].items()}
            for e in mu[w]:
                if e["roster_id"] in ix:
                    stt = [p for p in e["starters"] or [] if p not in ("0", None)]
                    Sc[w - 1, ix[e["roster_id"]]] = sum(a.get(p, 0) for p in stt)
                    Pj[w - 1, ix[e["roster_id"]]] = sum(pr.get(p, 0) for p in stt)
        adds = np.zeros((reg_end, n))
        for w in range(1, reg_end + 1):
            for t in txc(lid, w):
                if t["type"] in ("waiver", "free_agent"):
                    for r in (t.get("adds") or {}).values():
                        if r in ix:
                            adds[w - 1, ix[r]] += 1
        F = {}
        for w in range(1, reg_end - 1):
            V = {**info[s][w]["projmean"], **{p: v for p, (v, _) in P[s][w + 1].items()}}
            if (s, w) not in ranked_c:
                ranked_c[(s, w)] = by_pos(V)
            ros_ = {r: e["players"] or [] for e in mu[w] for r in [e["roster_id"]] if r in ix}
            f = team_feats(ros_, V, slots, ranked_c[(s, w)])
            F[w] = {k: np.array([f[r][k] for r in rids]) for k in WEEKLY}
        for W in CPS:
            if W > reg_end - 2:
                continue
            sm = {k: np.mean([F[W - j][k] for j in range(3)], 0) for k in WEEKLY}
            for r in rids:
                i = ix[r]
                u = owners.get(r)
                rows.append({"league": lid, "season": s, "cp": W, "rid": r,
                             **{k: sm[k][i] for k in WEEKLY}, "st1": F[W]["st"][i], "vorp1": F[W]["vorp"][i],
                             "pf_g": Sc[:W, i].mean(), "proj_pf_g": Pj[:W, i].mean(),
                             "over": Sc[:W, i].mean() - Pj[:W, i].mean(), "adds_wk": adds[:W, i].mean(),
                             "prior_adds": prior_act.get((u, s - 1), np.nan) if u else np.nan,
                             "ros_pf_g": Sc[W:reg_end, i].mean()})
        if li % 200 == 0:
            print(f"{li}/{len(clean)}", flush=True)
    df = pd.DataFrame(rows)
    df["gid"] = df.league + "_" + df.cp.astype(str)
    return df, len(clean)


# ---------------- evaluation ----------------

def z(df, cols):
    g = df.groupby("gid")
    for c in cols:
        sd = g[c].transform("std")
        df["z_" + c] = ((df[c] - g[c].transform("mean")) / sd.where(sd > 0)).fillna(0.0)


def loso(df, feats):
    """Per-checkpoint, leave-one-season-out OLS on z features -> prediction; also pooled coefs per cp."""
    zf = ["z_" + f for f in feats]
    pred, coefs = pd.Series(0.0, index=df.index), {}
    for W, d in df.groupby("cp"):
        coefs[W] = np.linalg.lstsq(d[zf].values, d.z_ros_pf_g.values, rcond=None)[0]
        for s in d.season.unique():
            tr, te = d[d.season != s], d[d.season == s]
            b = np.linalg.lstsq(tr[zf].values, tr.z_ros_pf_g.values, rcond=None)[0]
            pred[te.index] = te[zf].values @ b
    return pred, coefs


def per_gid(df, pred):
    out = []
    for gid, d in df.groupby("gid"):
        x, y = pred[d.index].values, d.ros_pf_g.values
        i, j = np.triu_indices(len(x), 1)
        dx, dy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
        ok = (dx != 0) & (dy != 0)
        out.append((gid, d.league.iat[0], d.cp.iat[0], pm.sp_corr(x, y), np.mean(dx[ok] == dy[ok])))
    return pd.DataFrame(out, columns=["gid", "league", "cp", "spear", "pair"]).set_index("gid")


def boot(v, reps=2000):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    b = v[RNG.integers(0, len(v), (reps, len(v)))].mean(1)
    return f"{v.mean():+.4f} [{np.percentile(b, 2.5):+.4f},{np.percentile(b, 97.5):+.4f}]"


def compare(df, base_m, feats):
    """Rows: pooled + each cp; Spearman diff and pairwise diff vs baseline, league bootstrap."""
    p, coefs = loso(df, feats)
    m = per_gid(df, p)
    d = m[["spear", "pair"]] - base_m.loc[m.index, ["spear", "pair"]]
    d["league"], d["cp"] = m.league, m.cp
    res = []
    for lab, sub in [("pooled", d)] + [(f"wk{W}", d[d.cp == W]) for W in CPS]:
        g = sub.groupby("league")[["spear", "pair"]].mean()
        res.append((lab, len(g), boot(g.spear), boot(g.pair)))
    return res, coefs, p


def table(L, title, df, base_m, feats, note=""):
    res, coefs, p = compare(df, base_m, feats)
    L += [f"### {title}\n", f"features: {' + '.join(feats)}. {note}\n",
          "| cp | leagues | Spearman diff vs base [95% CI] | pairwise diff vs base [95% CI] |", "|---|---|---|---|"]
    L += [f"| {a} | {b} | {c} | {e} |" for a, b, c, e in res]
    L += ["\nFitted z-weights (all seasons) by cp: " +
          "; ".join(f"wk{W}: " + ", ".join(f"{f} {c:+.3f}" for f, c in zip(feats, coefs[W])) for W in sorted(coefs)), ""]
    return coefs, p


def sleeper_section(L):
    df, n_clean = sleeper_rows()
    allc = ["st", "pf_g", "proj_pf_g", "over", "adds_wk", "prior_adds", "vorp", "vorp_full", "scar", "elite", "ros_pf_g"]
    z(df, allc)
    base = ["st", "pf_g"]
    bp, bco = loso(df, base)
    bm = per_gid(df, bp)
    L += [f"## Sleeper, clean set ({n_clean} leagues, {df.gid.nunique()} league-checkpoints, {len(df)} team rows)\n",
          "Baseline blend = LOSO OLS on z(strength 3-wk smoothed), z(PF/g), fit per checkpoint.\n",
          "| cp | baseline Spearman | baseline pairwise | baseline z-weights (strength, PF/g) |", "|---|---|---|---|"]
    for W in CPS:
        b = bm[bm.cp == W]
        L.append(f"| wk{W} | {b.spear.mean():.3f} | {b.pair.mean():.3f} | {bco[W][0]:+.3f}, {bco[W][1]:+.3f} |")
    L.append("")

    L.append("## 1. PF split: projected PF/g + over/under replaces PF/g\n")
    co, sp_ = table(L, "1. split", df, bm, ["st", "proj_pf_g", "over"],
                    "proj_pf_g = mean of started players' Sleeper projections (weeks 1..W); over = PF/g - proj_pf_g.")
    # luck-heavy teams: rank change, split vs baseline
    df["_rb"] = df.assign(x=bp).groupby("gid").x.rank(ascending=False)
    df["_rs"] = df.assign(x=sp_).groupby("gid").x.rank(ascending=False)
    df["_rp"] = df.groupby("gid").pf_g.rank(ascending=False)
    L += ["Luck-heavy teams: rank under split minus rank under baseline (positive = split ranks them lower).\n",
          "| group | n | median rank chg | mean rank chg | mean over | mean ROS rank minus base rank |", "|---|---|---|---|---|---|"]
    df["_rr"] = df.groupby("gid").ros_pf_g.rank(ascending=False)
    for lab, msk in (("over +10..+14", df.over.between(10, 14)), ("over >= +12", df.over >= 12),
                     ("over <= -12", df.over <= -12)):
        d = df[msk]
        L.append(f"| {lab} | {len(d)} | {np.median(d._rs - d._rb):+.1f} | {(d._rs - d._rb).mean():+.2f} | "
                 f"{d.over.mean():+.1f} | {(d._rr - d._rb).mean():+.2f} |")
    L.append("")

    L.append("## 2. Activity\n")
    table(L, "2a. same-season adds/wk to date", df, bm, base + ["adds_wk"], "waiver + FA adds, legs 1..W, per week.")
    sub = df[df.prior_adds.notna()].copy()
    sub = sub[sub.groupby("gid").gid.transform("size") >= 6].copy()
    z(sub, allc)
    sbp, _ = loso(sub, base)
    sbm = per_gid(sub, sbp)
    table(L, f"2b. prior-season adds/wk (subset with user's prior season: {sub.league.nunique()} leagues, {len(sub)} rows)",
          sub, sbm, base + ["prior_adds"], "baseline refit on the same subset; teams without a prior dropped.")

    L.append("## 3. VORP\n")
    g = df[["gid", "st1", "vorp1"]].groupby("gid")
    ident = g.apply(lambda d: bool((d.st1.rank() == d.vorp1.rank()).all()))
    rho = g.apply(lambda d: pm.sp_corr(d.st1, d.vorp1))
    L.append(f"Starter VORP vs strength (same week, unsmoothed): identical within-league ranking in {ident.mean():.1%} "
             f"of league-checkpoints; mean within-league Spearman {rho.mean():.4f}. Not identical because flex slots "
             "subtract the replacement level of whichever position fills them.\n")
    table(L, "3a. starter VORP (added)", df, bm, base + ["vorp"])
    table(L, "3a'. starter VORP (replaces strength)", df, bm, ["vorp", "pf_g"])
    table(L, f"3b. full-roster VORP, bench above replacement x{BENCH_W} (added)", df, bm, base + ["vorp_full"])
    table(L, "3b'. full-roster VORP (replaces strength)", df, bm, ["vorp_full", "pf_g"])
    table(L, "3c. scarcity-scaled VORP (added)", df, bm, base + ["scar"],
          "each starter (proj - best FA at pos) x sd(starter proj at pos in league) / mean sd over positions.")
    table(L, "3c'. scarcity-scaled VORP (replaces strength)", df, bm, ["scar", "pf_g"])

    L.append("## 4. Elite market share\n")
    table(L, "4. elite share (added)", df, bm, base + ["elite"],
          "share of league's top-12 QB / 24 RB / 24 WR / 12 TE projected pts (all rostered players) owned.")
    return {W: bco[W] for W in CPS}, co


# ---------------- ESPN (Max's league) ----------------

def espn_section(L, sl_base, sl_split):
    pos_all = {}
    for y in range(2021, 2026):
        pw = pd.read_csv(pm.HIST / str(y) / "players_weekly.csv")
        pos_all.update({f"e{i}": ("DEF" if p == "DST" else p) for i, p in zip(pw.espn_id, pw.position)})
    for p, ps in pos_all.items():
        sp.POS.setdefault(p, set()).add(ps)
    rows, desk_rows = [], []
    for y in range(2021, 2026):
        d = pm.load_season(y)
        teams, Sc, rw, pj = d["teams"], d["Sc"], d["rw"], d["pj"]
        slots = [pm.ESPN_SLOT[s] for s, c in d["st"]["lineup_slot_counts"].items() if s in pm.ESPN_SLOT for _ in range(c)]
        proj = {w: {f"e{i}": v for i, v in zip(g.espn_id, g.espn_proj)} for w, g in pj.groupby("week")}
        roster = {(w, t): [f"e{i}" for i in g.espn_id] for (w, t), g in rw.groupby(["week", "team_id"])}
        n, ix = len(teams), {t: i for i, t in enumerate(teams)}
        Pj = np.zeros(Sc.shape)
        for (w, t), g in rw[rw.week <= Sc.shape[0]].groupby(["week", "team_id"]):
            Pj[w - 1, ix[t]] = sum(proj.get(w, {}).get(f"e{i}", 0) for i, sl in zip(g.espn_id, g.lineup_slot)
                                   if sl not in ("BE", "IR"))
        psum, pn, F, Fd = collections.Counter(), collections.Counter(), {}, {}
        for w in range(1, 11):
            for p, v in proj.get(w, {}).items():
                psum[p] += v
                pn[p] += 1
            V = {**{p: psum[p] / pn[p] for p in pn}, **proj.get(w + 1, {})}
            f = team_feats({t: roster.get((w, t), []) for t in teams}, V, slots, by_pos(V))
            F[w] = {k: np.array([f[t][k] for t in teams]) for k in WEEKLY}
            lf = LEDGER / str(y) / f"wk{w + 1:02d}_eq0_players.csv"
            if lf.exists():
                lp = pd.read_csv(lf)
                col = f"xp_w{w + 1}"
                vd = np.where(lp[col] > 0, lp[col], lp.est_ppg)  # bye week -> est_ppg, like the ESPN bye fill
                Vd = {f"e{i}": v for i, v in zip(lp.espn_id, vd)}
                Fd[w] = np.array([se.lineup(roster.get((w, t), []), Vd, slots)[0] for t in teams])
        for W in CPS:
            sm = {k: np.mean([F[W - j][k] for j in range(3)], 0) for k in WEEKLY}
            for t in teams:
                i = ix[t]
                rows.append({"league": str(y), "season": y, "cp": W, "rid": t, **{k: sm[k][i] for k in WEEKLY},
                             "pf_g": Sc[:W, i].mean(), "proj_pf_g": Pj[:W, i].mean(),
                             "over": Sc[:W, i].mean() - Pj[:W, i].mean(), "ros_pf_g": Sc[W:d["reg"], i].mean()})
        for W in (4, 6, 7):
            if all(W - j in Fd for j in range(3)):
                sd_ = np.mean([Fd[W - j] for j in range(3)], 0)
                se_ = np.mean([F[W - j]["st"] for j in range(3)], 0)
                for t in teams:
                    i = ix[t]
                    desk_rows.append({"season": y, "cp": W, "desk": sd_[i], "espn": se_[i], "pf_g": Sc[:W, i].mean(),
                                      "ros": Sc[W:d["reg"], i].mean()})
    df = pd.DataFrame(rows)
    df["gid"] = df.league + "_" + df.cp.astype(str)
    z(df, ["st", "pf_g", "proj_pf_g", "over", "vorp", "scar", "ros_pf_g"])
    L += ["## 5. Max's ESPN league, 2021-25 (12 teams x 5 seasons per cp; n is tiny, CIs bootstrap the 5 seasons)\n",
          "Sleeper-fit weights transferred (not refit) so nothing is tuned on this league; LOSO refit on 48 "
          "rows/cp in the last column.\n",
          "| test | cp | Spearman (transfer) | diff vs base | pairwise diff | LOSO-refit Spearman diff |", "|---|---|---|---|---|---|"]
    feats = {"base": ["st", "pf_g"], "1 split": ["st", "proj_pf_g", "over"], "3a vorp+": ["st", "pf_g", "vorp"],
             "3c scar+": ["st", "pf_g", "scar"]}
    # transfer weights: base and split from Sleeper; 3a/3c use Sleeper base weights with candidate replacing strength
    def tw(name, W):
        if name == "1 split":
            return sl_split[W]
        return sl_base[W]
    tfeats = {"base": ["st", "pf_g"], "1 split": ["st", "proj_pf_g", "over"], "3a vorp+": ["vorp", "pf_g"],
              "3c scar+": ["scar", "pf_g"]}
    mets = {}
    for name in feats:
        tp = pd.Series(0.0, index=df.index)
        for W in CPS:
            m = df.cp == W
            tp[m] = df.loc[m, ["z_" + f for f in tfeats[name]]].values @ tw(name, W)
        lp, _ = loso(df, feats[name])
        mets[name] = (per_gid(df, tp), per_gid(df, lp))
    for name in feats:
        for W in list(CPS) + ["all"]:
            sel = (lambda m: m) if W == "all" else (lambda m, W=W: m[m.cp == W])
            t, b = sel(mets[name][0]), sel(mets["base"][0])
            lo, lb = sel(mets[name][1]), sel(mets["base"][1])
            L.append(f"| {name} | {W} | {t.spear.mean():.3f} | {boot(t.spear - b.spear) if name != 'base' else '-'} | "
                     f"{boot(t.pair - b.pair) if name != 'base' else '-'} | "
                     f"{boot(lo.spear - lb.spear) if name != 'base' else f'{lb.spear.mean():.3f} (base)'} |")
    L.append("\n3a/3c transfer = Sleeper baseline weights with VORP/scarcity in strength's place. "
             f"Over/under mean {df.over.mean():+.1f}, sd {df.over.std():.1f} pts/wk (ESPN projections, league scoring).\n")
    dr = pd.DataFrame(desk_rows)
    if len(dr):
        L += ["### Desk LOSO ledger vs ESPN projections as the strength input\n",
              "Desk = history/ledger/<y>/wk<W+1>_eq0_players.csv xp_w<W+1> (bye -> est_ppg); ledger only covers "
              "weeks 3-8, so checkpoints 4, 6, 7 (7 stands in for 8). Same rosters, same 3-wk smoothing. "
              "Blend = Sleeper-fit baseline weights for the nearest cp.\n",
              "| cp | n seasons | Spearman strength ESPN | Desk | blend ESPN | blend Desk | pairwise blend ESPN | Desk |",
              "|---|---|---|---|---|---|---|---|"]
        for W, g in dr.groupby("cp"):
            w_ = sl_base[min(CPS, key=lambda c: abs(c - W))]
            r = collections.defaultdict(list)
            for s, h in g.groupby("season"):
                for k in ("espn", "desk"):
                    r[k].append(pm.sp_corr(h[k], h.ros))
                    bl = w_[0] * pm.zs(h[k]) + w_[1] * pm.zs(h.pf_g)
                    r["b" + k].append(pm.sp_corr(bl, h.ros))
                    x, yv = bl.values if hasattr(bl, "values") else bl, h.ros.values
                    i, j = np.triu_indices(len(x), 1)
                    dx, dy = np.sign(x[i] - x[j]), np.sign(yv[i] - yv[j])
                    ok = (dx != 0) & (dy != 0)
                    r["p" + k].append(np.mean(dx[ok] == dy[ok]))
            L.append(f"| {W} | {g.season.nunique()} | {np.mean(r['espn']):.3f} | {np.mean(r['desk']):.3f} | "
                     f"{np.mean(r['bespn']):.3f} | {np.mean(r['bdesk']):.3f} | {np.mean(r['pespn']):.3f} | {np.mean(r['pdesk']):.3f} |")
        diff = [pm.sp_corr(pm.zs(h.desk), h.ros) - pm.sp_corr(pm.zs(h.espn), h.ros) for _, h in dr.groupby(["season", "cp"])]
        L.append(f"\nDesk minus ESPN strength Spearman over all season-cps (n={len(diff)}): {boot(diff)}\n")


def main():
    L = ["# Power inputs: does any new input improve Desk Power?\n",
         "Target ROS PF/g. Strength = leak-free 3-wk smoothed next-week projection lineup (byes filled). "
         "No projection after week W+1 used. Diffs are candidate minus baseline, per league averaged over "
         "checkpoints for 'pooled'; 95% CI = league bootstrap.\n"]
    sl_base, sl_split = sleeper_section(L)
    espn_section(L, sl_base, sl_split)
    text = "\n".join(L) + "\n"
    (sc.DATA / "power_inputs.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
