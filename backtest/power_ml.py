"""'Everything' model as a ceiling test for Desk Power on the clean Sleeper set.

Per team at checkpoint W: top-9 starters (by next-week projection) x player features + team features.
Target = within-league z of ROS PF/g. GBM, MLP and a pairwise GBM, all leave-one-season-out (checkpoints
pooled, cp as a feature). Same leak-free rules as power_inputs: nothing after week W+1 projections.

    .venv/bin/python backtest/power_ml.py     # appends to ~/Developer/sleeper-data/power_inputs.md
"""
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
import power_inputs as pi  # noqa: E402
import sleeper_crawl as sc  # noqa: E402
import sleeper_experiments as se  # noqa: E402
import sleeper_power as sp  # noqa: E402

NSLOT = 9
POS4 = ("QB", "RB", "WR", "TE")
PF = ("proj", "ppg", "games", "last3", "d1", "d3", "opp_sh", "snap_sh")
TEAM = ("pf_g", "proj_pf_g", "over", "adds_wk", "st", "bench3")
ROWS_PK = sc.DATA / "power_rows.pkl"
RNG = np.random.default_rng(7)


def load_stats(s, w):
    rows = sc.get(f"https://api.sleeper.app/stats/nfl/{s}/{w}?season_type=regular") or []
    out, team_opp = {}, collections.Counter()
    for r in rows:
        st = r.get("stats") or {}
        if st.get("pts_ppr") is None:
            continue
        opp = (st.get("rec_tgt") or 0) + (st.get("rush_att") or 0)
        out[r["player_id"]] = (st["pts_ppr"], st.get("gp") or 0, opp, st.get("off_snp") or 0,
                               st.get("tm_off_snp") or 0, r.get("team"))
        team_opp[r.get("team")] += opp
    return out, team_opp


def player_agg(SS, W):
    """Per player through week W: games, ppg, last-3 ppg, opportunity share, snap share."""
    a = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0, 0.0, [], ])
    for w in range(1, W + 1):
        stats, topp = SS[w]
        for p, (pts, gp, opp, snp, tsnp, tm) in stats.items():
            if gp <= 0:
                continue
            x = a[p]
            x[0] += 1
            x[1] += pts
            x[2] += opp
            x[3] += topp.get(tm, 0)
            x[4] += snp
            x[5] += tsnp
            x[6].append((w, pts))
    out = {}
    for p, x in a.items():
        l3 = [v for w, v in x[6] if w > W - 3]
        out[p] = {"games": x[0], "ppg": x[1] / x[0], "last3": np.mean(l3) if l3 else np.nan,
                  "opp_sh": x[2] / x[3] if x[3] else np.nan, "snap_sh": x[4] / x[5] if x[5] else np.nan}
    return out


def build():
    if ROWS_PK.exists():
        team = pd.read_pickle(ROWS_PK)
    else:
        team, _ = pi.sleeper_rows()
        team.to_pickle(ROWS_PK)
    seasons = list(sc.SEASONS)
    P = {s: {w: se.load("projections", s, w) for w in range(1, 18)} for s in seasons}
    S = {s: {w: se.load("stats", s, w) for w in range(1, 18)} for s in seasons}
    info = se.player_info(P, S)
    SS = {s: {w: load_stats(s, w) for w in range(1, 12)} for s in seasons}
    agg = {(s, W): player_agg(SS[s], W) for s in seasons for W in pi.CPS}
    V = lambda s, w: {**info[s][w]["projmean"], **{p: v for p, (v, _) in P[s][w + 1].items()}}
    Vc = {(s, w): V(s, w) for s in seasons for W in pi.CPS for w in (W, W - 1, W - 3)}
    objs = sp.league_objects()
    meta = {m["league_id"]: m for m in json.loads((sc.DATA / "kept_leagues.json").read_text())}
    feats = []
    for (lid, W), g in team.groupby(["league", "cp"]):
        m, s = meta[lid], int(g.season.iat[0])
        slots = [x for x in objs[lid]["roster_positions"] if x in sp.SLOT_ELIG]
        mu = {e["roster_id"]: e["players"] or [] for e in sc.get(f"{sc.BASE}/league/{lid}/matchups/{W}", offline=True) or []}
        v0, v1, v3, A = Vc[(s, W)], Vc[(s, W - 1)], Vc[(s, W - 3)], agg[(s, W)]
        for r in g.rid:
            _, chosen, left = se.lineup(mu.get(r, []), v0, slots)
            top = sorted((p for p, _ in chosen), key=lambda p: -v0.get(p, 0.0))[:NSLOT]
            f = {"league": lid, "cp": W, "rid": r, "n_teams": len(g),
                 "bench3": sum(sorted((v0.get(q, 0.0) for q in left), reverse=True)[:3])}
            for k in range(NSLOT):
                p = top[k] if k < len(top) else None
                ps = se.pos_of(p) if p else None
                for q in POS4:
                    f[f"s{k}_{q}"] = float(ps == q)
                a = A.get(p, {}) if p else {}
                f[f"s{k}_proj"] = v0.get(p, np.nan) if p else np.nan
                f[f"s{k}_d1"] = v0.get(p, 0.0) - v1.get(p, 0.0) if p else np.nan
                f[f"s{k}_d3"] = v0.get(p, 0.0) - v3.get(p, 0.0) if p else np.nan
                for c in ("ppg", "games", "last3", "opp_sh", "snap_sh"):
                    f[f"s{k}_{c}"] = a.get(c, 0.0 if c == "games" else np.nan) if p else np.nan
            feats.append(f)
    df = team.merge(pd.DataFrame(feats), on=["league", "cp", "rid"])
    g = df.groupby("gid")
    for c in TEAM:
        sd = g[c].transform("std")
        df["lz_" + c] = ((df[c] - g[c].transform("mean")) / sd.where(sd > 0)).fillna(0.0)
    return df


def zpred(df, p):
    p = pd.Series(p, index=df.index)
    g = p.groupby(df.gid)
    return (p - g.transform("mean")) / g.transform("std").replace(0, 1)


def loso_model(df, X, make, y="z_ros_pf_g"):
    pred = pd.Series(0.0, index=df.index)
    for s in sorted(df.season.unique()):
        tr, te = df.season != s, df.season == s
        mdl = make().fit(X[tr.values], df.loc[tr, y].values)
        pred[te] = mdl.predict(X[te.values])
    return pred


def pairwise(df, X):
    """GBM on within-league pair differences; team score = mean P(beats each league-mate)."""
    gidx = [np.where(df.gid.values == g)[0] for g in df.gid.unique()]
    I, J = [], []
    for ix in gidx:
        i, j = np.triu_indices(len(ix), 1)
        I.append(ix[i])
        J.append(ix[j])
    I, J = np.concatenate(I), np.concatenate(J)
    y = df.ros_pf_g.values
    keep = y[I] != y[J]
    I, J = I[keep], J[keep]
    flip = RNG.random(len(I)) < 0.5
    I, J = np.where(flip, J, I), np.where(flip, I, J)
    season = df.season.values[I]
    score = np.zeros(len(df))
    cnt = np.zeros(len(df))
    for s in sorted(set(season)):
        tr, te = season != s, season == s
        mdl = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
        mdl.fit(X[I[tr]] - X[J[tr]], (y[I[tr]] > y[J[tr]]).astype(int))
        pr = mdl.predict_proba(X[I[te]] - X[J[te]])[:, 1]
        np.add.at(score, I[te], pr)
        np.add.at(score, J[te], 1 - pr)
        np.add.at(cnt, I[te], 1)
        np.add.at(cnt, J[te], 1)
    return pd.Series(score / np.maximum(cnt, 1), index=df.index)


def compare(df, base_m, pred):
    m = pi.per_gid(df, pred)
    d = m[["spear", "pair"]] - base_m.loc[m.index, ["spear", "pair"]]
    d["league"], d["cp"] = m.league, m.cp
    out = []
    for lab, sub in [("pooled", d)] + [(f"wk{W}", d[d.cp == W]) for W in pi.CPS]:
        g = sub.groupby("league")[["spear", "pair"]].mean()
        out.append((lab, m.loc[sub.index].spear.mean(), pi.boot(g.spear), pi.boot(g.pair)))
    return out


def main():
    pk = sc.DATA / "power_ml_rows.pkl"
    if pk.exists():
        df = pd.read_pickle(pk)
    else:
        df = build()
        df.to_pickle(pk)
    pi.z(df, ["st", "pf_g", "proj_pf_g", "over", "ros_pf_g"])
    cols = [c for c in df.columns if c.startswith("s") and c[1:2].isdigit()] + list(TEAM) + \
        ["lz_" + c for c in TEAM] + ["n_teams", "cp"]
    X = df[cols].values.astype(float)
    Xf = np.nan_to_num(X, nan=0.0)
    base_p, _ = pi.loso(df, ["st", "pf_g"])
    split_p, _ = pi.loso(df, ["st", "proj_pf_g", "over"])
    bm, smm = pi.per_gid(df, base_p), pi.per_gid(df, split_p)
    gbm = lambda: HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                                min_samples_leaf=50, l2_regularization=1.0, random_state=0)
    mlp = lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-2, early_stopping=True,
                                                               max_iter=300, random_state=0))
    preds = {"GBM": loso_model(df, X, gbm), "MLP": loso_model(df, Xf, mlp)}
    print("gbm/mlp done", flush=True)
    preds["pairwise GBM"] = pairwise(df, X)
    print("pairwise done", flush=True)
    preds["blend+split + GBM (avg of z)"] = (zpred(df, split_p) + zpred(df, preds["GBM"])) / 2
    shuf = df.copy()
    shuf["z_ros_pf_g"] = shuf.groupby("gid").z_ros_pf_g.transform(lambda x: RNG.permutation(x.values))
    shuf_p = loso_model(shuf, X, gbm)
    L = [f"\n## 6. 'Everything' model ceiling test (power_ml.py; {df.league.nunique()} clean leagues, {len(df)} team rows, "
         f"{len(cols)} features)\n",
         "Features: top-9 starters by next-week projection x (pos, next-week proj, ppg to date, games, last-3 ppg, "
         "proj change 1 and 3 wks, target+carry share of team, snap share) + team PF/g, projected PF/g, over/under, "
         "adds/wk, strength, bench top-3 proj (raw and within-league z) + league size + cp. Target = within-league "
         "z of ROS PF/g. LOSO by season, checkpoints pooled.\n",
         "| model | cp | Spearman | diff vs baseline blend | pairwise diff vs blend | diff vs blend+split | pairwise diff vs split |",
         "|---|---|---|---|---|---|---|"]
    for name, p in [("baseline blend", base_p), ("blend+split", split_p)] + list(preds.items()):
        r1, r2 = compare(df, bm, p), compare(df, smm, p)
        for (lab, sp_, d1, p1), (_, _, d2, p2) in zip(r1, r2):
            L.append(f"| {name} | {lab} | {sp_:.3f} | {d1} | {p1} | {d2} | {p2} |")
    sm = pi.per_gid(df, shuf_p)
    L.append(f"\nLeakage check, GBM trained on target shuffled within league: Spearman vs real target "
             f"{pi.boot(sm.groupby('league').spear.mean())} (should be ~0).\n")
    # permutation importance, GBM, each held-out season, averaged
    imp = np.zeros(len(cols))
    for s in sorted(df.season.unique()):
        tr, te = (df.season != s).values, (df.season == s).values
        mdl = gbm().fit(X[tr], df.z_ros_pf_g.values[tr])
        imp += permutation_importance(mdl, X[te], df.z_ros_pf_g.values[te], n_repeats=3, random_state=0).importances_mean
    imp /= df.season.nunique()
    top = np.argsort(-imp)[:10]
    L.append("GBM permutation importance (drop in held-out R^2, mean over seasons), top 10: " +
             ", ".join(f"{cols[i]} {imp[i]:.4f}" for i in top) + "\n")
    text = "\n".join(L) + "\n"
    with open(sc.DATA / "power_inputs.md", "a") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
