"""Creative PFF experiments (2026-10-17). Shared loaders + one subcommand per experiment.

Data: PFF caches copied from the Mac (gitignored): PFF_MAIN = main checkout .cache/pff (raw v2 reports, full
seasons 2021-25), PFF_WT = desk-pff-wt .cache/pff (v1 facet summaries, cumulative w01-NN and 3-week windows).
League points: history/actuals/<season>/players_weekly.csv. Preseason ECR: history/nflverse/ecr.csv.

  python3 backtest/pff_creative.py nextseason|roles|ranges|injury|waivers

Results: pff/creative.md in the project files. Lead: QB twpRate -> next-season ppg beyond ECR; the rest is null.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PFF_MAIN = Path(os.environ.get("PFF_MAIN", ROOT / ".cache" / "main" / "pff"))
PFF_WT = Path(os.environ.get("PFF_WT", ROOT / ".cache" / "wt" / "pff"))
SEAS = [2021, 2022, 2023, 2024, 2025]
POS = ["QB", "RB", "WR", "TE"]
RNG = np.random.default_rng(7)


def rows_of(body):
    if isinstance(body, list):
        return body
    if isinstance(body, dict):
        if isinstance(body.get("rows"), list):
            return body["rows"]
        best = []
        for v in body.values():
            r = rows_of(v) if isinstance(v, (dict, list)) else []
            if len(r) > len(best) and (not r or isinstance(r[0], dict)):
                best = r
        return best
    return []


def ids() -> pd.DataFrame:
    d = pd.read_csv(PFF_MAIN / "db_playerids.csv", usecols=["pff_id", "espn_id", "fantasypros_id", "gsis_id", "birthdate", "draft_year", "position"])
    return d


def pff_map(col: str = "espn_id") -> dict:
    d = ids().dropna(subset=["pff_id", col])
    return dict(zip(d["pff_id"].astype(int), d[col].astype(int)))


def v2_report(rep: str, season: int, lo: int, hi: int) -> pd.DataFrame:
    for f in PFF_MAIN.glob(f"raw/v2_nfl_positions_reports_{rep}__*.json"):
        d = json.loads(f.read_text())
        p = d["params"]
        if p.get("season") == season and p.get("week") == lo and p.get("weekTo") == hi:
            return pd.DataFrame(rows_of(d["body"]))
    raise FileNotFoundError((rep, season, lo, hi))


def wt(fam: str, season: int, win: str) -> pd.DataFrame:
    d = json.loads((PFF_WT / fam / str(season) / f"{win}.json").read_text())
    return pd.DataFrame(rows_of(d))


def league_points() -> pd.DataFrame:
    out = []
    for y in SEAS + [2026]:
        p = ROOT / "history" / "actuals" / str(y) / "players_weekly.csv"
        if p.exists():
            d = pd.read_csv(p)
            out.append(d.assign(season=y))
    d = pd.concat(out, ignore_index=True)
    d = d[(d["kind"] == "actual") & d["position"].isin(POS) & (d["week"] <= 17)]
    return d


def season_ppg(pts: pd.DataFrame) -> pd.DataFrame:
    g = pts[pts["played"].astype(bool)].groupby(["season", "espn_id"])
    return pd.DataFrame({"ppg": g["points"].mean(), "games": g["points"].size(), "total": g["points"].sum()}).reset_index()


def preseason_ecr() -> pd.DataFrame:
    e = pd.read_csv(ROOT / "history" / "nflverse" / "ecr.csv")
    e = e[e["page_type"].isin([f"redraft-{p.lower()}" for p in POS])]
    e["season"] = e["scrape_date"].str[:4].astype(int)
    e = e[e["scrape_date"].str[5:] <= "09-05"]
    last = e.groupby("season")["scrape_date"].transform("max")
    e = e[e["scrape_date"] == last]
    fp = ids().dropna(subset=["fantasypros_id", "espn_id"])
    m = dict(zip(fp["fantasypros_id"].astype(int), fp["espn_id"].astype(int)))
    e["espn_id"] = e["id"].map(m)
    e["pos"] = e["page_type"].str.split("-").str[1].str.upper()
    return e.dropna(subset=["espn_id"]).astype({"espn_id": int})[["season", "espn_id", "pos", "ecr", "sd"]].drop_duplicates(["season", "espn_id"])


def num(d: pd.DataFrame) -> pd.DataFrame:
    for c in d.columns:
        if d[c].dtype == object:
            v = pd.to_numeric(d[c], errors="coerce")
            if v.notna().mean() > 0.8:
                d[c] = v
    return d


# ---------------------------------------------------------------- E1: next-season value
REC_F = {
    "yprr": "yprr", "route_rate": "routeRate", "tprr": None, "tgt_share": "teamTargetsPercent", "adot": "avgDepthOfTarget",
    "g_route": "gradesPassRoute", "g_off": "gradesOffense", "slot": "slotRate", "roe": "receptionsOe",
    "yac_rec": "yardsAfterCatchPerReception", "drop": "dropRate", "contested": "contestedCatchRate",
}
RUSH_F = ["grades_run", "yco_attempt", "elusive_rating", "breakaway_percent", "ypa", "mtf_att", "rush_share"]
PASS_F = ["gradesPass", "bttRate", "twpRate", "accuracyOe", "avgTimeToThrow", "pressureToSackRate", "epa_db", "scr_db", "passRateOe", "avgDepthOfTarget"]


def season_features(y: int) -> pd.DataFrame:
    pe = pff_map()
    r = num(v2_report("receiving", y, 1, 18))
    r["tprr"] = r["targets"] / r["routes"].replace(0, np.nan)
    rf = pd.DataFrame({"pff_id": r["playerId"].astype(int)})
    for k, c in REC_F.items():
        rf[k] = r[c or k].to_numpy(float) if (c or k) in r else r[k]
    rf["routes"] = r["routes"].to_numpy(float)
    ru = num(wt("rushing_summary", y, "w01-12"))
    ru["mtf_att"] = ru["avoided_tackles"] / ru["attempts"].replace(0, np.nan)
    tm = ru.groupby("franchise_id")["attempts"].transform("sum")
    ru["rush_share"] = ru["attempts"] / tm
    uf = ru[["player_id"] + [c for c in RUSH_F]].rename(columns={"player_id": "pff_id"}).assign(attempts=ru["attempts"].to_numpy(float))
    p = num(v2_report("passing", y, 1, 18))
    p["epa_db"] = p["epa"] / p["dropbacks"].replace(0, np.nan)
    p["scr_db"] = p["scrambles"] / p["dropbacks"].replace(0, np.nan)
    pf = p[["playerId"] + PASS_F].rename(columns={"playerId": "pff_id"}).assign(dropbacks=p["dropbacks"].to_numpy(float))
    f = rf.merge(uf, on="pff_id", how="outer").merge(pf, on="pff_id", how="outer")
    f["pff_id"] = f["pff_id"].astype(int)
    f["espn_id"] = f["pff_id"].map(pe)
    return f.dropna(subset=["espn_id"]).astype({"espn_id": int}).drop_duplicates("espn_id").assign(season=y)


PFF_SET = {
    "WR": ["yprr", "route_rate", "tprr", "tgt_share", "adot", "g_route", "slot", "roe", "yac_rec", "contested"],
    "TE": ["yprr", "route_rate", "tprr", "tgt_share", "adot", "g_route", "slot", "roe", "yac_rec"],
    "RB": ["grades_run", "yco_attempt", "elusive_rating", "breakaway_percent", "mtf_att", "rush_share", "yprr", "route_rate", "tprr"],
    "QB": ["gradesPass", "bttRate", "twpRate", "accuracyOe", "avgTimeToThrow", "pressureToSackRate", "epa_db", "scr_db"],
}
DEPTH = {"QB": 24, "RB": 48, "WR": 60, "TE": 20}


def nextseason_rows() -> pd.DataFrame:
    sp = season_ppg(league_points())
    ecr = preseason_ecr()
    idd = ids().dropna(subset=["espn_id"]).drop_duplicates("espn_id")
    bd = dict(zip(idd["espn_id"].astype(int), pd.to_datetime(idd["birthdate"], errors="coerce")))
    dy = dict(zip(idd["espn_id"].astype(int), idd["draft_year"]))
    out = []
    for y in SEAS[:-1]:
        f = season_features(y)
        prev = sp[sp["season"] == y].rename(columns={"ppg": "ppg0", "games": "g0", "total": "tot0"}).drop(columns="season")
        nxt = sp[sp["season"] == y + 1].rename(columns={"ppg": "ppg1", "games": "g1", "total": "tot1"}).drop(columns="season")
        e = ecr[ecr["season"] == y + 1].drop(columns="season")
        d = e.merge(prev, on="espn_id").merge(f, on="espn_id", how="left").merge(nxt, on="espn_id", how="left")
        d = d[d["ecr"] <= d["pos"].map(DEPTH)]
        d["season"] = y + 1
        d["age"] = [(pd.Timestamp(f"{y + 1}-09-01") - bd.get(i, pd.NaT)).days / 365.25 for i in d["espn_id"]]
        d["exp"] = y + 1 - d["espn_id"].map(dy)
        out.append(d)
    d = pd.concat(out, ignore_index=True)
    d["g1"] = d["g1"].fillna(0)
    d["tot1"] = d["tot1"].fillna(0)
    return d


def _ridge_oof(d: pd.DataFrame, X: list[str], y: str, alpha: float = 3.0) -> np.ndarray:
    from sklearn.linear_model import Ridge
    pred = np.full(len(d), np.nan)
    for s in sorted(d["season"].unique()):
        tr, te = d["season"] != s, d["season"] == s
        A = d.loc[tr, X]
        mu, sd = A.mean(), A.std().replace(0, 1)
        A = ((A - mu) / sd).fillna(0)
        B = ((d.loc[te, X] - mu) / sd).fillna(0)
        m = Ridge(alpha=alpha).fit(A, d.loc[tr, y])
        pred[te.to_numpy()] = m.predict(B)
    return pred


def _score(d, pred, y):
    from scipy.stats import spearmanr
    r = [spearmanr(pred[d["season"] == s], d.loc[d["season"] == s, y]).correlation for s in sorted(d["season"].unique())]
    return float(np.mean(np.abs(pred - d[y]))), np.array(r)


def nextseason() -> None:
    d = nextseason_rows()
    d["lecr"] = np.log(d["ecr"])
    base = ["lecr", "ppg0", "g0", "age", "exp"]
    print(f"rows {len(d)}  seasons {sorted(d.season.unique())}")
    for tgt, filt in (("ppg1", d["g1"] >= 4), ("tot1", d["g1"] >= 0)):
        print(f"\n== target {tgt}")
        for pos in POS:
            g = d[(d["pos"] == pos) & filt].reset_index(drop=True)
            if pos in ("WR", "TE"):
                g = g[g["routes"].fillna(0) >= 100].reset_index(drop=True)
            elif pos == "RB":
                g = g[g["attempts"].fillna(0) >= 30].reset_index(drop=True)
            else:
                g = g[g["dropbacks"].fillna(0) >= 100].reset_index(drop=True)
            P = PFF_SET[pos]
            m0, r0 = _score(g, _ridge_oof(g, ["lecr"], tgt), tgt)
            m1, r1 = _score(g, _ridge_oof(g, base, tgt), tgt)
            m2, r2 = _score(g, _ridge_oof(g, base + P, tgt, alpha=10.0), tgt)
            # placebo: shuffle PFF block across players within season
            pl = []
            for _ in range(200):
                h = g.copy()
                for s in h["season"].unique():
                    ix = h.index[h["season"] == s]
                    h.loc[ix, P] = h.loc[RNG.permutation(ix), P].to_numpy()
                pl.append(_score(h, _ridge_oof(h, base + P, tgt, alpha=10.0), tgt)[0])
            p = float(np.mean(np.array(pl) <= m2))
            print(f"{pos} n={len(g)}  MAE ecr {m0:.3f} base {m1:.3f} +pff {m2:.3f} (placebo p={p:.2f})  "
                  f"rho ecr {r0.mean():.3f} base {r1.mean():.3f} +pff {r2.mean():.3f}  seasons better {int((r2 > r1).sum())}/{len(r1)}")
            # single features
            res = []
            for c in P:
                mc, rc = _score(g, _ridge_oof(g, base + [c], tgt), tgt)
                res.append((c, m1 - mc, rc.mean() - r1.mean(), int((rc > r1).sum())))
            print("   single adds (MAE gain, rho gain, seasons):", ", ".join(f"{c} {a:+.3f}/{b:+.3f}/{k}" for c, a, b, k in sorted(res, key=lambda t: -t[1])[:5]))


# ---------------------------------------------------------------- E2: do grades predict the coach's next move?
def _win(y: int, lo: int, hi: int) -> pd.DataFrame:
    w = f"w{lo:02d}-{hi:02d}"
    r = num(wt("receiving_summary", y, w))
    u = num(wt("rushing_summary", y, w))
    p = num(wt("passing_summary", y, w))
    db = p.groupby("franchise_id")["dropbacks"].sum()
    ra = u.groupby("franchise_id")["attempts"].sum()
    r = r.assign(rt_share=r["routes"] / r["franchise_id"].map(db), g_route=r["grades_pass_route"], yprr_=r["yprr"],
                 tgt_rt=r["targets"] / r["routes"].replace(0, np.nan))
    u = u.assign(ru_share=u["attempts"] / u["franchise_id"].map(ra), g_run=u["grades_run"], ypc=u["ypa"])
    a = r[["player_id", "position", "franchise_id", "player_game_count", "rt_share", "g_route", "yprr_", "tgt_rt", "routes"]]
    b = u[["player_id", "ru_share", "g_run", "ypc", "attempts"]]
    return a.merge(b, on="player_id", how="outer")


def role_rows() -> pd.DataFrame:
    out = []
    for y in SEAS:
        for W in range(5, 11):
            cum, rec, fut = _win(y, 1, W - 1), _win(y, W - 3, W - 1), _win(y, W, W + 2)
            d = cum.merge(rec[["player_id", "rt_share", "ru_share"]].rename(columns={"rt_share": "rt_rec", "ru_share": "ru_rec"}), on="player_id", how="left")
            d = d.merge(fut[["player_id", "rt_share", "ru_share", "franchise_id"]].rename(columns={"rt_share": "rt_fut", "ru_share": "ru_fut", "franchise_id": "fr_fut"}), on="player_id", how="left")
            out.append(d.assign(season=y, week=W))
    d = pd.concat(out, ignore_index=True)
    d = d[d["franchise_id"] == d["fr_fut"].fillna(d["franchise_id"])]  # drop mid-window team changes
    for c in ("rt_fut", "ru_fut"):
        d[c] = d[c].fillna(0)  # no routes/carries in the next 3 weeks = share 0 (benched, hurt or cut)
    return d


def _block_slope(g: pd.DataFrame, x: str, y: str, ctrl: list[str], boots: int = 500, perms: int = 300):
    """Partial slope of y on x (per sd) after centring within season-week and residualising on ctrl.
    CI: player-season cluster bootstrap. Placebo: swap whole player-season x paths within season."""
    g = g.dropna(subset=[x, y] + ctrl).copy()
    by = [g["season"], g["week"]]
    cols = [x, y] + ctrl
    C = g[cols] - g[cols].groupby(by).transform("mean")
    Z = C[ctrl].to_numpy(float)

    def fit(xv, yv, Zm):
        A = np.column_stack([xv, Zm])
        return np.linalg.lstsq(A, yv, rcond=None)[0][0]

    sx = C[x].std()
    b = fit(C[x].to_numpy(float), C[y].to_numpy(float), Z) * sx
    key = (g["season"].astype(str) + "_" + g["player_id"].astype(str)).to_numpy()
    uk = np.unique(key)
    idx = {k: np.where(key == k)[0] for k in uk}
    X, Y = C[x].to_numpy(float), C[y].to_numpy(float)
    bs = []
    for _ in range(boots):
        ii = np.concatenate([idx[k] for k in RNG.choice(uk, len(uk))])
        bs.append(fit(X[ii], Y[ii], Z[ii]) * sx)
    # placebo: permute x among player-seasons (keep within-season structure), using each player's mean x
    pl = []
    gm = pd.Series(X).groupby(key).transform("mean").to_numpy()
    dev = X - gm
    seas = g["season"].to_numpy()
    for _ in range(perms):
        xp = X.copy()
        for s in np.unique(seas):
            ks = np.unique(key[seas == s])
            perm = dict(zip(ks, RNG.permutation(ks)))
            mk = {k: gm[idx[k][0]] for k in ks}
            m = seas == s
            xp[m] = np.array([mk[perm[k]] for k in key[m]])
        pl.append(fit(xp, Y, Z) * sx)
    p = float(np.mean(np.abs(pl) >= abs(b)))
    return b, np.percentile(bs, [2.5, 97.5]), p, len(g), len(uk)


def roles() -> None:
    d = role_rows()
    pe = pff_map()
    d["espn_id"] = d["player_id"].map(pe)
    print(f"rows {len(d)}")
    tests = [
        ("WR route share", d[(d["position"] == "WR") & (d["routes"] >= 30)], "g_route", "rt_fut", ["rt_share", "rt_rec"], ["yprr_", "tgt_rt"]),
        ("TE route share", d[(d["position"] == "TE") & (d["routes"] >= 30)], "g_route", "rt_fut", ["rt_share", "rt_rec"], ["yprr_", "tgt_rt"]),
        ("RB rush share", d[(d["position"] == "HB") & (d["attempts"] >= 10)], "g_run", "ru_fut", ["ru_share", "ru_rec"], ["ypc"]),
    ]
    for name, g, x, y, ctrl, free in tests:
        for lab, c in (("controls=share", ctrl), ("controls=share+free efficiency", ctrl + free)):
            b, ci, p, n, k = _block_slope(g, x, y, c)
            print(f"{name}: grade -> next-3-wk share, {lab}: {b * 100:+.2f} pts per sd [{ci[0] * 100:+.2f}, {ci[1] * 100:+.2f}] placebo p={p:.3f} n={n} ({k} player-seasons)")
        # backups: share between 10% and 50%
        lo = g[(g[ctrl[0]] > 0.1) & (g[ctrl[0]] < 0.5)]
        b, ci, p, n, k = _block_slope(lo, x, y, ctrl + free)
        print(f"   part-timers (share 10-50%): {b * 100:+.2f} [{ci[0] * 100:+.2f}, {ci[1] * 100:+.2f}] p={p:.3f} n={n}")
        for fx in free:
            b, ci, p, n, k = _block_slope(g, fx, y, ctrl)
            print(f"   free {fx} alone: {b * 100:+.2f} [{ci[0] * 100:+.2f}, {ci[1] * 100:+.2f}] p={p:.3f}")


# ---------------------------------------------------------------- E3: style-aware floor/ceiling
QS = (0.1, 0.5, 0.9)


def style_rows() -> pd.DataFrame:
    pe = pff_map()
    pts = league_points()
    out = []
    for y in SEAS:
        pr = pd.read_csv(ROOT / "history" / "actuals" / str(y) / "espn_weekly_proj.csv")
        a = pts[(pts["season"] == y) & pts["played"].astype(bool)].merge(pr, on=["week", "espn_id"])
        a = a[a["espn_proj"] >= 5]
        # free history: season-to-date sd of (points - proj), TD-dependence not available weekly -> sd + mean residual
        a = a.sort_values("week")
        a["res"] = a["points"] - a["espn_proj"]
        g = a.groupby("espn_id")["res"]
        a["res_sd_std"] = g.transform(lambda s: s.shift().expanding(3).std())
        a["res_mean_std"] = g.transform(lambda s: s.shift().expanding(3).mean())
        for W in range(3, 14):
            w = f"w01-{W - 1:02d}"
            r = num(wt("receiving_summary", y, w))
            u = num(wt("rushing_summary", y, w))
            p = num(wt("passing_summary", y, w))
            db = p.groupby("franchise_id")["dropbacks"].sum()
            ra = u.groupby("franchise_id")["attempts"].sum()
            rf = pd.DataFrame({"pff_id": r["player_id"], "adot": r["avg_depth_of_target"], "slot": r["slot_rate"],
                               "rt_share": r["routes"] / r["franchise_id"].map(db), "tgt_rt": r["targets"] / r["routes"].replace(0, np.nan),
                               "yprr": r["yprr"], "contested": r["contested_targets"] / r["targets"].replace(0, np.nan),
                               "yac_rec": r["yards_after_catch_per_reception"]})
            uf = pd.DataFrame({"pff_id": u["player_id"], "ru_share": u["attempts"] / u["franchise_id"].map(ra),
                               "breakaway": u["breakaway_percent"], "elusive": u["elusive_rating"],
                               "qb_runs_g": (u["attempts"] - u["scrambles"].fillna(0)) / u["player_game_count"]})
            pf = pd.DataFrame({"pff_id": p["player_id"], "scr_db": p["scrambles"] / p["dropbacks"].replace(0, np.nan),
                               "p_adot": p["avg_depth_of_target"], "btt": p["btt_rate"], "twp": p["twp_rate"], "sack_pct": p["sack_percent"]})
            f = rf.merge(uf, on="pff_id", how="outer").merge(pf, on="pff_id", how="outer").drop_duplicates("pff_id")
            f["espn_id"] = f["pff_id"].map(pe)
            f = f.dropna(subset=["espn_id"]).astype({"espn_id": int}).drop(columns="pff_id")
            out.append(a[a["week"] == W].merge(f, on="espn_id", how="left").assign(season=y))
    return pd.concat(out, ignore_index=True)


STYLE = {
    "WR": ["adot", "slot", "rt_share", "tgt_rt", "yprr", "contested", "yac_rec"],
    "TE": ["adot", "slot", "rt_share", "tgt_rt", "yprr", "yac_rec"],
    "RB": ["ru_share", "rt_share", "tgt_rt", "breakaway", "elusive"],
    "QB": ["scr_db", "qb_runs_g", "p_adot", "btt", "twp", "sack_pct"],
}


def _pinball(y, q, tau):
    e = y - q
    return np.mean(np.maximum(tau * e, (tau - 1) * e))


def ranges() -> None:
    from sklearn.ensemble import HistGradientBoostingRegressor as H
    d = style_rows()
    print(f"rows {len(d)}")
    for pos in POS:
        g = d[d["position"] == pos].reset_index(drop=True)
        sets = {"proj only": ["espn_proj"], "proj+own history": ["espn_proj", "res_sd_std", "res_mean_std"],
                "proj+PFF style": ["espn_proj"] + STYLE[pos], "proj+history+PFF": ["espn_proj", "res_sd_std", "res_mean_std"] + STYLE[pos]}
        res = {}
        for name, X in sets.items():
            loss, cov, width = [], [], []
            per = []
            for s in SEAS:
                tr, te = g[g["season"] != s], g[g["season"] == s]
                preds = {}
                for tau in QS:
                    m = H(loss="quantile", quantile=tau, max_iter=200, learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=80)
                    m.fit(tr[X], tr["points"])
                    preds[tau] = m.predict(te[X])
                yv = te["points"].to_numpy()
                l = np.mean([_pinball(yv, preds[t], t) for t in QS])
                per.append(l)
                loss.append(l * len(te))
                cov.append(((yv >= preds[0.1]) & (yv <= preds[0.9])).sum())
                width.append((preds[0.9] - preds[0.1]).sum())
            n = len(g)
            res[name] = (sum(loss) / n, sum(cov) / n, sum(width) / n, np.array(per))
        b = res["proj only"]
        print(f"\n{pos} n={len(g)}")
        for name, (l, c, w, per) in res.items():
            print(f"  {name:20s} pinball {l:.4f} ({(l / b[0] - 1) * 100:+.2f}%)  80% cover {c:.3f}  width {w:.2f}  seasons better {int((per < b[3]).sum())}/5")


# ---------------------------------------------------------------- E4: contact load -> missed games
INJ = Path(os.environ.get("NFL_INJ", Path.home() / "nfl_dl"))  # nflverse injuries_<season>.csv (releases/download/injuries)


def _full(y: int, lo: int, hi: int) -> pd.DataFrame:
    w = f"w{lo:02d}-{hi:02d}"
    r = num(wt("receiving_summary", y, w))
    u = num(wt("rushing_summary", y, w))
    a = r[["player_id", "position", "franchise_id", "player_game_count", "routes", "receptions", "contested_targets", "avoided_tackles", "yards_after_catch", "pass_blocks"]]
    b = u[["player_id", "attempts", "yards_after_contact", "avoided_tackles", "player_game_count", "franchise_id"]].rename(
        columns={"avoided_tackles": "ru_avoided", "player_game_count": "ru_gc", "franchise_id": "ru_fr"})
    d = a.merge(b, on="player_id", how="outer")
    d["franchise_id"] = d["franchise_id"].fillna(d["ru_fr"])
    d["player_game_count"] = d[["player_game_count", "ru_gc"]].max(axis=1)
    d = d.fillna({c: 0 for c in ("routes", "receptions", "contested_targets", "avoided_tackles", "yards_after_catch", "pass_blocks", "attempts", "yards_after_contact", "ru_avoided")})
    d["team_g"] = d.groupby("franchise_id")["player_game_count"].transform("max")
    return d


def injury_rows() -> pd.DataFrame:
    idd = ids().dropna(subset=["pff_id"])
    gs = dict(zip(idd["pff_id"].astype(int), idd["gsis_id"]))
    bd = dict(zip(idd["pff_id"].astype(int), pd.to_datetime(idd["birthdate"], errors="coerce")))
    out = []
    for y in SEAS:
        inj = pd.read_csv(INJ / f"injuries_{y}.csv")
        inj = inj[inj["game_type"] == "REG"]
        q = inj.assign(flag=inj["report_status"].isin(["Questionable", "Doubtful", "Out"]) | inj["practice_status"].fillna("").str.contains("Did Not|Limited"))
        for W in range(5, 11):
            cum, fut = _full(y, 1, W - 1), _full(y, W, W + 2)
            d = cum.merge(fut[["player_id", "player_game_count", "team_g"]].rename(columns={"player_game_count": "g_fut", "team_g": "tg_fut"}), on="player_id", how="left")
            d["tg_fut"] = d["tg_fut"].fillna(d["franchise_id"].map(fut.groupby("franchise_id")["team_g"].max()))
            d["g_fut"] = d["g_fut"].fillna(0)
            d["miss"] = (d["g_fut"] < d["tg_fut"]).astype(int)
            gp = d["player_game_count"].replace(0, np.nan)
            d["touch_g"] = (d["attempts"] + d["receptions"]) / gp
            d["routes_g"] = d["routes"] / gp
            d["missed_sofar"] = d["team_g"] - d["player_game_count"]
            d["yco_g"] = d["yards_after_contact"] / gp
            d["avoid_g"] = (d["avoided_tackles"] + d["ru_avoided"]) / gp
            d["contested_g"] = d["contested_targets"] / gp
            d["yac_g"] = d["yards_after_catch"] / gp
            d["pblock_g"] = d["pass_blocks"] / gp
            d["gsis_id"] = d["player_id"].map(gs)
            d["age"] = [(pd.Timestamp(f"{y}-10-01") - bd.get(int(i), pd.NaT)).days / 365.25 for i in d["player_id"]]
            prev = q[q["week"] == W - 1].groupby("gsis_id")["flag"].max()
            d["flag_prev"] = d["gsis_id"].map(prev).fillna(False).astype(int)
            d["flags_sofar"] = d["gsis_id"].map(q[q["week"] < W].groupby("gsis_id")["flag"].sum()).fillna(0)
            out.append(d.assign(season=y, week=W))
    d = pd.concat(out, ignore_index=True)
    return d


def injury() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score
    d = injury_rows()
    d = d[d["player_game_count"] >= d["team_g"] - 1]  # mostly-healthy so far
    base = ["touch_g", "routes_g", "missed_sofar", "age", "flag_prev", "flags_sofar"]
    pff = ["yco_g", "avoid_g", "contested_g", "yac_g", "pblock_g"]
    for pos, keep in (("HB", d["touch_g"] >= 10), ("WR", d["routes_g"] >= 20), ("TE", d["routes_g"] >= 15)):
        g = d[(d["position"] == pos) & keep].dropna(subset=["age"]).reset_index(drop=True)
        res = {}
        for name, X in (("base", base), ("base+PFF contact", base + pff)):
            p = np.zeros(len(g))
            for s in SEAS:
                tr, te = g["season"] != s, g["season"] == s
                mu, sd = g.loc[tr, X].mean(), g.loc[tr, X].std()
                m = LogisticRegression(C=0.5, max_iter=500).fit(((g.loc[tr, X] - mu) / sd).fillna(0), g.loc[tr, "miss"])
                p[te.to_numpy()] = m.predict_proba(((g.loc[te, X] - mu) / sd).fillna(0))[:, 1]
            res[name] = p
        y = g["miss"].to_numpy()
        key = (g["season"].astype(str) + g["player_id"].astype(str)).to_numpy()
        uk = np.unique(key)
        ix = {k: np.where(key == k)[0] for k in uk}
        dl = []
        for _ in range(300):
            ii = np.concatenate([ix[k] for k in RNG.choice(uk, len(uk))])
            dl.append(log_loss(y[ii], res["base"][ii]) - log_loss(y[ii], res["base+PFF contact"][ii]))
        lo, hi = np.percentile(dl, [2.5, 97.5])
        print(f"{pos} n={len(g)} miss rate {y.mean():.3f}  AUC base {roc_auc_score(y, res['base']):.3f} +PFF {roc_auc_score(y, res['base+PFF contact']):.3f}  "
              f"logloss gain {(log_loss(y, res['base']) - log_loss(y, res['base+PFF contact'])) * 1000:+.2f}e-3 [{lo * 1000:+.2f}, {hi * 1000:+.2f}]")


# ---------------------------------------------------------------- E6: waiver picks with PFF usage
def waiver_rows() -> pd.DataFrame:
    pe = pff_map()
    d = pd.concat([pd.read_csv(ROOT / "history" / "backtest" / "multi" / f"usage_rows_{y}.csv") for y in SEAS], ignore_index=True)
    d = d[d["position"].isin(["RB", "WR", "TE"]) & ~d["rostered"].astype(bool) & d["week"].between(3, 13)]
    out = []
    for y in SEAS:
        for W in range(3, 14):
            f = _win(y, 1, W - 1)
            r = _win(y, max(1, W - 3), W - 1) if W >= 4 else f
            f = f.assign(rt_rec=f["player_id"].map(r.drop_duplicates("player_id").set_index("player_id")["rt_share"]))
            f["espn_id"] = f["player_id"].map(pe)
            f = f.dropna(subset=["espn_id"]).astype({"espn_id": int}).drop_duplicates("espn_id")
            out.append(f[["espn_id", "rt_share", "rt_rec", "tgt_rt", "ru_share", "yprr_"]].assign(season=y, week=W))
    f = pd.concat(out, ignore_index=True)
    d = d.merge(f, on=["espn_id", "season", "week"], how="left")
    d["future_ppg"] = d["future_ppg"].fillna(0)
    d["ea"] = d["est_ppg"] * d["avail"]
    d["fpw"] = d["future_points"] / (18 - d["week"])  # ROS points per remaining week (availability included)
    return d


def waivers() -> None:
    from sklearn.linear_model import Ridge
    d = waiver_rows()
    free = ["ea", "avail", "est_ppg", "espn_now", "snap_l2", "tshare_l3", "opp_l3", "xfp_l3"]
    pff = ["rt_share", "rt_rec", "tgt_rt", "ru_share", "yprr_"]
    d = d.dropna(subset=["est_ppg"])
    print(f"unrostered rows {len(d)}")
    for pos in ("RB", "WR", "TE"):
        g = d[d["position"] == pos].reset_index(drop=True)
        preds = {}
        for name, X in (("est_ppg x avail", ["ea"]), ("free usage", free), ("free+PFF", free + pff)):
            p = np.zeros(len(g))
            for s in SEAS:
                tr, te = g["season"] != s, g["season"] == s
                mu, sd = g.loc[tr, X].mean(), g.loc[tr, X].std().replace(0, 1)
                m = Ridge(alpha=5).fit(((g.loc[tr, X] - mu) / sd).fillna(0), g.loc[tr, "fpw"])
                p[te.to_numpy()] = m.predict(((g.loc[te, X] - mu) / sd).fillna(0))
            preds[name] = p
        line = []
        k = 3
        hit = g["fpw"] >= STARTER_PPG[pos] * 0.75
        for name, p in preds.items():
            g["_p"] = p
            top = g.sort_values("_p", ascending=False).groupby(["season", "week"]).head(k)
            line.append((name, top["fpw"].mean(), hit[top.index].mean(), top.groupby("season")["fpw"].mean()))
        b = line[1]
        # season-week bootstrap of free+PFF minus free picks
        g["_a"], g["_b"] = preds["free usage"], preds["free+PFF"]
        ta = g.sort_values("_a", ascending=False).groupby(["season", "week"]).head(k).groupby(["season", "week"])["fpw"].mean()
        tb = g.sort_values("_b", ascending=False).groupby(["season", "week"]).head(k).groupby(["season", "week"])["fpw"].mean()
        diff = (tb - ta).to_numpy()
        bs = [RNG.choice(diff, len(diff)).mean() for _ in range(2000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"{pos}: top-{k} free agents per week, ROS pts/wk / hit rate (>=0.75x starter ppg): " + "; ".join(f"{n} {m:.2f}/{h:.2f}" for n, m, h, _ in line)
              + f"  | +PFF vs free {diff.mean():+.2f} pts/wk [{lo:+.2f}, {hi:+.2f}], seasons better {int((line[2][3] > b[3]).sum())}/5")


STARTER_PPG = {"RB": 8.0, "WR": 8.0, "TE": 6.0}


if __name__ == "__main__":
    {"nextseason": nextseason, "roles": roles, "ranges": ranges, "injury": injury, "waivers": waivers}[sys.argv[1]]()
