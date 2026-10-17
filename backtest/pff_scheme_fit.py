"""Do PFF scheme splits x the opponent's scheme predict anything beyond ESPN's weekly projection?  (2021-25, weeks 4-13)

Features, fixed before looking at results (higher = better for the offense player; all inputs from weeks before W):
  man_fit   WR/TE: (man YPRR - zone YPRR) / overall YPRR, each split shrunk toward his overall YPRR (K_ROUTES routes),
            prior season at half weight + season to date (receiving/scheme)  x  opponent man coverage % minus the
            league mean (v2 teams/stats defense-passing, weeks 1..W-1)
  press_fit QB: (fantasy pts per dropback under pressure - not under pressure) / overall, shrunk (K_DB dropbacks),
            same windows (passing/pressure), minus the league mean  x  opponent pressure rate minus league mean
  blitz_fit QB: the same with blitz / no blitz and the opponent's blitz rate
  adot_fit  WR/TE/RB: his aDOT minus the position mean (receiving/summary, season to date)  x  opponent aDOT
            allowed minus league mean (v2 defense-opponent-tendencies)
  yac_fit   WR/TE/RB: his YAC share of receiving yards minus position mean  x  opponent YAC % allowed minus league mean
Each fit is tested with both of its main effects as controls. Rows, target and stats as backtest/matchup_pff.py:
Desk's rostered player-weeks with an ESPN projection, resid = actual - ESPN projection, effect as a share of the
projection per sd (matchup_mechanisms.ols_prop), season-week cluster bootstrap CI, shuffled-opponent placebo, plus
leave-one-season-out: fit on four seasons, score the fifth, report the change in resid MSE.

Result (2026-10-17, 2021-25 wk 4-13, ~6.4k rostered player-weeks): nothing clears noise. Every 95% CI crosses zero
(largest: WR man_fit -1.7% of proj per sd [-4.2, +0.8], placebo p=0.14, wrong sign); no placebo p below 0.14; LOSO
never beats the controls-only fit by more than 0.1% of MSE. The player splits barely persist year to year (season r:
WR/TE man-zone YPRR 0.22, QB pressure split 0.01, QB blitz split 0.10), which is why the interactions carry nothing.

PFF data is cached raw under the main checkout's .cache/pff/raw (gitignored, never committed).
  python3 backtest/pff_scheme_fit.py fetch   # ~210 API calls, cached
  python3 backtest/pff_scheme_fit.py         # prints results
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backtest"))
warnings.filterwarnings("ignore")
import pff_api as api  # noqa: E402
from pff_probe import rows_of  # noqa: E402
from backtest import matchup_mechanisms as mm  # noqa: E402
from backtest import matchup_pff as mp  # noqa: E402
from backtest import vegas_signal as vs  # noqa: E402

mp.CACHE = api.MAIN / ".cache" / "pff"
SEAS = [2021, 2022, 2023, 2024, 2025]
WEEKS = range(4, 14)
K_ROUTES, K_DB = 60.0, 30.0
STARTER = mp.STARTER
FIX = {"LAR": "LA", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "OAK": "LV", "SD": "LAC", "WSH": "WAS", "JAC": "JAX"}
TESTS = {"WR": [["man_fit", "opp_man", "man_rel"], ["adot_fit", "opp_adot", "adot_dev"], ["yac_fit", "opp_yac", "yac_dev"]],
         "TE": [["man_fit", "opp_man", "man_rel"], ["adot_fit", "opp_adot", "adot_dev"], ["yac_fit", "opp_yac", "yac_dev"]],
         "RB": [["adot_fit", "opp_adot", "adot_dev"], ["yac_fit", "opp_yac", "yac_dev"]],
         "QB": [["press_fit", "opp_press", "press_rel"], ["blitz_fit", "opp_blitz", "blitz_rel"]]}
OPP = ["opp_man", "opp_press", "opp_blitz", "opp_adot", "opp_yac"]


def wk(lo: int, hi: int) -> str:
    return ",".join(str(w) for w in range(lo, hi + 1))


def calls():
    for y in SEAS:
        last = 17 if y - 1 == 2020 else 18
        for f in ("receiving/scheme", "passing/pressure"):
            yield f"/v1/facet/{f}", {"league": "nfl", "season": y - 1, "week": wk(1, last)}
        for W in WEEKS:
            for c in ("defense-passing", "defense-opponent-tendencies"):
                yield "/v2/nfl/teams/stats", {"season": y, "weekIds": wk(1, W - 1), "category": c}
            for f in ("receiving/scheme", "passing/pressure"):
                yield f"/v1/facet/{f}", {"league": "nfl", "season": y, "week": wk(1, W - 1)}


def fetch():
    for i, (p, q) in enumerate(calls()):
        r = api.get(p, q)
        if r["status"] != 200:
            print("FAIL", r["status"], p, q)
        if i % 20 == 0:
            print(i, p, q.get("season"), r["ratelimit"].get("x-ratelimit-remaining"), flush=True)


def df(p: str, q: dict) -> pd.DataFrame:
    return pd.DataFrame(rows_of(api.get(p, q)["body"]))


def num(d: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0) if c in d else 0.0
    return d


# ---------- opponent side ----------
def defense() -> pd.DataFrame:
    out = []
    for y in SEAS:
        for W in WEEKS:
            a = df("/v2/nfl/teams/stats", {"season": y, "weekIds": wk(1, W - 1), "category": "defense-passing"})
            b = df("/v2/nfl/teams/stats", {"season": y, "weekIds": wk(1, W - 1), "category": "defense-opponent-tendencies"})
            d = a[["abbreviation", "manCoveragePct", "pressureRate", "blitzRate", "yacPctAllowed"]].merge(b[["abbreviation", "adotAgainst"]], on="abbreviation")
            d.columns = ["opp", "opp_man", "opp_press", "opp_blitz", "opp_yac", "opp_adot"]
            d["opp"] = d["opp"].replace(FIX)
            for c in OPP:
                d[c] = pd.to_numeric(d[c], errors="coerce")
                d[c] = d[c] - d[c].mean()
            out.append(d.assign(season=y, week=W))
    return pd.concat(out, ignore_index=True)


# ---------- player side ----------
def _blend(cur: pd.DataFrame, prev: pd.DataFrame, cols) -> pd.DataFrame:
    a = cur.set_index("player_id")[cols]
    b = prev.set_index("player_id")[cols].reindex(a.index).fillna(0.0) * 0.5
    return (a + b).reset_index()


def receivers(pe: dict) -> pd.DataFrame:
    cols = ["routes", "yards", "man_routes", "man_yards", "zone_routes", "zone_yards"]
    out = []
    for y in SEAS:
        last = 17 if y - 1 == 2020 else 18
        prev = num(df("/v1/facet/receiving/scheme", {"league": "nfl", "season": y - 1, "week": wk(1, last)}), cols + ["player_id"])
        prev["routes"], prev["yards"] = prev["man_routes"] + prev["zone_routes"], prev["man_yards"] + prev["zone_yards"]
        for W in WEEKS:
            cur = num(df("/v1/facet/receiving/scheme", {"league": "nfl", "season": y, "week": wk(1, W - 1)}), cols + ["player_id"])
            cur["routes"], cur["yards"] = cur["man_routes"] + cur["zone_routes"], cur["man_yards"] + cur["zone_yards"]
            d = _blend(cur, prev, cols)
            d = d[d["routes"] >= 40]
            yp = d["yards"] / d["routes"]
            man = (d["man_yards"] + K_ROUTES * yp) / (d["man_routes"] + K_ROUTES)
            zone = (d["zone_yards"] + K_ROUTES * yp) / (d["zone_routes"] + K_ROUTES)
            rel = ((man - zone) / yp.clip(lower=0.3))
            # aDOT / YAC share from the season-to-date receiving summary already cached for earlier backtests
            p = mp.CACHE / "receiving_summary" / str(y) / f"w01-{W - 1:02d}.json"
            rs = num(pd.DataFrame(mp._rows(p)), ["player_id", "avg_depth_of_target", "yards_after_catch", "yards", "targets"]) if p.exists() else None
            e = pd.DataFrame({"player_id": d["player_id"].astype(int), "man_rel": rel.to_numpy()})
            if rs is not None:
                rs = rs[rs["targets"] >= 8]
                rs = pd.DataFrame({"player_id": rs["player_id"].astype(int), "adot": rs["avg_depth_of_target"],
                                   "yac_share": rs["yards_after_catch"] / rs["yards"].clip(lower=1)})
                e = e.merge(rs, on="player_id", how="outer")
            e["espn_id"] = e["player_id"].map(pe)
            out.append(e.dropna(subset=["espn_id"]).assign(season=y, week=W))
    R = pd.concat(out, ignore_index=True)
    R["espn_id"] = R["espn_id"].astype(int)
    return R.drop_duplicates(["espn_id", "season", "week"])


def _fppdb(d: pd.DataFrame, p: str) -> tuple:
    return 0.04 * d[f"{p}yards"] + 4 * d[f"{p}touchdowns"] - 2 * d[f"{p}interceptions"], d[f"{p}dropbacks"]


def qbs(pe: dict) -> pd.DataFrame:
    sp = ["pressure_", "no_pressure_", "blitz_", "no_blitz_"]
    cols = [f"{s}{c}" for s in sp for c in ("yards", "touchdowns", "interceptions", "dropbacks")]
    out = []
    for y in SEAS:
        last = 17 if y - 1 == 2020 else 18
        prev = num(df("/v1/facet/passing/pressure", {"league": "nfl", "season": y - 1, "week": wk(1, last)}), cols + ["player_id"])
        for W in WEEKS:
            cur = num(df("/v1/facet/passing/pressure", {"league": "nfl", "season": y, "week": wk(1, W - 1)}), cols + ["player_id"])
            d = _blend(cur, prev, cols)
            e = {"player_id": d["player_id"].astype(int)}
            for a, b, name in (("pressure_", "no_pressure_", "press_rel"), ("blitz_", "no_blitz_", "blitz_rel")):
                pa, na = _fppdb(d, a)
                pb, nb = _fppdb(d, b)
                ov = (pa + pb) / (na + nb).clip(lower=1)
                ra = (pa + K_DB * ov) / (na + K_DB)
                rb = (pb + K_DB * ov) / (nb + K_DB)
                e[name] = ((ra - rb) / ov.clip(lower=0.05)).where((na + nb) >= 60)
            e = pd.DataFrame(e)
            e["espn_id"] = e["player_id"].map(pe)
            out.append(e.dropna(subset=["espn_id"]).assign(season=y, week=W))
    Q = pd.concat(out, ignore_index=True)
    Q["espn_id"] = Q["espn_id"].astype(int)
    return Q.drop_duplicates(["espn_id", "season", "week"])


def build() -> pd.DataFrame:
    pe = mp.pff_espn()
    raw = vs.rows()
    X = raw[raw["position"].isin(TESTS) & raw["season"].isin(SEAS) & raw["week"].isin(list(WEEKS))].copy()
    X["resid"] = X["points"] - X["espn_proj"]
    X = X.merge(receivers(pe).drop(columns="player_id"), on=["espn_id", "season", "week"], how="left")
    X = X.merge(qbs(pe).drop(columns="player_id"), on=["espn_id", "season", "week"], how="left")
    for c in ("man_rel", "press_rel", "blitz_rel", "adot", "yac_share"):
        X[c] = X[c] - X.groupby(["season", "position"])[c].transform("mean")
    X = X.rename(columns={"adot": "adot_dev", "yac_share": "yac_dev"})
    X = X.merge(defense(), on=["opp", "season", "week"], how="left")
    return derive(X)


def derive(X: pd.DataFrame) -> pd.DataFrame:
    X["man_fit"] = X["man_rel"] * X["opp_man"]
    X["press_fit"] = X["press_rel"] * X["opp_press"]
    X["blitz_fit"] = X["blitz_rel"] * X["opp_blitz"]
    X["adot_fit"] = X["adot_dev"] * X["opp_adot"]
    X["yac_fit"] = X["yac_dev"] * X["opp_yac"]
    return X


def shuffled(X: pd.DataFrame, rng) -> pd.DataFrame:
    """Placebo: every row gets a random other defense from the same season-week (its own player features kept)."""
    D = X.dropna(subset=["opp"]).drop_duplicates(["season", "week", "opp"]).set_index(["season", "week", "opp"])[OPP]
    m = {}
    for (s, w), g in D.groupby(level=[0, 1]):
        o = g.index.get_level_values(2).to_numpy()
        for a, b in zip(o, rng.permutation(o)):
            m[(s, w, a)] = (s, w, b)
    k = [m.get(t, t) for t in zip(X["season"], X["week"], X["opp"])]
    Y = X.copy()
    Y[OPP] = D.reindex(k)[OPP].to_numpy()
    return derive(Y)


def _design(g: pd.DataFrame, xc, ycol="resid"):
    by = [g["season"], g["week"], g["position"]]
    Y = (g[ycol] - g.groupby(by)[ycol].transform("mean")).to_numpy(float)
    Z = g[xc].mul(g["espn_proj"], axis=0)
    return (Z - Z.groupby(by).transform("mean")).to_numpy(float), Y


def loso(g: pd.DataFrame, xc) -> tuple:
    """Fit on four seasons, score the held-out one: % change in centred resid MSE (negative = better), per season."""
    per = {}
    se = sb = 0.0
    for s in SEAS:
        tr, te = g[g["season"] != s], g[g["season"] == s]
        if len(te) < 50:
            continue
        b = mm.ols_prop(tr.reset_index(drop=True), "resid", xc)
        X, Y = _design(te.reset_index(drop=True), xc)
        e1, e0 = float(((Y - X @ b) ** 2).sum()), float((Y ** 2).sum())
        per[s] = 100 * (e1 / e0 - 1)
        se, sb = se + e1, sb + e0
    return 100 * (se / sb - 1), per


def run(X: pd.DataFrame, boots: int = 1000, perms: int = 200) -> list:
    rng = np.random.default_rng(0)
    P = [shuffled(X, rng) for _ in range(perms)]
    res = []
    for pos, sets in TESTS.items():
        for xc in sets:
            c = xc[0]
            for sub in ("rostered", "starters"):
                def pick(D):
                    m = D["position"] == pos
                    if sub == "starters":
                        m &= D["espn_proj"] >= STARTER[pos]
                    return D[m].dropna(subset=xc + ["resid"]).reset_index(drop=True)
                g = pick(X)
                if len(g) < 200:
                    continue
                sd = float(g.groupby(["season", "week"])[c].std().mean())
                b, lo, hi = mm.fit_prop(g, "resid", xc, boots)
                null = np.abs(np.array([mm.ols_prop(pick(Y), "resid", xc)[0] for Y in P]))
                lo_mse, per = loso(g, xc)
                lo_c, _ = loso(g, xc[1:])  # controls only, to isolate the fit term's out-of-sample value
                t = {"pos": pos, "rows": sub, "feature": c, "n": len(g), "pct_per_sd": 100 * b[0] * sd,
                     "ci": (100 * lo[0] * sd, 100 * hi[0] * sd), "perm_p": float((null >= abs(b[0])).mean()),
                     "loso_mse_pct": lo_mse, "loso_controls_only": lo_c, "loso_by_season": per}
                res.append(t)
                print(f"{pos:2} {sub:8} {c:9} n={len(g):5} {t['pct_per_sd']:+5.1f}% of proj per sd [{t['ci'][0]:+5.1f},{t['ci'][1]:+5.1f}] "
                      f"placebo p={t['perm_p']:.2f}  LOSO dMSE {lo_mse:+.2f}% (controls only {lo_c:+.2f}%)  "
                      + " ".join(f"{s % 100}:{v:+.2f}" for s, v in per.items()), flush=True)
    return res


def main() -> None:
    if sys.argv[1:] == ["fetch"]:
        return fetch()
    X = build()
    cov = {c: int(X[c].notna().sum()) for c in ("man_fit", "press_fit", "blitz_fit", "adot_fit", "yac_fit")}
    print("rows", len(X), "feature coverage", cov)
    run(X)


if __name__ == "__main__":
    main()
