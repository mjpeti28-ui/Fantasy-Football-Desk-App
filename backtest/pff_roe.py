"""Is PFF receptions-over-expected (and QB accuracy/completion over expected) luck or skill for rest-of-season points?

Question: does a player's ROE over weeks 1..W-1 predict future_ppg - forecast over weeks W..17, where the forecast
is the v2 model's est_ppg (params fit without that season) or ESPN's ROS per-game number (espn_now)? Positive slope =
buy (skill the forecast misses); negative = sell (luck that regresses harder than the forecast assumes).

Features (v2 positions/reports, weeks 1..W-1, regular season), fixed before looking:
  roe      WR/TE/RB: receptionsOe (catch rate over expected, pct points), shrunk toward 0 by targets/(targets+K_TGT)
  qb_coe   WR/TE/RB: attempt-weighted completionOe of his team's passers, shrunk by attempts/(attempts+K_ATT)
  coe, aoe QB: his completionOe / accuracyOe, shrunk the same way
Rows: backtest usage_rows_* checkpoints W=4..13 (history/backtest/multi), rostered going into W or with est_ppg at
starter level, future_played >= 3. Effect in ROS ppg per sd, centred within season-week-position, weighted by games
played, 95% CI from a player-season cluster bootstrap, placebo = each player-season gets another player-season's feature path (same season, position),
leave-one-season-out change in weighted MSE. Stability: half-to-half (wk 1-9 vs 10-18) and year-to-year r.

Result (2026-10-17, 2021-25, W=4..13, ~8.7k rostered/starter checkpoint rows): nothing clears noise. Every 95% CI
crosses zero, every block-placebo p >= 0.23, LOSO never cuts MSE by more than 0.5% and is positive (worse) in 12 of 16
tests. Point estimates: WR ROE -0.13 ROS ppg per sd [-0.44,+0.17] (faint "sell"), TE ROE +0.25 [-0.17,+0.65] (faint
"buy"), RB ROE +0.09, QB completionOe -0.13, accuracyOe +0.13. Stability: receptionsOe half-to-half r 0.05,
year-to-year 0.20; completionOe 0.33 / 0.31; accuracyOe 0.39 / 0.36. A row-level shuffle placebo gave p=0.004 for TE
ROE; that was an artefact of ignoring repeated checkpoints, hence the player-season block placebo.

  python3 backtest/pff_roe.py fetch   # ~130 API calls, cached in the main checkout's .cache/pff/raw (gitignored)
  python3 backtest/pff_roe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backtest"))
import pff_api as api  # noqa: E402
from pff_probe import rows_of  # noqa: E402
from backtest import matchup_pff as mp  # noqa: E402

mp.CACHE = api.MAIN / ".cache" / "pff"
SEAS = [2021, 2022, 2023, 2024, 2025]
CHECKS = range(4, 14)
K_TGT, K_ATT = 40.0, 150.0
STARTER = mp.STARTER
REP = "/v2/nfl/positions/reports/"


def q(y: int, lo: int, hi: int) -> dict:
    return {"season": y, "weekGroup": "REG", "week": lo, "weekTo": hi}


def calls():
    for y in SEAS:
        for lo, hi in ((1, 18), (1, 9), (10, 18)):
            for r in ("receiving", "passing"):
                yield REP + r, q(y, lo, hi)
        for W in CHECKS:
            for r in ("receiving", "passing"):
                yield REP + r, q(y, 1, W - 1)


def fetch():
    for i, (p, pr) in enumerate(calls()):
        r = api.get(p, pr)
        if r["status"] != 200:
            print("FAIL", r["status"], p, pr)
        if i % 20 == 0:
            print(i, p, pr, r["ratelimit"].get("x-ratelimit-remaining"), flush=True)


def tab(rep: str, y: int, lo: int, hi: int) -> pd.DataFrame:
    d = pd.DataFrame(rows_of(api.get(REP + rep, q(y, lo, hi))["body"]))
    for c in ("playerId", "franchiseId", "targets", "attempts", "receptionsOe", "completionOe", "accuracyOe"):
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def features(pe: dict) -> pd.DataFrame:
    out = []
    for y in SEAS:
        for W in CHECKS:
            r, p = tab("receiving", y, 1, W - 1), tab("passing", y, 1, W - 1)
            r = r.dropna(subset=["receptionsOe"])
            r = r.assign(roe=r["receptionsOe"] * r["targets"] / (r["targets"] + K_TGT))
            p = p[p["attempts"] > 0].dropna(subset=["completionOe"])
            tm = p.assign(w=p["attempts"] * p["completionOe"]).groupby("franchiseId")[["w", "attempts"]].sum()
            r["qb_coe"] = r["franchiseId"].map(tm["w"] / tm["attempts"] * tm["attempts"] / (tm["attempts"] + K_ATT))
            s = p["attempts"] / (p["attempts"] + K_ATT)
            p = p.assign(coe=p["completionOe"] * s, aoe=p["accuracyOe"] * s)
            f = pd.concat([r[["playerId", "roe", "qb_coe"]], p[["playerId", "coe", "aoe"]]]).groupby("playerId").first().reset_index()
            f["espn_id"] = f["playerId"].map(pe)
            out.append(f.dropna(subset=["espn_id"]).assign(season=y, week=W))
    F = pd.concat(out, ignore_index=True)
    F["espn_id"] = F["espn_id"].astype(int)
    return F.drop(columns="playerId").drop_duplicates(["espn_id", "season", "week"])


def rows() -> pd.DataFrame:
    d = pd.concat([pd.read_csv(f) for f in sorted((ROOT / "history" / "backtest" / "multi").glob("usage_rows_*.csv"))], ignore_index=True)
    d = d[d["position"].isin(["QB", "RB", "WR", "TE"]) & d["season"].isin(SEAS) & d["week"].isin(list(CHECKS))]
    d = d[(d["future_played"] >= 3) & d["future_ppg"].notna()]
    start = d["est_ppg"] >= d["position"].map(STARTER)
    d = d[d["rostered"].astype(bool) | start].copy()
    d["r_v2"] = d["future_ppg"] - d["est_ppg"]
    d["r_espn"] = d["future_ppg"] - d["espn_now"]
    d.loc[d["espn_now"].fillna(0) <= 0, "r_espn"] = np.nan
    return d


def _c(g: pd.DataFrame, cols) -> np.ndarray:
    by = [g["season"], g["week"], g["position"]]
    return (g[cols] - g[cols].groupby(by).transform("mean")).to_numpy(float)


def slope(g: pd.DataFrame, x: str, y: str) -> float:
    X, Y, w = _c(g, [x])[:, 0], _c(g, [y])[:, 0], g["future_played"].to_numpy(float)
    return float((w * X * Y).sum() / (w * X * X).sum())


def loso(g: pd.DataFrame, x: str, y: str) -> tuple:
    se = sb = 0.0
    per = {}
    for s in SEAS:
        tr, te = g[g["season"] != s], g[g["season"] == s]
        if len(te) < 40:
            continue
        b = slope(tr, x, y)
        X, Y, w = _c(te, [x])[:, 0], _c(te, [y])[:, 0], te["future_played"].to_numpy(float)
        e1, e0 = float((w * (Y - b * X) ** 2).sum()), float((w * Y ** 2).sum())
        per[s] = 100 * (e1 / e0 - 1)
        se, sb = se + e1, sb + e0
    return 100 * (se / sb - 1), per


def test(g: pd.DataFrame, x: str, y: str, boots=1000, perms=500) -> dict:
    g = g.dropna(subset=[x, y]).reset_index(drop=True)
    sd = float(g.groupby(["season", "week", "position"])[x].std().mean())
    b = slope(g, x, y)
    rng = np.random.default_rng(0)
    g["key"] = g["espn_id"].astype(str) + "_" + g["season"].astype(str)
    keys = g["key"].to_numpy()
    uk = np.unique(keys)
    idx = {k: np.flatnonzero(keys == k) for k in uk}
    bs = [slope(g.iloc[np.concatenate([idx[k] for k in rng.choice(uk, len(uk))])], x, y) for _ in range(boots)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    # placebo: hand each player-season another player-season's whole feature path (same season and position), so the
    # null keeps the repeated-checkpoint structure that a row-level shuffle would destroy
    lut = g.set_index(["key", "week"])[x]
    grp = g.drop_duplicates("key").groupby(["season", "position"])["key"].apply(list)
    null = []
    for _ in range(perms):
        m = {}
        for ks in grp:
            m.update(zip(ks, rng.permutation(ks)))
        h = g.assign(**{x: lut.reindex(list(zip(g["key"].map(m), g["week"]))).to_numpy()}).dropna(subset=[x])
        null.append(slope(h, x, y))
    m, per = loso(g, x, y)
    return {"n": len(g), "players": len(uk), "b_sd": b * sd, "ci": (lo * sd, hi * sd), "p": float((np.abs(null) >= abs(b)).mean()),
            "loso": m, "per": per, "sd": sd}


def stability() -> None:
    def pair(a, b, col, nmin, w):
        j = a[["playerId", col, w]].merge(b[["playerId", col, w]], on="playerId").dropna()
        j = j[(j[f"{w}_x"] >= nmin) & (j[f"{w}_y"] >= nmin)]
        return np.corrcoef(j[f"{col}_x"], j[f"{col}_y"])[0, 1], len(j)
    for rep, col, w, nh, ny in (("receiving", "receptionsOe", "targets", 25, 50), ("passing", "completionOe", "attempts", 120, 250),
                                ("passing", "accuracyOe", "attempts", 120, 250)):
        hh = [pair(tab(rep, y, 1, 9), tab(rep, y, 10, 18), col, nh, w) for y in SEAS]
        yy = [pair(tab(rep, y, 1, 18), tab(rep, y + 1, 1, 18), col, ny, w) for y in SEAS[:-1]]
        print(f"stability {col:13} half-to-half r " + " ".join(f"{r:+.2f}" for r, _ in hh) + f" (mean {np.mean([r for r, _ in hh]):+.2f}, n={sum(n for _, n in hh)})"
              f" | year-to-year r " + " ".join(f"{r:+.2f}" for r, _ in yy) + f" (mean {np.mean([r for r, _ in yy]):+.2f}, n={sum(n for _, n in yy)})")


def main() -> None:
    if sys.argv[1:] == ["fetch"]:
        return fetch()
    stability()
    X = rows().merge(features(mp.pff_espn()), on=["espn_id", "season", "week"], how="left")
    print("rows", len(X), {c: int(X[c].notna().sum()) for c in ("roe", "qb_coe", "coe", "aoe")})
    for pos, cols in (("WR", ["roe", "qb_coe"]), ("TE", ["roe", "qb_coe"]), ("RB", ["roe", "qb_coe"]), ("QB", ["coe", "aoe"])):
        for c in cols:
            for y in ("r_v2", "r_espn"):
                t = test(X[X["position"] == pos], c, y)
                print(f"{pos} {c:6} vs {y:6} n={t['n']:5} ({t['players']} player-seasons) {t['b_sd']:+.2f} ROS ppg per sd "
                      f"[{t['ci'][0]:+.2f},{t['ci'][1]:+.2f}] placebo p={t['p']:.3f} LOSO dMSE {t['loso']:+.2f}% "
                      + " ".join(f"{s % 100}:{v:+.2f}" for s, v in t["per"].items()) + f"  (sd={t['sd']:.2f})", flush=True)


if __name__ == "__main__":
    main()
