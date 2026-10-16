"""How uncertain is a player's rest-of-season *level* (not week-to-week noise)?

For each checkpoint row (model_rows_*.csv) with >= 4 future games played, per-game ratio
r = points / est_ppg. Split the played games into odd/even halves: Cov(mean_odd, mean_even)
across players estimates the variance of the persistent level (role, talent, situation);
the rest of the per-game variance is weekly noise. Groups: position x role (depth-chart rank)
x checkpoint half; optionally x expert-spread tercile.

Then a distribution test (LOSO): simulate each player's rest-of-season total with
  point model : availability draws x weekly noise           (level fixed at 1)
  level model : the same x a persistent level ~ lognormal(mean m_g, var v_g)
and score against actual totals: CRPS, 10/50/90% quantile coverage, Brier of P(total >= 50)
and P(total >= 100), by group.
  python3 backtest/level_uncertainty.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "history" / "backtest" / "multi"
SEASONS = [2021, 2022, 2023, 2024, 2025]
LAST = 17


def load() -> pd.DataFrame:
    d = pd.concat([pd.read_csv(OUT / f"model_rows_{y}.csv") for y in SEASONS], ignore_index=True)
    d["role"] = np.select([d["position"].isin(["K", "DST"]), d["depth"] == 1, d["depth"] == 2],
                          ["-", "starter", "backup"], "deep/off")
    d["half"] = np.where(d["W"] <= 7, "early", "late")
    return d


def split_half(d: pd.DataFrame) -> pd.DataFrame:
    """Per row: mean ratio over odd / even played future games, n games, per-game ratio var."""
    recs = []
    for r in d.itertuples(index=False):
        rr = []
        for w in range(int(r.W), LAST + 1):
            if getattr(r, f"play_w{w}", 0) == 1:
                rr.append(getattr(r, f"pts_w{w}") / r.est_ppg)
        n = len(rr)
        if n >= 4:
            a = np.array(rr)
            recs.append((a[0::2].mean(), a[1::2].mean(), a.mean(), a.var(ddof=1), n))
        else:
            recs.append((np.nan, np.nan, np.nan, np.nan, n))
    x = pd.DataFrame(recs, columns=["odd", "even", "lvl", "within_var", "n_played"], index=d.index)
    return d.join(x)


def group_params(x: pd.DataFrame, keys) -> pd.DataFrame:
    g = x.dropna(subset=["odd"]).groupby(keys)
    out = g.apply(lambda h: pd.Series({
        "n": len(h),
        "mean_level": h["lvl"].mean(),
        "level_var": max(np.cov(h["odd"], h["even"])[0, 1], 0.0),
        "weekly_var": max(h["within_var"].mean(), 1e-6),
    }), include_groups=False)
    out["level_sd"] = np.sqrt(out["level_var"])
    out["weekly_cv"] = np.sqrt(out["weekly_var"])
    return out


def simulate_totals(x: pd.DataFrame, params: pd.DataFrame, keys, use_level: bool, use_mean: bool,
                    n_sims: int = 300, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tot = np.zeros((len(x), n_sims))
    for j, r in enumerate(x.itertuples(index=False)):
        key = tuple(getattr(r, k) for k in keys)
        pr = params.loc[key] if key in params.index else params.iloc[0] * np.nan
        lv_var = pr["level_var"] if use_level and pd.notna(pr["level_var"]) else 0.0
        m = pr["mean_level"] if use_mean and pd.notna(pr["mean_level"]) else 1.0
        wv = pr["weekly_var"] if pd.notna(pr["weekly_var"]) else 0.5
        # lognormal level with mean m and variance lv_var
        if lv_var > 0:
            s2 = np.log(1 + lv_var / m ** 2)
            L = rng.lognormal(np.log(m) - s2 / 2, np.sqrt(s2), n_sims)
        else:
            L = np.full(n_sims, m)
        # weekly noise: gamma with mean 1, var wv (per game, multiplicative)
        k = 1.0 / wv
        acc = np.zeros(n_sims)
        for w in range(int(r.W), LAST + 1):
            a = getattr(r, f"xa_w{w}")
            if a <= 0:
                continue
            playw = rng.random(n_sims) < a
            acc += playw * r.est_ppg * L * rng.gamma(k, 1 / k, n_sims)
        tot[j] = acc
    return tot


def crps(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    # sample CRPS: E|X - y| - 0.5 E|X - X'|
    s = np.sort(samples, axis=1)
    n = s.shape[1]
    t1 = np.abs(s - y[:, None]).mean(axis=1)
    i = np.arange(1, n + 1)
    t2 = (2 * i - n - 1) @ s.T / (n * n)
    return t1 - t2


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    d = load()
    x = split_half(d)
    x.to_pickle(OUT / "level_rows.pkl")
    keys = ["position", "role", "half"]
    P = group_params(x, keys)
    print("persistent level vs weekly noise (all seasons):")
    print(P.round(3).to_string())
    # stability: level_sd by season-held-out fits
    stab = {}
    for held in SEASONS:
        stab[held] = group_params(x[x["season"] != held], ["position", "role"])["level_sd"]
    print("\nlevel_sd fitted without each season (position x role):")
    print(pd.DataFrame(stab).round(3).to_string())


def distribution_test(x: pd.DataFrame, keys=("position", "role", "half"), checkpoints=(3, 5, 7, 9, 11), seed=0):
    keys = list(keys)
    ev = x[x["W"].isin(checkpoints) & ((x["team_id"] > 0) | (x["est_ppg"] >= 5))].copy()
    ev["actual"] = sum(ev[f"pts_w{w}"].fillna(0) for w in range(3, LAST + 1) if f"pts_w{w}" in ev) * 0
    ev["actual"] = [sum(getattr(r, f"pts_w{w}") for w in range(int(r.W), LAST + 1)) for r in ev.itertuples(index=False)]
    variants = {"point": (False, False), "mean only": (False, True), "level": (True, False), "level+mean": (True, True)}
    res = []
    for held in SEASONS:
        te = ev[ev["season"] == held]
        P = group_params(x[x["season"] != held], keys)
        y = te["actual"].to_numpy()
        for name, (ul, um) in variants.items():
            S = simulate_totals(te, P, keys, ul, um, seed=seed + held)
            q10, q50, q90 = np.percentile(S, [10, 50, 90], axis=1)
            res.append(pd.DataFrame({
                "season": held, "variant": name, "position": te["position"].to_numpy(), "role": te["role"].to_numpy(),
                "crps": crps(S, y), "in80": ((y >= q10) & (y <= q90)).astype(float),
                "below10": (y < q10).astype(float), "above90": (y > q90).astype(float),
                "p50": (S >= 50).mean(axis=1), "y50": (y >= 50).astype(float),
                "p100": (S >= 100).mean(axis=1), "y100": (y >= 100).astype(float),
                "abs_med_err": np.abs(q50 - y), "mean_pred": S.mean(axis=1), "actual": y}))
    return pd.concat(res, ignore_index=True)


def summarize(r: pd.DataFrame, by):
    g = r.assign(b50=(r["p50"] - r["y50"]) ** 2, b100=(r["p100"] - r["y100"]) ** 2).groupby(by)
    return g.agg(n=("crps", "size"), crps=("crps", "mean"), cover80=("in80", "mean"), below10=("below10", "mean"),
                 above90=("above90", "mean"), brier50=("b50", "mean"), brier100=("b100", "mean"),
                 p100=("p100", "mean"), hit100=("y100", "mean"), bias=("mean_pred", "mean"), actual=("actual", "mean"))
