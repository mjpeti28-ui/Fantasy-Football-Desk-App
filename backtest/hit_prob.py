"""Thin probability lookups on est_ppg alone (shipping-recommendations items 2 and 4).

  P(start) = P(rest-of-season ppg >= the position's starter line | est_ppg, position)
  P(boom)  = same with the "boom" line (top half of starters)
  P(bust)  = P(rest-of-season ppg < 0.7 x est_ppg | est_ppg, position)   -- ledger tracking only

Lines come from the 10-team 2026 league: the (10*k)-th best rest-of-season ppg at each position
(k = QB 1, RB 2, WR 3, TE 1; flex/RB-WR/WR-TE slots split by assumption), averaged over 2021-25
checkpoints. Per-position logistic on est_ppg, the same machinery as ml_signal.py clf (L0).

  python3 backtest/hit_prob.py check     # leave-one-season-out reliability (ECE, worst bin)
  python3 backtest/hit_prob.py fit       # fit on all seasons -> valuation_params.json "hit_prob"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sklearn.linear_model import LogisticRegression  # noqa: E402

ROWS = ROOT / "history" / "backtest" / "multi" / "ml_rows.csv"
PARAMS = ROOT / "valuation_params.json"
SKILL = ["QB", "RB", "WR", "TE"]
TEAMS = 10
STARTERS = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}
TASKS = ("start", "boom", "bust")


def load() -> pd.DataFrame:
    d = pd.read_csv(ROWS)
    return d[(d["future_played"] >= 3) & d["position"].isin(SKILL)].reset_index(drop=True)


def lines(d: pd.DataFrame) -> dict:
    """Starter and boom lines in ppg per position, averaged over season x checkpoint pools."""
    out = {}
    for pos, k in STARTERS.items():
        g = d[d["position"] == pos]
        ranked = g.groupby(["season", "W"])["future_ppg"].apply(lambda s: s.sort_values(ascending=False).to_numpy())
        out[pos] = {"start": float(np.mean([r[min(TEAMS * k, len(r)) - 1] for r in ranked])),
                    "boom": float(np.mean([r[min(TEAMS * k // 2, len(r)) - 1] for r in ranked]))}
    return out


def target(d: pd.DataFrame, task: str, ln: dict) -> np.ndarray:
    if task == "bust":
        return (d["future_ppg"] < 0.7 * d["est_ppg"]).astype(int).to_numpy()
    return (d["future_ppg"] >= d["position"].map({p: x[task] for p, x in ln.items()})).astype(int).to_numpy()


def fit(d: pd.DataFrame, ln: dict) -> dict:
    out = {}
    for task in TASKS:
        y = target(d, task, ln)
        out[task] = {}
        for pos in SKILL:
            b = (d["position"] == pos).to_numpy()
            m = LogisticRegression(C=1e6, max_iter=2000).fit(d.loc[b, ["est_ppg"]].to_numpy(), y[b])
            out[task][pos] = [float(m.intercept_[0]), float(m.coef_[0][0])]
    return out


def predict(coefs: dict, pos: pd.Series, est: pd.Series) -> np.ndarray:
    ab = np.array([coefs.get(p, [np.nan, np.nan]) for p in pos])
    return 1 / (1 + np.exp(-(ab[:, 0] + ab[:, 1] * est.to_numpy())))


def reliability(y, p, bins=10, min_n=50):
    q = pd.cut(p, np.linspace(0, 1, bins + 1), include_lowest=True)
    t = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q", observed=True).agg(n=("y", "size"), y=("y", "mean"), p=("p", "mean"))
    ece = float((t["n"] * (t["y"] - t["p"]).abs()).sum() / t["n"].sum())
    big = t[t["n"] >= min_n]
    return ece, float((big["y"] - big["p"]).abs().max()), t


def cmd_check() -> None:
    d = load()
    for task in TASKS:
        d["p_" + task], d["y_" + task] = np.nan, np.nan
    for held in sorted(d["season"].unique()):
        tr, te = d["season"] != held, d["season"] == held
        ln = lines(d[tr])
        c = fit(d[tr], ln)
        for task in TASKS:
            d.loc[te, "p_" + task] = predict(c[task], d.loc[te, "position"], d.loc[te, "est_ppg"])
            d.loc[te, "y_" + task] = target(d[te], task, ln)
    for task in TASKS:
        for label, m in (("all", d.index == d.index), ("free agents", ~d["rostered"].astype(bool))):
            ece, worst, _ = reliability(d.loc[m, "y_" + task], d.loc[m, "p_" + task])
            print(f"{task:5s} {label:11s} n={m.sum():6d} base={d.loc[m, 'y_' + task].mean():.3f} "
                  f"ECE={ece:.3f} worst bin (n>=50) off by {worst:.3f}")
        _, _, t = reliability(d["y_" + task], d["p_" + task])
        print(t.round(3).to_string(), "\n")


def cmd_fit() -> None:
    d = load()
    ln = lines(d)
    params = json.loads(PARAMS.read_text())
    params["hit_prob"] = {"source": "backtest/hit_prob.py fit, ml_rows.csv 2021-2025 W3-13, est_ppg only",
                          "lines_ppg": {p: {k: round(x, 2) for k, x in v.items()} for p, v in ln.items()},
                          "logit": fit(d, ln)}
    PARAMS.write_text(json.dumps(params, indent=1) + "\n")
    print(json.dumps(params["hit_prob"]["lines_ppg"]))


if __name__ == "__main__":
    {"check": cmd_check, "fit": cmd_fit}[sys.argv[1] if len(sys.argv) > 1 else "check"]()
