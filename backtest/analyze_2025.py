"""Score the 2025 backtest outputs (run backtest/season_2025.py first)."""
from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "history" / "backtest" / "2025"
WEEKS = [4, 5, 6, 7, 8, 9]  # the week-10 archive is a full-season file, not ROS -- excluded


def spearman(a, b):
    return float(pd.Series(list(a)).rank().corr(pd.Series(list(b)).rank()))


def load(kind):
    return pd.concat([pd.read_csv(OUT / f"{kind}_w{w}.csv") for w in WEEKS], ignore_index=True)


def pairwise_acc(df, pred, target, cross_only=True):
    recs = df[["pos", pred, target]].to_dict("records")
    hits = tot = 0
    for a, b in itertools.combinations(recs, 2):
        if cross_only and a["pos"] == b["pos"]:
            continue
        dp, dt = a[pred] - b[pred], a[target] - b[target]
        if dp == 0 or dt == 0:
            continue
        tot += 1
        hits += (dp > 0) == (dt > 0)
    return hits / tot if tot else float("nan")


def main():
    P, T, R = load("players"), load("teams"), load("repl")
    out = ["== Player projections (rostered QB/RB/WR/TE at checkpoints wk4-9; realized = actual ESPN points checkpoint->wk17)"]
    within = P.groupby(["week", "pos"]).apply(lambda g: spearman(g["proj"], g["ros_points"])).groupby("pos").mean()
    out.append("within-position rank corr (projection vs realized): " + ", ".join(f"{k} {v:.2f}" for k, v in within.items()))
    ratio = P.groupby("pos").apply(lambda g: g["ros_points"].sum() / g["proj"].sum())
    out.append("realized / projected points: " + ", ".join(f"{k} {v:.2f}" for k, v in ratio.items()))
    out.append("")
    out.append("== Cross-position value. Target = realized ROS pts minus realized pts of the 5 best unrostered (by projection) at that position.")
    rows = []
    for w, g in P.groupby("week"):
        rows.append({
            "week": w,
            "rho model VOR": spearman(g["vor"], g["realized_vor"]),
            "rho waiver VOR": spearman(g["vor_waiver"], g["realized_vor"]),
            "rho raw pts": spearman(g["proj"], g["realized_vor"]),
            "pairs model VOR": pairwise_acc(g, "vor", "realized_vor"),
            "pairs waiver VOR": pairwise_acc(g, "vor_waiver", "realized_vor"),
            "pairs raw pts": pairwise_acc(g, "proj", "realized_vor"),
        })
    X = pd.DataFrame(rows).set_index("week")
    out.append(X.round(3).to_string())
    out.append("mean: " + ", ".join(f"{c} {X[c].mean():.3f}" for c in X.columns))
    out.append("")
    out.append("== Replacement level, projected ROS pts (mean over checkpoints)")
    out.append(R.groupby("pos")[["model_repl_proj", "waiver_repl_proj", "waiver_repl_realized"]].mean().round(1).to_string())
    out.append("")
    out.append("== Team level: rank corr (12 teams) with actual points scored checkpoint->wk14")
    z = lambda s: (s - s.mean()) / (s.std(ddof=0) or 1)
    trow = []
    for w, g in T.groupby("week"):
        rec = {"week": w}
        for wb in (0.0, 0.2, 0.4):
            rec[f"comb b={wb}"] = spearman((1 - wb) * z(g["starter_vor"]) + wb * z(g["bench_total"]), g["ros_actual"])
        rec["starter pts"] = spearman(g["starter_proj"], g["ros_actual"])
        rec["ppg so far"] = spearman(g["ppg_to_date"], g["ros_actual"])
        rec["pts+ppg"] = spearman(z(g["starter_proj"]) + z(g["ppg_to_date"]), g["ros_actual"])
        trow.append(rec)
    Y = pd.DataFrame(trow).set_index("week")
    out.append(Y.round(3).to_string())
    out.append("mean: " + ", ".join(f"{c} {Y[c].mean():.3f}" for c in Y.columns))
    text = "\n".join(out)
    print(text)
    (OUT / "summary.txt").write_text(text)


if __name__ == "__main__":
    main()
