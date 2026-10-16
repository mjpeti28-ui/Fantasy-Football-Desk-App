"""Garbage-time filtering, and scoring backtests on the players that matter.

Two questions, 2021-25, on the usage_rows checkpoints (weeks 3-13, v2 forecasts fit without
that season):

1. Players that matter. Every backtest so far averaged Spearman over every player with
   est_ppg > 0, mostly bench-and-below. Here each season x week x position is also scored on
   the "relevant" set: rostered in the league OR inside the top N by the live forecast
   (QB 15, RB 36, WR 48, TE 15), plus NDCG@k on that set (gain = realized ROS ppg, so
   ordering the top of the board counts most). Does the live blend's edge over v2 change?

2. Garbage time. Plays in Q3/Q4 with a margin over M points (M = 14/21/28) and, separately,
   win probability outside [0.05, 0.95]. For each player, season-to-date:
     - clean_frac: share of his fantasy points scored outside garbage time, relative to
       his position's average share that week (1 = typical, < 1 = garbage-heavy)
     - est_clean: the live forecast with his observed ppg rescaled by clean_frac, entering
       the same k-weighted average v2 uses (so only the observed part moves)
   and those are scored against the live blend on both player sets.

The play-by-play is read from a local download (default /tmp/claude-0/pbp/pbp_<season>.csv.gz,
from nflverse-data releases/pbp); it is not committed.

    python backtest/garbage_time.py [pbp_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEASONS = [2021, 2022, 2023, 2024, 2025]
ROWS = ROOT / "history" / "backtest" / "multi"
TOP_N = {"QB": 15, "RB": 36, "WR": 48, "TE": 15}
NDCG_K = {"QB": 12, "RB": 24, "WR": 36, "TE": 12}
W_ECR = 1 / 3
COLS = ["season", "season_type", "week", "posteam", "qtr", "score_differential", "wp", "play_type",
        "two_point_attempt", "passer_player_id", "receiver_player_id", "rusher_player_id",
        "complete_pass", "passing_yards", "receiving_yards", "rushing_yards", "pass_touchdown",
        "rush_touchdown", "interception"]
FILTERS = {"m14": ("margin", 14), "m21": ("margin", 21), "m28": ("margin", 28), "wp05": ("wp", 0.05)}


def play_points(pbp_dir: Path, season: int) -> pd.DataFrame:
    """Per player-week fantasy points (PPR, no fumbles/2pt), total and per garbage filter."""
    b = pd.read_csv(pbp_dir / f"pbp_{season}.csv.gz", usecols=COLS, low_memory=False)
    b = b[(b["season_type"] == "REG") & b["play_type"].isin(["pass", "run"]) & (b["two_point_attempt"].fillna(0) != 1)].copy()
    for c in ["complete_pass", "passing_yards", "receiving_yards", "rushing_yards", "pass_touchdown", "rush_touchdown", "interception"]:
        b[c] = b[c].fillna(0.0)
    margin = b["score_differential"].abs()
    late = b["qtr"] >= 3
    flags = {}
    for name, (kind, v) in FILTERS.items():
        flags[name] = (late & (margin > v)) if kind == "margin" else ((b["wp"] < v) | (b["wp"] > 1 - v))
    parts = []
    rec = b[b["receiver_player_id"].notna()]
    parts.append(pd.DataFrame({"gsis": rec["receiver_player_id"], "week": rec["week"], "idx": rec.index,
                               "pts": rec["complete_pass"] + 0.1 * rec["receiving_yards"] + 6 * rec["pass_touchdown"]}))
    ru = b[b["rusher_player_id"].notna()]
    parts.append(pd.DataFrame({"gsis": ru["rusher_player_id"], "week": ru["week"], "idx": ru.index,
                               "pts": 0.1 * ru["rushing_yards"] + 6 * ru["rush_touchdown"]}))
    pa = b[b["passer_player_id"].notna() & (b["play_type"] == "pass")]
    parts.append(pd.DataFrame({"gsis": pa["passer_player_id"], "week": pa["week"], "idx": pa.index,
                               "pts": 0.04 * pa["passing_yards"] + 4 * pa["pass_touchdown"] - 2 * pa["interception"]}))
    p = pd.concat(parts, ignore_index=True)
    for name, f in flags.items():
        p[f"g_{name}"] = p["pts"] * f.reindex(p["idx"]).to_numpy()
    out = p.drop(columns="idx").groupby(["gsis", "week"], as_index=False).sum()
    out["season"] = season
    return out


def season_to_date(pts: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Attach season-to-date (weeks < W) clean fractions to each checkpoint row."""
    ids = pd.read_csv(ROOT / "history" / "nflverse" / "ids.csv").dropna(subset=["gsis_id", "espn_id"])
    m = ids.drop_duplicates("gsis_id").set_index("gsis_id")["espn_id"].astype("int64")
    pts = pts.assign(espn_id=pts["gsis"].map(m)).dropna(subset=["espn_id"])
    pts["espn_id"] = pts["espn_id"].astype("int64")
    out = rows.copy()
    for name in FILTERS:
        out[f"cf_{name}"] = np.nan
    for (s, W), idx in out.groupby(["season", "week"]).groups.items():
        sd = pts[(pts["season"] == s) & (pts["week"] < W)].groupby("espn_id").sum(numeric_only=True)
        sub = out.loc[idx]
        tot = sub["espn_id"].map(sd["pts"])
        for name in FILTERS:
            g = sub["espn_id"].map(sd[f"g_{name}"])
            out.loc[idx, f"cf_{name}"] = np.where(tot > 5, 1 - g / tot, np.nan)
    for name in FILTERS:
        c = f"cf_{name}"
        avg = out.groupby(["season", "week", "position"])[c].transform("mean")
        out[c] = (out[c] / avg).fillna(1.0)
    return out


def ecr_series() -> pd.DataFrame:
    import nflverse_data as nd

    E = nd.ecr_table().dropna(subset=["espn_id"])
    E = E[E["page_type"].isin(["redraft-qb", "redraft-rb", "redraft-wr", "redraft-te"])]
    return E.assign(espn_id=E["espn_id"].astype("int64")).groupby(["season", "week", "espn_id"])["ecr"].min().rename("ecr").reset_index()


def rank_matched(g: pd.DataFrame, rank_col: str) -> np.ndarray:
    vals = np.sort(g["est_ppg"].to_numpy())[::-1]
    order = g[rank_col].rank(method="first").astype(int).to_numpy() - 1
    return vals[order]


def spearman(a, b) -> float:
    a, b = pd.Series(np.asarray(a, float)), pd.Series(np.asarray(b, float))
    return float(a.rank().corr(b.rank()))


def ndcg(pred, gain, k: int) -> float:
    pred, gain = np.asarray(pred, float), np.clip(np.asarray(gain, float), 0, None)
    k = min(k, len(pred))
    disc = 1 / np.log2(np.arange(2, k + 2))
    dcg = (gain[np.argsort(-pred)][:k] * disc).sum()
    idcg = (np.sort(gain)[::-1][:k] * disc).sum()
    return float(dcg / idcg) if idcg > 0 else np.nan


def score(x: pd.DataFrame, cols, relevant: bool) -> pd.DataFrame:
    res = []
    for (s, w, pos), g in x.groupby(["season", "week", "position"]):
        if relevant:
            g = g[g["relevant"]]
        if len(g) <= 5:
            continue
        r = {"season": s, "week": w, "position": pos}
        for c in cols:
            r[f"rho {c}"] = spearman(g[c], g["future_ppg"])
            if relevant:
                r[f"ndcg {c}"] = ndcg(g[c], g["future_ppg"], NDCG_K[pos])
        res.append(r)
    return pd.DataFrame(res)


def summarize(r: pd.DataFrame, base: str, cols) -> pd.DataFrame:
    """Mean delta vs ``base`` by position, with seasons improved out of 5."""
    out = []
    for metric in ("rho", "ndcg"):
        if f"{metric} {base}" not in r:
            continue
        for c in cols:
            if c == base:
                continue
            d = r[f"{metric} {c}"] - r[f"{metric} {base}"]
            for pos, dd in d.groupby(r["position"]):
                per = dd.groupby(r.loc[dd.index, "season"]).mean()
                out.append({"metric": metric, "model": c, "position": pos, "delta": round(float(dd.mean()), 4),
                            "seasons better": f"{int((per > 0).sum())}/{len(per)}"})
    return pd.DataFrame(out)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    pbp_dir = Path(argv[0]) if argv else Path("/tmp/claude-0/pbp")
    pd.set_option("display.width", 220)
    params = json.loads((ROOT / "valuation_params.json").read_text())

    rows = pd.concat([pd.read_csv(ROWS / f"usage_rows_{s}.csv") for s in SEASONS], ignore_index=True)
    rows = rows[rows["position"].isin(TOP_N) & (rows["est_ppg"] > 0) & (rows["future_played"] >= 3)].copy()
    rows = rows.merge(ecr_series(), on=["season", "week", "espn_id"], how="left")
    rows["ecr_ppg"] = np.nan
    for _, g in rows[rows["ecr"].notna()].groupby(["season", "week", "position"]):
        rows.loc[g.index, "ecr_ppg"] = rank_matched(g, "ecr")
    rows["live"] = np.where(rows["ecr_ppg"].notna(), (1 - W_ECR) * rows["est_ppg"] + W_ECR * rows["ecr_ppg"], rows["est_ppg"])
    rk = rows.groupby(["season", "week", "position"])["live"].rank(ascending=False)
    rows["relevant"] = rows["rostered"].astype(bool) | (rk <= rows["position"].map(TOP_N))

    pts = pd.concat([play_points(pbp_dir, s) for s in SEASONS], ignore_index=True)
    x = season_to_date(pts, rows)

    # est with observed ppg rescaled by the clean fraction, through v2's own k-weighted average
    k = x["position"].map(params["k"]).astype(float)
    w_obs = x["n"] / (k + x["n"])
    w_espn = np.where(x["espn_now"] > 0, x["position"].map(params["espn_weight"]).astype(float), 0.0)
    for name in FILTERS:
        shift = (1 - w_espn) * w_obs * x["obs_ppg"] * (x[f"cf_{name}"] - 1)
        est_c = x["est_ppg"] + shift
        x[f"live_{name}"] = np.where(x["ecr_ppg"].notna(), (1 - W_ECR) * est_c + W_ECR * x["ecr_ppg"], est_c)

    n_all, n_rel = len(x), int(x["relevant"].sum())
    print(f"checkpoint rows: {n_all} (est>0, >=3 future games); relevant: {n_rel} "
          f"({x['relevant'].mean():.0%}); garbage share of fantasy points (m21): "
          f"{1 - (pts['pts'].sum() - pts['g_m21'].sum()) / pts['pts'].sum():.1%}\n")

    cols = ["est_ppg", "live"] + [f"live_{n}" for n in FILTERS]
    for relevant in (False, True):
        r = score(x, cols, relevant)
        label = "RELEVANT players (rostered or top-N)" if relevant else "ALL players (as in earlier backtests)"
        print(f"== {label}")
        means = r.drop(columns=["season", "week"]).groupby("position").mean().round(3)
        print(means.to_string())
        print("\nECR's edge (live - v2):")
        print(summarize(r, "est_ppg", ["live"]).to_string(index=False))
        print("\nGarbage-time filters vs live:")
        print(summarize(r, "live", cols[2:]).to_string(index=False))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
