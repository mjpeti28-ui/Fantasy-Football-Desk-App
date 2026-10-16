"""Do PFF Premium Stats add to v2's rest-of-season per-game forecast?

Joins PFF league-wide facet leaderboards (tools/fetch_pff_stats.py, cached under
.cache/pff/) onto the usage_rows_* checkpoints. Features for checkpoint W use only
weeks < W: season-to-date (1..W-1) and last 3 (W-3..W-1). Mirrors ecr_signal.ros_report:
LOSO OLS per position weighted by future_played, rows with >= 3 later games and
est_ppg > 0. Each feature group is tested on top of est_ppg (baseline) and on top of
the live blend (est_ppg with 1/3 rank-matched FantasyPros ECR).
  python backtest/pff_stats.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import nflverse_data as nd  # noqa: E402
from data import normalize_name  # noqa: E402
from backtest import ecr_signal as es  # noqa: E402
from backtest import multi_season as ms  # noqa: E402

CACHE = ROOT / ".cache" / "pff"
OUT = ROOT / "history" / "backtest" / "multi"
SEASONS, SKILL = es.SEASONS, es.SKILL
POS = {"HB": "RB", "FB": "RB", "WR": "WR", "TE": "TE", "QB": "QB"}
ECR_W = 1 / 3


def load(facet: str, season: int, lo: int, hi: int) -> pd.DataFrame:
    p = CACHE / facet.replace("/", "_") / str(season) / f"w{lo:02d}-{hi:02d}.json"
    d = json.loads(p.read_text())
    return pd.DataFrame(next(iter(d.values())))


def window_features(season: int, lo: int, hi: int) -> pd.DataFrame:
    """One row per PFF player_id with per-position features over weeks lo..hi."""
    rec = load("receiving/summary", season, lo, hi)
    rus = load("rushing/summary", season, lo, hi)
    pas = load("passing/summary", season, lo, hi)
    team_tgt = rec.groupby("team")["targets"].sum()
    team_db = pas.groupby("team")["dropbacks"].sum()
    team_att = rus.groupby("team")["attempts"].sum()

    r = rec.set_index("player_id")
    f = pd.DataFrame({
        "rec_grade": r["grades_offense"], "route_grade": r["grades_pass_route"], "yprr": r["yprr"],
        "tprr": r["targets"] / r["routes"].replace(0, np.nan),
        "route_share": r["routes"] / r["team"].map(team_db),
        "tgt_share": r["targets"] / r["team"].map(team_tgt),
    })
    u = rus.set_index("player_id")
    g = pd.DataFrame({
        "rush_grade": u["grades_offense"], "run_grade": u["grades_run"], "elusive": u["elusive_rating"],
        "yco": u["yco_attempt"], "rush_share": u["attempts"] / u["team"].map(team_att),
        "rush_pg": u["attempts"] / u["player_game_count"],
    })
    q = pas.set_index("player_id")
    h = pd.DataFrame({
        "pass_grade_off": q["grades_offense"], "pass_grade": q["grades_pass"], "btt": q["btt_rate"],
        "twp": q["twp_rate"], "adot": q["avg_depth_of_target"],
        "db_pg": q["dropbacks"] / q["player_game_count"],
    })
    out = f.join(g, how="outer").join(h, how="outer")
    out["routes_raw"] = r["routes"]  # leakage check only
    return out


# ------------------------------------------------------------------ id mapping

def id_map(season: int) -> pd.DataFrame:
    """PFF player_id -> espn_id for one season: normalized name + position, ties broken by team."""
    frames = [load(fc, season, 1, 12) for fc in ("receiving/summary", "rushing/summary", "passing/summary")]
    p = pd.concat(frames)[["player_id", "player", "position", "team"]]
    p["pos"] = p["position"].map(POS)
    p = p.dropna(subset=["pos"])
    p["nn"] = p["player"].map(normalize_name)
    p = p.groupby("player_id").agg(nn=("nn", "first"), pos=("pos", lambda s: s.mode().iat[0]),
                                   teams=("team", lambda s: set(s))).reset_index()

    st = pd.read_csv(ROOT / "history" / "nflverse" / str(season) / "player_stats.csv")
    st = st[st["season"] == season]
    ids = pd.read_csv(ROOT / "history" / "nflverse" / "ids.csv").dropna(subset=["gsis_id", "espn_id"])
    g2e = ids.drop_duplicates("gsis_id").set_index("gsis_id")["espn_id"].astype("int64")
    n = st.groupby("player_id").agg(name=("player_display_name", "first"), pos=("position", "first"),
                                    teams=("team", lambda s: set(s))).reset_index()
    n["espn_id"] = n["player_id"].map(g2e)
    n = n.dropna(subset=["espn_id"])
    n["nn"] = n["name"].map(normalize_name)
    by_name = {k: v for k, v in n.groupby("nn")}

    rows = []
    for r in p.itertuples():
        c = by_name.get(r.nn)
        if c is None:
            continue
        c2 = c[c["pos"] == r.pos]
        c2 = c2 if len(c2) else c
        if len(c2) > 1:
            c2 = c2[c2["teams"].map(lambda t: bool(t & r.teams))]
        if len(c2) == 1:
            rows.append((season, r.player_id, int(c2["espn_id"].iat[0])))
    m = pd.DataFrame(rows, columns=["season", "pff_id", "espn_id"])
    return m.drop_duplicates("espn_id", keep=False).drop_duplicates("pff_id", keep=False)


# ------------------------------------------------------------------ rows

def build_rows() -> pd.DataFrame:
    d = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob("usage_rows_*.csv"))], ignore_index=True)
    d = d[d["position"].isin(SKILL) & d["week"].between(3, 13)]
    feats = []
    for y in SEASONS:
        m = id_map(y).set_index("espn_id")["pff_id"]
        for W in range(3, 14):
            std = window_features(y, 1, W - 1)
            lo = W - 3 if W - 3 > 1 else 1
            l3 = window_features(y, lo, W - 1).add_suffix("_l3")
            f = std.join(l3, how="outer")
            sub = d[(d["season"] == y) & (d["week"] == W)][["espn_id"]].copy()
            sub["pff_id"] = sub["espn_id"].map(m)
            sub = sub.join(f, on="pff_id")
            sub["season"], sub["week"] = y, W
            feats.append(sub)
    F = pd.concat(feats, ignore_index=True)
    x = d.merge(F, on=["season", "week", "espn_id"], how="left")

    E = nd.ecr_table().dropna(subset=["espn_id"]).assign(espn_id=lambda e: e["espn_id"].astype("int64"))
    e = E[E["page_type"].isin([f"redraft-{p.lower()}" for p in SKILL])][["season", "week", "espn_id", "ecr"]]
    x = x.merge(e.drop_duplicates(["season", "week", "espn_id"]), on=["season", "week", "espn_id"], how="left")
    x = x[x["est_ppg"] > 0].copy()
    # live blend: rank-matched ECR at 1/3 (as multi_season.fit_ecr_weight); no ECR -> est_ppg
    x["ecr_ppg"] = np.nan
    for _, g in x[x["ecr"].notna()].groupby(["season", "week", "position"]):
        vals = np.sort(g["est_ppg"].to_numpy())[::-1][: len(g)]
        order = g["ecr"].rank(method="first").astype(int).to_numpy() - 1
        x.loc[g.index, "ecr_ppg"] = vals[order]
    x["live"] = np.where(x["ecr_ppg"].notna(), (1 - ECR_W) * x["est_ppg"] + ECR_W * x["ecr_ppg"], x["est_ppg"])
    return x


GROUPS = {
    "WR": {"grade": ["rec_grade"], "route_grade": ["route_grade"], "yprr": ["yprr"], "tprr": ["tprr"],
           "route_share": ["route_share", "route_share_l3"], "tgt_share": ["tgt_share", "tgt_share_l3"],
           "yprr_l3": ["yprr", "yprr_l3"]},
    "RB": {"grade": ["rush_grade"], "run_grade": ["run_grade"], "elusive": ["elusive"], "yco": ["yco"],
           "rush_share": ["rush_share", "rush_share_l3"], "tgt_share": ["tgt_share", "tgt_share_l3"],
           "yprr": ["yprr"], "route_share": ["route_share", "route_share_l3"]},
    "QB": {"grade": ["pass_grade_off"], "pass_grade": ["pass_grade"], "btt_twp": ["btt", "twp"],
           "adot": ["adot"], "rush_pg": ["rush_pg", "rush_pg_l3"], "db_pg": ["db_pg"]},
}
GROUPS["TE"] = {k: v for k, v in GROUPS["WR"].items()}


def evaluate(x: pd.DataFrame, base: str, cols) -> dict:
    """LOSO fit future_ppg ~ base + cols for one position; per-season rank corr and weighted MAE."""
    x = x.copy()
    x["p"] = es.loso_reg(x, [base] + list(cols), "future_ppg", "future_played")
    out = {}
    for y, g in x.groupby("season"):
        out[y] = (es.rank_corr(g, "p", "future_ppg", ("season", "week")),
                  float(np.average((g["p"] - g["future_ppg"]).abs(), weights=g["future_played"])))
    rc = np.mean([v[0] for v in out.values()])
    mae = float(np.average((x["p"] - x["future_ppg"]).abs(), weights=x["future_played"]))
    return {"rc": rc, "mae": mae, "by_season": out}


def impute(x: pd.DataFrame, cols) -> pd.DataFrame:
    x = x.copy()
    for c in cols:
        x[c] = x[c].replace([np.inf, -np.inf], np.nan)
        x[c] = x[c].fillna(x.groupby("position")[c].transform("median"))
    return x


def report(x: pd.DataFrame):
    x = x[x["future_played"] >= 3].copy()
    lines = []
    for pos in SKILL:
        g = x[x["position"] == pos].copy()
        allc = sorted({c for v in GROUPS[pos].values() for c in v})
        cov = g[allc[0]].notna().mean()
        g = impute(g, allc)
        for base in ("est_ppg", "live"):
            b = evaluate(g, base, [])
            res = {}
            for name, cols in GROUPS[pos].items():
                res[name] = evaluate(g, base, cols)
            # best together: groups that beat baseline in rank corr in >= 4 of 5 seasons
            good = [n for n, r in res.items()
                    if sum(r["by_season"][y][0] > b["by_season"][y][0] for y in SEASONS) >= 4 and r["rc"] > b["rc"]]
            good = sorted(good, key=lambda n: -res[n]["rc"])[:3]
            res["all"] = evaluate(g, base, sorted({c for v in GROUPS[pos].values() for c in v}))
            if len(good) > 1:
                res["best: " + "+".join(good)] = evaluate(g, base, sorted({c for n in good for c in GROUPS[pos][n]}))
            for name, r in res.items():
                drc = {y: r["by_season"][y][0] - b["by_season"][y][0] for y in SEASONS}
                dmae = {y: r["by_season"][y][1] - b["by_season"][y][1] for y in SEASONS}
                lines.append({"pos": pos, "base": base, "group": name, "n": len(g), "pff_cov": round(cov, 3),
                              "base_rc": b["rc"], "d_rc": r["rc"] - b["rc"], "base_mae": b["mae"], "d_mae": r["mae"] - b["mae"],
                              "rc_up_seasons": sum(v > 0 for v in drc.values()),
                              "mae_down_seasons": sum(v < 0 for v in dmae.values()),
                              "d_rc_by_season": " ".join(f"{v:+.3f}" for v in drc.values()),
                              "d_mae_by_season": " ".join(f"{v:+.3f}" for v in dmae.values())})
    return pd.DataFrame(lines)


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 200)
    x = build_rows()
    x["has_any"] = x[["rec_grade", "rush_grade", "pass_grade"]].notna().any(axis=1)
    print("== id match rate (all checkpoint rows, est_ppg > 0)")
    x["bucket"] = np.where(x["est_ppg"] > 5, "est>5", "est<=5")
    print(x.groupby(["position", "bucket"]).agg(rows=("espn_id", "size"), id_matched=("pff_id", lambda s: s.notna().mean()),
                                                has_pff_stats=("has_any", "mean")).round(3))
    print("\n== leakage check: mean season-to-date routes (WR/TE) by checkpoint week")
    print(x[x["position"].isin(["WR", "TE"])].groupby("week")[["routes_raw", "routes_raw_l3"]].mean().round(1).T)
    for tag, sub in (("all", x), ("est_gt5", x[x["est_ppg"] > 5])):
        r = report(sub)
        r.to_csv(OUT / f"pff_stats_results_{tag}.csv", index=False)
        print(f"\n== [{tag} rows] deltas vs base (d_rc > 0 and d_mae < 0 are improvements)")
        print(r.drop(columns=["n"]).round(4).to_string(index=False))
