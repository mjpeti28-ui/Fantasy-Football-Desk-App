"""Rest-of-season schedule strength (defensive EPA allowed) and team pass rate over expected (PROE):
does either add anything to the live blend's rest-of-season forecast?

v2 has no opponent term (it uses the NFL schedule only for byes). ESPN's weekly projection prices the
current week's matchup (backtest/matchup_mechanisms.py), but the rest-of-season forecast is a flat ppg.
At each usage_rows checkpoint (2021-25, weeks 3-13) and for each player's team (his last team before W):

  sos_pass / sos_run : mean rating of the team's remaining opponents (weeks >= W), each rated as of W
                       with app/engine/matchups_rating (EPA allowed per dropback / designed run, shrunk
                       toward last season; higher = softer). Same ratings the Matchups card shows.
  proe               : the team's pass rate over expected so far (nflverse pass_oe, shrunk with 200 plays)

Each is applied as a multiplier on the live forecast, live x (1 + X b), with b fit per position by least
squares on future_ppg / live - 1, leave-one-season-out. Scored like backtest/garbage_time.py: Spearman on
all players and on the relevant set (rostered or top-N), plus NDCG@k. Placebo: the same fit with the team
features shuffled across teams within each season-week.

The pbp (for pass_oe) is read from a local download (default /tmp/claude-0/pbp/pbp_<season>.csv.gz), not
committed; EPA ratings come from the Matchups cache (app/cache.nosync/matchups).

    python backtest/schedule_proe.py [pbp_dir]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.engine import matchups_rating as mr  # noqa: E402
from backtest import garbage_time as gt  # noqa: E402

SEASONS = gt.SEASONS
K_PROE = 200.0
FEATS = {"sos": ["sos_pass", "sos_run"], "proe": ["proe"], "both": ["sos_pass", "sos_run", "proe"]}


def plays(season: int) -> pd.DataFrame:
    return pd.read_csv(ROOT / "app" / "cache.nosync" / "matchups" / f"{season}.csv.gz",
                       usecols=["season_type", "week", "game_id", "posteam", "defteam", "epa", "db", "run"], low_memory=False)


def team_features(season: int, pbp_dir: Path) -> pd.DataFrame:
    """One row per (team, checkpoint week W): remaining-schedule strength and PROE to date."""
    cur, prev = plays(season), plays(season - 1)
    sched = cur.drop_duplicates(["game_id", "posteam"])[["week", "posteam", "defteam"]]
    oe = pd.read_csv(pbp_dir / f"pbp_{season}.csv.gz", usecols=["season_type", "week", "posteam", "pass_oe"], low_memory=False)
    oe = oe[(oe["season_type"] == "REG") & oe["pass_oe"].notna()]
    prior = {k: mr.team_table(prev, k) for k in ("pass", "run")}
    out = []
    for W in range(3, 14):
        before = cur[cur["week"] < W]
        r = {k: mr.rate(prior[k], mr.team_table(before, k)) for k in ("pass", "run")}
        rest = sched[sched["week"] >= W].assign(p=lambda d: d["defteam"].map(r["pass"]), q=lambda d: d["defteam"].map(r["run"]))
        s = rest.groupby("posteam")[["p", "q"]].mean().rename(columns={"p": "sos_pass", "q": "sos_run"})
        o = oe[oe["week"] < W].groupby("posteam")["pass_oe"].agg(["sum", "count"])
        s["proe"] = (o["sum"] / (o["count"] + K_PROE)).reindex(s.index).fillna(0.0) / 100  # pass_oe is in percent
        out.append(s.reset_index().rename(columns={"posteam": "team"}).assign(season=season, week=W))
    return pd.concat(out, ignore_index=True)


def player_teams(season: int) -> pd.DataFrame:
    """Each player's most recent team before checkpoint W."""
    r = pd.read_csv(ROOT / "app" / "cache.nosync" / "roles" / str(season) / "roles_weekly.csv", usecols=["week", "posteam", "espn_id"])
    r = r.dropna(subset=["espn_id"]).sort_values("week")
    out = []
    for W in range(3, 14):
        t = r[r["week"] < W].groupby("espn_id")["posteam"].last()
        out.append(pd.DataFrame({"espn_id": t.index.astype("int64"), "team": t.to_numpy(), "season": season, "week": W}))
    return pd.concat(out, ignore_index=True)


def fit_apply(x: pd.DataFrame, cols, out_col: str) -> None:
    """live x (1 + X b), b per position fit leave-one-season-out on future/live - 1 (rows with live >= 4)."""
    x[out_col] = np.nan
    for pos in gt.TOP_N:
        for held in SEASONS:
            tr = x[(x["position"] == pos) & (x["season"] != held) & (x["live"] >= 4)]
            te = (x["position"] == pos) & (x["season"] == held)
            X = np.c_[np.ones(len(tr)), tr[cols]]
            b = np.linalg.lstsq(X, tr["future_ppg"] / tr["live"] - 1, rcond=None)[0]
            x.loc[te, out_col] = x.loc[te, "live"] * (1 + x.loc[te, cols].to_numpy() @ b[1:])
            if out_col.endswith("both") and not out_col.startswith("placebo"):
                COEF.append({"position": pos, "held": held, **{c: b[i + 1] for i, c in enumerate(cols)}})


COEF: list = []


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    pbp_dir = Path(argv[0]) if argv else Path("/tmp/claude-0/pbp")
    pd.set_option("display.width", 220)

    rows = pd.concat([pd.read_csv(gt.ROWS / f"usage_rows_{s}.csv") for s in SEASONS], ignore_index=True)
    rows = rows[rows["position"].isin(gt.TOP_N) & (rows["est_ppg"] > 0) & (rows["future_played"] >= 3)].copy()
    rows = rows.merge(gt.ecr_series(), on=["season", "week", "espn_id"], how="left")
    rows["ecr_ppg"] = np.nan
    for _, g in rows[rows["ecr"].notna()].groupby(["season", "week", "position"]):
        rows.loc[g.index, "ecr_ppg"] = gt.rank_matched(g, "ecr")
    rows["live"] = np.where(rows["ecr_ppg"].notna(), (1 - gt.W_ECR) * rows["est_ppg"] + gt.W_ECR * rows["ecr_ppg"], rows["est_ppg"])
    rk = rows.groupby(["season", "week", "position"])["live"].rank(ascending=False)
    rows["relevant"] = rows["rostered"].astype(bool) | (rk <= rows["position"].map(gt.TOP_N))

    tf = pd.concat([team_features(s, pbp_dir) for s in SEASONS], ignore_index=True)
    pt = pd.concat([player_teams(s) for s in SEASONS], ignore_index=True)
    x = rows.merge(pt, on=["season", "week", "espn_id"], how="left").merge(tf, on=["season", "week", "team"], how="left")
    print(f"checkpoint rows: {len(x)}; with a team: {x['team'].notna().mean():.1%}; relevant: {int(x['relevant'].sum())}")
    allf = sorted({c for v in FEATS.values() for c in v})
    sd = x.drop_duplicates(["season", "week", "team"])[allf].std()
    print("team-level sd:", sd.round(4).to_dict(), "\n")
    x[allf] = x[allf].fillna(0.0)

    rng = np.random.default_rng(0)
    shuf = tf.copy()
    for _, idx in shuf.groupby(["season", "week"]).groups.items():
        shuf.loc[idx, allf] = shuf.loc[idx, allf].to_numpy()[rng.permutation(len(idx))]
    xp = rows.merge(pt, on=["season", "week", "espn_id"], how="left").merge(shuf, on=["season", "week", "team"], how="left")
    xp[allf] = xp[allf].fillna(0.0)

    cols = ["est_ppg", "live"]
    for name, f in FEATS.items():
        fit_apply(x, f, f"live_{name}")
        fit_apply(xp, f, f"placebo_{name}")
        x[f"placebo_{name}"] = xp[f"placebo_{name}"].to_numpy()
        cols += [f"live_{name}", f"placebo_{name}"]

    c = pd.DataFrame(COEF)
    print("LOSO coefficients (sos + proe model), mean and range over the 5 folds; multiplier per unit of feature:")
    print(c.groupby("position")[["sos_pass", "sos_run", "proe"]].agg(["mean", "min", "max"]).round(2).to_string())
    print("per +1 team-level sd, as % of the forecast:",
          {p: {f: round(100 * float(g[f].mean() * sd[f]), 1) for f in ["sos_pass", "sos_run", "proe"]} for p, g in c.groupby("position")}, "\n")

    for relevant in (False, True):
        r = gt.score(x, cols, relevant)
        print("== " + ("RELEVANT players (rostered or top-N)" if relevant else "ALL players"))
        print(r.drop(columns=["season", "week"]).groupby("position").mean().round(3)[[f"rho {c}" for c in cols]].to_string())
        print("\nvs live (delta, seasons better):")
        print(gt.summarize(r, "live", cols[2:]).to_string(index=False))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
