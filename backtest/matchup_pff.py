"""Do PFF unit and individual matchups predict anything beyond ESPN's weekly projection?  (2023-25, weeks 4-13)

Features, fixed before looking at results (higher = tougher for the offense player):
  cov_all    snap-weighted PFF coverage grade of the opponent's defenders, season to date (defense/summary w01-(W-1))
  cov_faced  the same grade, but weighted to where this receiver lines up: his slot rate (receiving/summary, same window)
             x the opponent's slot-snap-weighted grade + (1 - slot rate) x the corner-snap-weighted grade
  align_gap  cov_faced - cov_all: the part of the matchup that is about him, not the defense as a whole
  pr_mm      opponent pass-rush grade - own team pass-block grade (v1/teams/summary game grades, trailing, shrunk)
  run_mm     opponent run-defense grade - own team run-block grade (same)
  opp_cov    opponent team coverage grade (same)
Each team grade = (sum of earlier games' grades this season + K * league mean) / (games + K); position-group grades
the same way with K in snaps. Rows: Desk's rostered player-weeks with an ESPN projection (backtest/vegas_signal.rows).
Target: actual - ESPN projection, effect as a share of the projection (as matchup_mechanisms.ols_prop), centred within
season-week-position, 95% CI from a season-week cluster bootstrap. Also: how much ESPN's number already moves with each.

PFF data is read from .cache/pff (gitignored, never committed); pull it with the PFF key on the Mac first.
  python3 backtest/matchup_pff.py            # prints results, writes .cache/pff/report/matchup_pff.json
  python3 backtest/matchup_pff.py schema     # prints what the cached files look like
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
from backtest import matchup_mechanisms as mm  # noqa: E402
from backtest import vegas_signal as vs  # noqa: E402

CACHE = ROOT / ".cache" / "pff"
SEAS = [2023, 2024, 2025]
WEEKS = range(4, 14)  # receiving windows exist for w01-02..w01-12
K_GAMES = 3.0
K_OUT, K_SLOT, K_ALL = 300.0, 120.0, 600.0  # snaps of prior (about 2 games of a team's corner / slot / coverage snaps)
STARTER = {"QB": 14.0, "RB": 8.0, "WR": 8.0, "TE": 6.0}
TESTS = {"WR": [["cov_all", "align_gap"], ["pr_mm"]], "TE": [["cov_all", "align_gap"], ["pr_mm"]],
         "QB": [["opp_cov", "pr_mm"]], "RB": [["run_mm"], ["pr_mm"]]}
FIX = {"ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "LAR": "LA", "SL": "LA", "OAK": "LV", "SD": "LAC", "WSH": "WAS", "JAC": "JAX"}
IDS_URL = "https://raw.githubusercontent.com/dynastyprocess/data/master/files/db_playerids.csv"


def _rows(p: Path) -> list:
    d = json.loads(p.read_text())
    if isinstance(d, list):
        return d
    for v in d.values():
        if isinstance(v, list):
            return v
    return []


def teams() -> dict:
    """PFF franchise_id -> nflverse abbreviation."""
    for p in sorted((CACHE / "v1_teams").rglob("*.json")):
        out = {}
        for r in _rows(p):
            fid = r.get("franchise_id", r.get("id"))
            ab = r.get("abbreviation") or r.get("abbr") or r.get("short_name")
            if fid is not None and ab:
                out[int(fid)] = FIX.get(str(ab).upper(), str(ab).upper())
        if out:
            return out
    raise SystemExit("no v1_teams cache")


def pff_espn() -> dict:
    p = CACHE / "db_playerids.csv"
    if not p.exists():
        import requests
        p.write_bytes(requests.get(IDS_URL, timeout=60).content)
    i = pd.read_csv(p, usecols=["pff_id", "espn_id"]).dropna()
    return {int(a): int(b) for a, b in zip(i["pff_id"], i["espn_id"])}


def _num(d: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        d[c] = pd.to_numeric(d.get(c), errors="coerce")
    return d


# ---------- team unit grades (per game) ----------
def game_grades(tm: dict) -> pd.DataFrame:
    rows = []
    for p in (CACHE / "v1_teams_summary").rglob("f*.json"):
        season = int(p.parent.name)
        fid = int(p.stem[1:])
        for r in _rows(p):
            rows.append({"season": season, "team": tm.get(fid), "week": r.get("week"), "opp": tm.get(r.get("opponent_franchise_id")),
                         **{k: r.get(f"grades_{k}") for k in ("pass_rush_defense", "run_defense", "coverage_defense", "pass_block", "run_block")}})
    g = pd.DataFrame(rows)
    g = _num(g, ["week", "pass_rush_defense", "run_defense", "coverage_defense", "pass_block", "run_block"])
    return g[(g["week"] >= 1) & g["coverage_defense"].notna()]


def trailing(g: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Each team's shrunk mean grade from games before week W, for W in WEEKS."""
    out = []
    for s in SEAS:
        gs = g[g["season"] == s]
        for W in WEEKS:
            e = gs[gs["week"] < W]
            lg = e[cols].mean()
            a = e.groupby("team")[cols].agg(["sum", "count"])
            r = pd.DataFrame({c: (a[(c, "sum")] + K_GAMES * lg[c]) / (a[(c, "count")] + K_GAMES) for c in cols})
            out.append(r.assign(season=s, week=W).reset_index())
    return pd.concat(out, ignore_index=True)


# ---------- coverage by alignment ----------
def coverage(tm: dict) -> pd.DataFrame:
    out = []
    for s in SEAS:
        for W in WEEKS:
            p = CACHE / "defense_summary" / str(s) / f"w01-{W - 1:02d}.json"
            if not p.exists():
                continue
            d = pd.DataFrame(_rows(p))
            d = _num(d, ["grades_coverage_defense", "snap_counts_corner", "snap_counts_slot", "snap_counts_coverage", "franchise_id"])
            d["team"] = d["franchise_id"].map(lambda f: tm.get(int(f)) if pd.notna(f) else None)
            d = d.dropna(subset=["team", "grades_coverage_defense"])
            res = {}
            for name, w, k in (("cov_out", "snap_counts_corner", K_OUT), ("cov_slot", "snap_counts_slot", K_SLOT), ("cov_all", "snap_counts_coverage", K_ALL)):
                x = d[d[w] > 0]
                lg = float((x["grades_coverage_defense"] * x[w]).sum() / x[w].sum())
                a = x.assign(gw=x["grades_coverage_defense"] * x[w]).groupby("team")[["gw", w]].sum()
                res[name] = (a["gw"] + k * lg) / (a[w] + k)
            out.append(pd.DataFrame(res).assign(season=s, week=W).rename_axis("team").reset_index())
    return pd.concat(out, ignore_index=True)


def slot_rates(pe: dict) -> pd.DataFrame:
    out = []
    for s in SEAS:
        for W in WEEKS:
            p = CACHE / "receiving_summary" / str(s) / f"w01-{W - 1:02d}.json"
            if not p.exists():
                continue
            d = _num(pd.DataFrame(_rows(p)), ["slot_snaps", "pass_plays", "player_id"])
            d = d[d["pass_plays"] >= 20]
            d["espn_id"] = d["player_id"].map(lambda i: pe.get(int(i)) if pd.notna(i) else None)
            d = d.dropna(subset=["espn_id"])
            out.append(pd.DataFrame({"espn_id": d["espn_id"].astype(int), "slot": d["slot_snaps"] / d["pass_plays"], "season": s, "week": W}))
    return pd.concat(out, ignore_index=True).drop_duplicates(["espn_id", "season", "week"])


def build() -> pd.DataFrame:
    tm = teams()
    pe = pff_espn()
    raw = vs.rows()
    X = raw[raw["position"].isin(TESTS) & raw["season"].isin(SEAS) & raw["week"].isin(list(WEEKS))].copy()
    X["resid"] = X["points"] - X["espn_proj"]
    X = X.sort_values(["espn_id", "season", "week"])
    trail = raw.sort_values(["espn_id", "season", "week"]).groupby(["espn_id", "season"])["points"].transform(lambda s: s.shift(1).rolling(4, min_periods=3).mean())
    X["credit_y"] = X["espn_proj"] - trail.reindex(X.index)
    g = game_grades(tm)
    T = trailing(g, ["pass_rush_defense", "run_defense", "coverage_defense", "pass_block", "run_block"])
    off = T[["team", "season", "week", "pass_block", "run_block"]]
    dfn = T[["team", "season", "week", "pass_rush_defense", "run_defense", "coverage_defense"]].rename(columns={"team": "opp"})
    X = X.merge(off, on=["team", "season", "week"], how="left").merge(dfn, on=["opp", "season", "week"], how="left")
    C = coverage(tm).rename(columns={"team": "opp"})
    X = X.merge(C, on=["opp", "season", "week"], how="left")
    S = slot_rates(pe)
    X = X.merge(S, on=["espn_id", "season", "week"], how="left")
    X["slot"] = X["slot"].fillna(X.groupby(["season", "week", "position"])["slot"].transform("mean"))
    return derive(X)


OPP = ["pass_rush_defense", "run_defense", "coverage_defense", "cov_out", "cov_slot", "cov_all"]


def derive(X: pd.DataFrame) -> pd.DataFrame:
    X["pr_mm"] = X["pass_rush_defense"] - X["pass_block"]
    X["run_mm"] = X["run_defense"] - X["run_block"]
    X["opp_cov"] = X["coverage_defense"]
    X["cov_faced"] = X["slot"] * X["cov_slot"] + (1 - X["slot"]) * X["cov_out"]
    X["align_gap"] = X["cov_faced"] - X["cov_all"]
    return X


def shuffled(X: pd.DataFrame, rng) -> pd.DataFrame:
    """Placebo: every row gets a random other defense from the same season-week (its own offense features kept)."""
    D = X.drop_duplicates(["season", "week", "opp"]).set_index(["season", "week", "opp"])[OPP]
    m = {}
    for (s, w), g in D.groupby(level=[0, 1]):
        o = g.index.get_level_values(2).to_numpy()
        for a, b in zip(o, rng.permutation(o)):
            m[(s, w, a)] = (s, w, b)
    k = [m.get(t, t) for t in zip(X["season"], X["week"], X["opp"])]
    Y = X.copy()
    Y[OPP] = D.reindex(k)[OPP].to_numpy()
    return derive(Y)


def run(X: pd.DataFrame, boots: int = 1000, perms: int = 300) -> dict:
    res = {"n": int(len(X)), "tests": []}
    rng = np.random.default_rng(0)
    P = [shuffled(X, rng) for _ in range(perms)]
    for pos, sets in TESTS.items():
        for xc in sets:
            for sub, g in (("rostered", X[X["position"] == pos]), ("starters", X[(X["position"] == pos) & (X["espn_proj"] >= STARTER[pos])])):
                g = g.dropna(subset=xc + ["resid"]).reset_index(drop=True)
                sd = {c: float(g.groupby(["season", "week"])[c].std().mean()) for c in xc}
                b, lo, hi = mm.fit_prop(g, "resid", xc, boots)
                null = []
                for Y in P:
                    gp = Y[(Y["position"] == pos) & ((Y["espn_proj"] >= STARTER[pos]) if sub == "starters" else True)].dropna(subset=xc + ["resid"]).reset_index(drop=True)
                    null.append(mm.ols_prop(gp, "resid", xc))
                null = np.abs(np.array(null))
                gc = g.dropna(subset=["credit_y"]).reset_index(drop=True)
                bc = mm.ols_prop(gc, "credit_y", xc)
                for i, c in enumerate(xc):
                    t = {"pos": pos, "rows": sub, "feature": c, "with": xc, "n": int(len(g)), "mean_proj": float(g["espn_proj"].mean()),
                         "left_pct_per_sd": float(b[i] * sd[c]), "ci": [float(lo[i] * sd[c]), float(hi[i] * sd[c])],
                         "espn_moves_pct_per_sd": float(bc[i] * sd[c]), "per_season": {},
                         "perm_p": float((null[:, i] >= abs(b[i])).mean())}
                    for s in SEAS:
                        gs = g[g["season"] == s].reset_index(drop=True)
                        if len(gs) > 50:
                            t["per_season"][s] = float(mm.ols_prop(gs, "resid", xc)[i] * sd[c])
                    res["tests"].append(t)
                    print(f"{pos:2} {sub:8} {c:9} n={len(g):5} left over {100 * t['left_pct_per_sd']:+5.1f}% of proj per sd "
                          f"[{100 * t['ci'][0]:+5.1f},{100 * t['ci'][1]:+5.1f}] placebo p={t['perm_p']:.2f}  ESPN already moves {100 * t['espn_moves_pct_per_sd']:+5.1f}%  "
                          f"by season " + " ".join(f"{100 * v:+.1f}" for v in t["per_season"].values()))
    return res


def schema() -> None:
    for d in sorted(p for p in CACHE.iterdir() if p.is_dir()):
        f = next(iter(sorted(d.rglob("*.json"))), None)
        if f:
            r = _rows(f)
            print(d.name, f.relative_to(CACHE), len(r), sorted(r[0])[:60] if r else None)


def main() -> None:
    if sys.argv[1:] == ["schema"]:
        return schema()
    X = build()
    print("coverage:", {c: int(X[c].notna().sum()) for c in ("pr_mm", "run_mm", "opp_cov", "cov_all", "cov_faced", "slot")}, "of", len(X), "rows")
    res = run(X)
    out = CACHE / "report" / "matchup_pff.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
