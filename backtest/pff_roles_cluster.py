"""Do PFF alignment/usage features make the Roles style clusters better? Aggregates only.

Unit: player-season 2021-25, weeks 1-12 (PFF w01-12 window + nflverse roles_weekly weeks 1-12), real-volume
filter from pff_explore.eligible (>= 6 games). Per position, three feature sets on the same rows:
  cur      R.STYLE features (what roles_fit uses today)
  cur+pff  cur plus PFF alignment/usage shape features
  pff      PFF features alone (plus PFF aDOT for receivers)
For each: k by BIC over 2-6 (full-cov GMM, pooled seasons so labels are comparable across years), then at k=3
and k=BIC:
  early    label of the w01-02 window (same scaler + GMM, hard predict) vs the w01-12 label: accuracy and kappa
  yoy      same label in consecutive seasons (w01-12): rate and kappa
  ari      adjusted Rand vs the shipped roles_model.json labels on the same rows
Optional comps check: add w01-(W-1) cluster probabilities from a PFF-features-only GMM to the comps matrix for 2021-25 rows and
rerun roles_fit.comps_backtest restricted to those seasons.

  ~/.local/share/desk/venv/bin/python backtest/pff_roles_cluster.py [--comps]
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app" / "engine"))
import roles as R  # noqa: E402
from backtest import pff_explore as pe  # noqa: E402
from backtest import pff_stats as ps  # noqa: E402

SEASONS = range(2021, 2026)
ROLES = ROOT / "app" / "cache.nosync" / "roles"
MODEL = json.loads((ROOT / "app" / "engine" / "roles_model.json").read_text())

# PFF shape features per position (clip ranges); wide_rate ~ 1 - slot - inline and pass_block_rate ~ 1 - route_rate
# on pass snaps, so the redundant member of each pair is left out (full-cov GMMs go singular otherwise).
PFF = {
    "WR": {"slot_rate": (0, 100), "route_rate": (50, 100)},
    "TE": {"inline_rate": (0, 100), "slot_rate": (0, 100), "route_rate": (20, 100)},
    "RB": {"route_rate": (0, 100), "tprr": (0, .5), "zone_sh": (0, 1)},
    "QB": {"ttt": (2, 3.6), "pff_scr": (0, .15)},
}
PFF_ONLY_EXTRA = {"WR": {"adot_pff": (-5, 30)}, "TE": {"adot_pff": (-5, 25)}, "RB": {}, "QB": {"adot_pass": (4, 12), "des_rush_pff": (0, 12)}}


def pff_window(y: int, lo: int, hi: int) -> pd.DataFrame:
    rec, rus, pas = (ps.load(f, y, lo, hi) for f in pe.FACETS)
    r, u, q = rec.set_index("player_id"), rus.set_index("player_id"), pas.set_index("player_id")
    out = pd.DataFrame({"slot_rate": r["slot_rate"], "inline_rate": r["inline_rate"], "wide_rate": r["wide_rate"],
                        "route_rate": r["route_rate"], "pass_block_rate": r["pass_block_rate"],
                        "tprr": r["targets"] / r["routes"].replace(0, np.nan), "adot_pff": r["avg_depth_of_target"]})
    out = out.join(pd.DataFrame({"zone_sh": u["zone_attempts"] / (u["zone_attempts"] + u["gap_attempts"]).replace(0, np.nan),
                                 "des_rush_pff": (u["attempts"] - u["scrambles"]) / u["player_game_count"]}), how="outer")
    out = out.join(pd.DataFrame({"ttt": q["avg_time_to_throw"], "pff_scr": q["scrambles"] / q["dropbacks"].replace(0, np.nan),
                                 "adot_pass": q["avg_depth_of_target"]}), how="outer")
    out["espn_id"] = out.index.map(pe.id_map(y))
    return out.dropna(subset=["espn_id"]).drop_duplicates("espn_id").set_index("espn_id")


def cur_window(w: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    a = R.aggregate(w[w.week.between(lo, hi)], ("season", "espn_id", "gsis", "pos"))
    return a.set_index(["season", "espn_id"])


def build() -> dict[str, pd.DataFrame]:
    """{'full': w01-12 rows (filtered), 'early': w01-02 rows for the same player-seasons}."""
    w = pd.concat([pd.read_csv(ROLES / str(y) / "roles_weekly.csv") for y in SEASONS], ignore_index=True)
    w = w[w.pos.isin(["QB", "RB", "WR", "TE"]) & w.espn_id.notna()]
    full, early = [], []
    for y in SEASONS:
        t = pe.window_table(y, 1, 12)
        keep = t[pe.eligible(t, 6)].dropna(subset=["espn_id"]).drop_duplicates("espn_id")[["espn_id", "pos"]].rename(columns={"pos": "pff_pos"})
        for lo, hi, dst in ((1, 12, full), (1, 2, early)):
            c = cur_window(w[w.season == y], lo, hi).reset_index()
            p = pff_window(y, lo, hi)
            d = c.merge(keep, on="espn_id").join(p, on="espn_id")
            dst.append(d[d.pos == d.pff_pos])
    return {"full": pd.concat(full, ignore_index=True), "early": pd.concat(early, ignore_index=True)}


def sets(pos: str) -> dict[str, dict]:
    cur = dict(R.STYLE[pos])
    return {"cur": cur, "cur+pff": {**cur, **PFF[pos]}, "pff": {**PFF[pos], **PFF_ONLY_EXTRA[pos]}}


def prep(d: pd.DataFrame, f: dict) -> pd.DataFrame:
    d = d.dropna(subset=list(f)).copy()
    for c, (a, b) in f.items():
        d[c] = d[c].clip(a, b)
    return d


def shipped_labels(pos: str, d: pd.DataFrame) -> np.ndarray:
    m = MODEL["style"][pos]
    z = (d[m["features"]].clip(lower=[R.STYLE[pos][c][0] for c in m["features"]], upper=[R.STYLE[pos][c][1] for c in m["features"]], axis=1).values
         - np.array(m["mean"])) / np.array(m["scale"])
    lp = np.column_stack([np.log(wt) + multivariate_normal(mu, cov, allow_singular=True).logpdf(z)
                          for wt, mu, cov in zip(m["weights"], m["means"], m["covs"])])
    return lp.argmax(1)


def boot_se(a, b, B: int = 300) -> float:
    """Bootstrap SE of kappa over rows (labels held fixed; ignores refit noise, so a floor on the real SE)."""
    rng = np.random.default_rng(0); a, b = np.asarray(a), np.asarray(b)
    return float(np.std([cohen_kappa_score(a[i], b[i]) for i in (rng.integers(0, len(a), len(a)) for _ in range(B))]))


def evaluate(pos: str, name: str, f: dict, F: pd.DataFrame, E: pd.DataFrame, rows: list, cents: dict, ship: pd.Series):
    cols = list(f)
    d = prep(F[F.pos == pos], f)
    mu, sd = d[cols].mean().values, d[cols].std(ddof=0).values
    X = (d[cols].values - mu) / sd
    bic = {k: GaussianMixture(k, n_init=5, random_state=0, covariance_type="full").fit(X).bic(X) for k in range(2, 7)}
    kb = min(bic, key=bic.get)
    e = prep(E[E.pos == pos], f).merge(d[["season", "espn_id"]], on=["season", "espn_id"])
    for k in sorted({3, kb}):
        gm = GaussianMixture(k, n_init=10, random_state=0, covariance_type="full").fit(X)
        lab = pd.Series(gm.predict(X), index=pd.MultiIndex.from_frame(d[["season", "espn_id"]]))
        el = gm.predict((e[cols].values - mu) / sd)
        ft = lab.reindex(pd.MultiIndex.from_frame(e[["season", "espn_id"]])).values
        nx = lab.copy(); nx.index = nx.index.set_levels(nx.index.levels[0] - 1, level=0)
        j = pd.concat([lab.rename("a"), nx.rename("b")], axis=1).dropna()
        sh = ship.reindex(lab.index)
        rows.append({"pos": pos, "set": name, "k": k, "bic_k": kb, "n": len(d), "early_acc": np.mean(el == ft),
                     "early_kappa": cohen_kappa_score(el, ft), "early_kappa_se": boot_se(el, ft), "n_early": len(e), "yoy_same": np.mean(j.a == j.b),
                     "yoy_kappa": cohen_kappa_score(j.a, j.b), "yoy_kappa_se": boot_se(j.a.values, j.b.values), "n_yoy": len(j),
                     "ari_shipped": adjusted_rand_score(sh[sh.notna()], lab[sh.notna()]),
                     "chance": float((np.bincount(lab, minlength=k) / len(lab)) @ (np.bincount(lab, minlength=k) / len(lab)))})
        c = d.assign(cl=lab.values).groupby("cl")[cols].mean()
        c["share"] = d.assign(cl=lab.values).cl.value_counts(normalize=True).sort_index()
        c["ppg"] = d.assign(cl=lab.values).groupby("cl").ppr.sum() / d.assign(cl=lab.values).groupby("cl").games.sum()
        c["shipped_mode"] = [pd.Series(MODEL["style"][pos]["names"])[sh[lab.values == i].dropna().astype(int)].mode().iat[0]
                             if sh[lab.values == i].notna().any() else "" for i in c.index]
        cents[(pos, name, k)] = c


def main():
    D = build()
    F, E = D["full"], D["early"]
    print("rows w01-12:", F.groupby("pos").size().to_dict(), " early:", E.groupby("pos").size().to_dict())
    rows, cents = [], {}
    for pos in ["WR", "TE", "RB", "QB"]:
        d0 = prep(F[F.pos == pos], R.STYLE[pos])
        ship = pd.Series(shipped_labels(pos, d0), index=pd.MultiIndex.from_frame(d0[["season", "espn_id"]]))
        for name, f in sets(pos).items():
            evaluate(pos, name, f, F, E, rows, cents, ship)
    T = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(T.round(3).to_string(index=False))
    for (pos, name, k), c in cents.items():
        if name != "pff" or k == 3:
            print(f"\n{pos} {name} k={k}\n{c.round(3).to_string()}")
    if "--comps" in sys.argv:
        comps(F)
    return T, cents


# ------------------------------------------------------------------ optional: comps backtest with PFF probabilities

def comps(F: pd.DataFrame):
    import comps as C
    import roles_fit as RF
    L = pd.read_csv(ROOT / "app" / "engine" / "roles_library.csv.gz")
    L = L[L.season.isin(SEASONS)].copy()
    ids = pd.read_csv(ROOT / "history" / "nflverse" / "ids.csv").dropna(subset=["gsis_id", "espn_id"])
    L["espn_id"] = L.gsis.map(ids.drop_duplicates("gsis_id").set_index("gsis_id").espn_id)
    base = RF.comps_backtest(L)
    # cluster probabilities from a cur+pff GMM fit on w01-12, applied to the w01-(W-1) PFF window for each checkpoint
    add = []
    for pos in ["WR", "TE", "RB", "QB"]:
        f = {**PFF[pos]}
        d = prep(F[F.pos == pos], f); cols = list(f)
        mu, sd = d[cols].mean().values, d[cols].std(ddof=0).values
        gm = GaussianMixture(3, n_init=10, random_state=0, covariance_type="full").fit((d[cols].values - mu) / sd)
        for y in SEASONS:
            for W in (3, 5, 8, 11):
                p = prep(pff_window(y, 1, W - 1).reset_index(), f)
                pr = gm.predict_proba((p[cols].values - mu) / sd)
                add.append(pd.DataFrame({"season": y, "W": W, "espn_id": p.espn_id.values, "pos": pos,
                                         **{f"pp{i}": pr[:, i] for i in range(3)}}))
    A = pd.concat(add)
    L2 = L.merge(A, on=["season", "W", "espn_id", "pos"], how="left")
    for i in range(3):
        L2[f"pp{i}"] = L2[f"pp{i}"].fillna(1 / 3)
    orig = C.comp_matrix

    def cm(d, pos, wt=1.0):
        d, cols, wts = orig(d, pos)
        return d, cols + ["pp0", "pp1", "pp2"], np.r_[wts, [wt] * 3]
    out = {}
    for wt in (0.5, 1.0):
        RF.comp_matrix = lambda d, pos, wt=wt: cm(d, pos, wt)
        out[wt] = RF.comps_backtest(L2)
    RF.comp_matrix = orig
    print("\ncomps 2021-25 base:", {p: (v["mae"], v["rho"]) for p, v in base.items()})
    for wt, r in out.items():
        print(f"comps 2021-25 +pff probs w={wt}:", {p: (v["mae"], v["rho"]) for p, v in r.items()})


if __name__ == "__main__":
    main()
