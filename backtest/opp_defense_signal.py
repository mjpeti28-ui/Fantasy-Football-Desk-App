"""Does an in-season, opponent-adjusted read of the defense add to ESPN's weekly projection?

Defense rating = what the defense has allowed relative to ESPN's own projection for the players it
faced, using only that season's earlier weeks (so matchup quality of the opposing players is already
netted out: a defense that faced stars is judged against the stars' projections). Variants:
  pos_add    : points over ESPN projection allowed per player-game to this position (shrunk, k=15 games)
  pos_ratio  : same, as a ratio of projected points (multiplied back by this player's projection)
  all_ratio  : all four skill positions pooled (more data per defense, k=330 projected points)
  raw_pa     : raw points allowed to the position minus league average (the old, unadjusted DvP)
  prev_season: last season's all_ratio (only for the early weeks)
Main metric = persistence slope of (actual - ESPN) on the rating: 1 means the defense effect fully carries
over to next week, 0 means ESPN already has it or it is noise. Placebo = same code with the defense each
player faced shuffled within the week. Then leave-one-season-out regressions vs ESPN alone.
  python3 backtest/opp_defense_signal.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backtest import ecr_signal as es  # noqa: E402
from backtest import multi_season as ms  # noqa: E402
from backtest import usage_signal as us  # noqa: E402
from backtest import vegas_signal as vs  # noqa: E402

SK = ["QB", "RB", "WR", "TE"]
K_POS, K_ALL_PROJ = 15.0, 330.0
FEATS = ["pos_add", "pos_ratio_x", "all_ratio_x", "raw_pa", "prev_ratio_x"]


def build(x: pd.DataFrame, shuffle: bool = False, seed: int = 0) -> pd.DataFrame:
    x = x[x["position"].isin(SK)].copy().reset_index(drop=True)
    x["resid"] = x["points"] - x["espn_proj"]
    if shuffle:
        rng = np.random.default_rng(seed)
        x["opp"] = x.groupby(["season", "week"])["opp"].transform(lambda s: rng.permutation(s.to_numpy()))
    for f in FEATS:
        x[f] = np.nan
    x["n_prior_wks"] = 0
    # last season's pooled ratio per defense
    prev = {}
    for s, g in x.groupby("season"):
        t = g.groupby("opp")[["resid", "espn_proj"]].sum()
        prev[s] = (t["resid"] / (t["espn_proj"] + K_ALL_PROJ)).to_dict()
    for s, g in x.groupby("season"):
        for w in sorted(g["week"].unique()):
            idx = g.index[g["week"] == w]
            pr = g[g["week"] < w]
            nw = pr["week"].nunique()
            x.loc[idx, "n_prior_wks"] = nw
            pv = prev.get(s - 1, {})
            x.loc[idx, "prev_ratio_x"] = x.loc[idx, "opp"].map(pv).fillna(0.0) * x.loc[idx, "espn_proj"]
            if nw < 3:
                continue
            a = pr.groupby(["opp", "position"]).agg(sr=("resid", "sum"), n=("resid", "size"), sp=("points", "sum"), se=("espn_proj", "sum"))
            a["pos_add"] = a["sr"] / (a["n"] + K_POS)
            a["pos_ratio"] = a["sr"] / (a["se"] + K_ALL_PROJ / 2)
            lg = pr.groupby("position")["points"].mean()
            a["raw_pa"] = a["sp"] / (a["n"] + K_POS) - lg.reindex(a.index.get_level_values(1)).to_numpy() * a["n"] / (a["n"] + K_POS)
            b = pr.groupby("opp").agg(sr=("resid", "sum"), se=("espn_proj", "sum"))
            b["all_ratio"] = b["sr"] / (b["se"] + K_ALL_PROJ)
            key = list(zip(x.loc[idx, "opp"], x.loc[idx, "position"]))
            for c in ("pos_add", "pos_ratio", "raw_pa"):
                m = a[c].to_dict()
                v = pd.Series([m.get(k, 0.0) for k in key], index=idx)
                if c == "pos_ratio":
                    x.loc[idx, "pos_ratio_x"] = v * x.loc[idx, "espn_proj"]
                else:
                    x.loc[idx, c] = v
            x.loc[idx, "all_ratio_x"] = x.loc[idx, "opp"].map(b["all_ratio"]).fillna(0.0) * x.loc[idx, "espn_proj"]
    return x


def slope(g: pd.DataFrame, f: str, boots: int = 1000, seed: int = 1):
    g = g.dropna(subset=[f, "resid"])
    if len(g) < 50:
        return None
    # center both within season-week-position: ESPN is biased on average and the rating inherits that level
    grp = [g["season"], g["week"], g["position"]]
    X = (g[f] - g.groupby(grp)[f].transform("mean")).to_numpy(float)
    y = (g["resid"] - g.groupby(grp)["resid"].transform("mean")).to_numpy(float)
    g = g.reset_index(drop=True)
    b = float((X * y).sum() / (X * X).sum())
    cl = g.groupby(["season", "week"]).indices
    keys = list(cl)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(boots):
        pick = rng.integers(0, len(keys), len(keys))
        ii = np.concatenate([cl[keys[p]] for p in pick])
        bs.append((X[ii] * y[ii]).sum() / (X[ii] * X[ii]).sum())
    return b, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(g), float(X.std())


def quartile_gap(g: pd.DataFrame, f: str, boots: int = 1000, seed: int = 2):
    """Mean (actual - ESPN) facing the softest quarter of defenses minus facing the toughest quarter, by that week's rating."""
    g = g.dropna(subset=[f]).copy()
    g["r"] = g.groupby(["season", "week", "position"])[f].rank(pct=True)
    g = g[(g["r"] <= 0.25) | (g["r"] >= 0.75)]
    g["soft"] = (g["r"] >= 0.75).astype(int)  # high rating = allows more than ESPN expects
    cl = g.groupby(["season", "week"]).indices
    keys = list(cl)
    y, sft = g["resid"].to_numpy(float), g["soft"].to_numpy(int)
    def gap(ii):
        return y[ii][sft[ii] == 1].mean() - y[ii][sft[ii] == 0].mean()
    base = gap(np.arange(len(g)))
    rng = np.random.default_rng(seed)
    bs = [gap(np.concatenate([cl[keys[p]] for p in rng.integers(0, len(keys), len(keys))])) for _ in range(boots)]
    return base, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(g)


def loso(x: pd.DataFrame, cols: list[str]) -> pd.Series:
    pred = pd.Series(np.nan, index=x.index)
    for held in sorted(x["season"].unique()):
        for pos in SK:
            tr = x[(x["season"] != held) & (x["position"] == pos)]
            te = (x["season"] == held) & (x["position"] == pos)
            if len(tr) and te.any():
                bb = us.ols(tr[cols].to_numpy(float), tr["points"].to_numpy(float), np.ones(len(tr)))
                pred[te] = us.predict(bb, x.loc[te, cols].to_numpy(float))
    return pred


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    raw = vs.rows()
    print("rows", len(raw), "skill", int(raw["position"].isin(SK).sum()))
    x = build(raw)
    pl = build(raw, shuffle=True)
    for tag, d in (("REAL", x), ("PLACEBO (shuffled opponents)", pl)):
        print(f"\n== persistence slope of (actual - ESPN) on rating, weeks >= 4  [{tag}]")
        rows = []
        for f in FEATS:
            r = {"feature": f}
            for pos in SK + ["ALL"]:
                g = d[(d["n_prior_wks"] >= 3)] if pos == "ALL" else d[(d["n_prior_wks"] >= 3) & (d["position"] == pos)]
                if f == "prev_ratio_x":
                    g = d[(d["n_prior_wks"] >= 0) & (d["week"] <= 17)] if pos == "ALL" else d[(d["position"] == pos)]
                s = slope(g, f)
                r[pos] = "n/a" if s is None else f"{s[0]:+.2f} [{s[1]:+.2f},{s[2]:+.2f}]"
            rows.append(r)
        print(pd.DataFrame(rows).to_string(index=False))
    # spread of the rating late in the season, in points per player-game
    late = x[(x["week"] >= 8) & x["pos_add"].notna()]
    print("\nsd of rating across player-games, week>=8: pos_add %.2f pts, all_ratio_x %.2f pts, raw_pa %.2f pts" % (late["pos_add"].std(), late["all_ratio_x"].std(), late["raw_pa"].std()))
    print("sd of weekly residual (actual - ESPN): %.2f pts" % x["resid"].std())
    # by week bucket, pooled positions, pos_add and all_ratio_x
    print("\n== slope by week bucket (pooled QB/RB/WR/TE)")
    for lo, hi in ((4, 6), (7, 10), (11, 17)):
        g = x[(x["week"] >= lo) & (x["week"] <= hi)]
        print(f"weeks {lo}-{hi}:", {f: (lambda s: "n/a" if s is None else f"{s[0]:+.2f} [{s[1]:+.2f},{s[2]:+.2f}]")(slope(g, f)) for f in ("pos_add", "all_ratio_x", "raw_pa")})
    print("\n== softest quarter minus toughest quarter of defenses (by in-season rating): extra points vs ESPN projection per player-game")
    for tag, d in (("REAL", x), ("PLACEBO", pl)):
        out = {}
        for f in ("pos_add", "all_ratio_x", "raw_pa"):
            sub = d[d["n_prior_wks"] >= 3]
            q = quartile_gap(sub, f)
            out[f] = f"{q[0]:+.2f} [{q[1]:+.2f},{q[2]:+.2f}] n={q[3]}"
        print(tag, out)
    for pos in SK:
        q = quartile_gap(x[(x["n_prior_wks"] >= 3) & (x["position"] == pos)], "pos_add")
        print(f"  {pos} (pos_add): {q[0]:+.2f} [{q[1]:+.2f},{q[2]:+.2f}] n={q[3]}")
    # leave-one-season-out regression vs ESPN alone, weeks >= 4
    e = x[x["n_prior_wks"] >= 3].reset_index(drop=True)
    e["p_espn"] = loso(e, ["espn_proj"])
    for name, cols in {"+pos_add": ["espn_proj", "pos_add"], "+all_ratio": ["espn_proj", "all_ratio_x"], "+pos_ratio": ["espn_proj", "pos_ratio_x"],
                       "+raw_pa": ["espn_proj", "raw_pa"], "+pos_add+all_ratio": ["espn_proj", "pos_add", "all_ratio_x"]}.items():
        e["p_" + name] = loso(e, cols)
        m, lo, hi, sea = vs.boot_diff(e, "p_espn", "p_" + name)
        mae = (e["p_" + name] - e["points"]).abs().mean()
        print(f"{name:20s} rank-corr gain {m:+.4f} [{lo:+.4f},{hi:+.4f}] seasons {sea} | MAE {mae:.3f} vs espn-only {(e['p_espn'] - e['points']).abs().mean():.3f}")
