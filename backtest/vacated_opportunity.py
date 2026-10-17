"""When a starter misses a game, who inherits his work -- and does knowing it add anything?

Event: team T plays week W without a "starter" S who played >= 2 of the team's previous 4 games:
  RB starter      = carry share >= 0.35 over those games (vacates carries; targets ride along)
  WR/TE starter   = target share >= 0.18
Beneficiaries: teammates who played in W and in >= 2 of S's reference games. Their "base" is their
average share / ppr in the reference games S played.

Inheritance rules (each predicts the beneficiary's share change in W, ex ante):
  prorata  : vacated share split pro-rata to teammates' base shares (all eligible teammates)
  samepos  : same, but only teammates at S's position (RB->RB; WR->WR; TE->TE)
  depth    : all of it to the top same-position teammate on that week's depth chart (2021+)
A rule's absorption factor k (what fraction of the vacated share the group soaks up) is fit
leave-one-season-out, so nothing is tuned on the season it is scored on.

Fantasy: predicted ppr bump = predicted share change x team opportunities x the beneficiary's own
points per opportunity. Scored against (a) "no change" and (b) ESPN's own weekly projection for that
week (2021-25), which already knows S is out. The question that matters for the Desk is whether the
bump adds anything to ESPN's number, and how often the beneficiary was on waivers in Max's league.

    python backtest/vacated_opportunity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ROLES = ROOT / "app" / "cache.nosync" / "roles"
SEASONS = list(range(2016, 2026))
ESPN_SEASONS = list(range(2021, 2026))
RB_SHARE, REC_SHARE = 0.35, 0.18
GROUP = {"RB": ("car", "t_car"), "WR": ("tgt", "t_tgt"), "TE": ("tgt", "t_tgt")}


def load_roles(season: int) -> pd.DataFrame:
    d = pd.read_csv(ROLES / str(season) / "roles_weekly.csv")
    d["season"] = season
    d = d[d["pos"].isin(["RB", "WR", "TE"]) & (d["snap"] > 0)].copy()
    d["opp_n"] = np.where(d["pos"] == "RB", d["car"] + d["tgt"], d["tgt"])
    return d


def events(d: pd.DataFrame) -> pd.DataFrame:
    """One row per (starter out, beneficiary) pair."""
    rows = []
    team_weeks = d.groupby("posteam")["week"].apply(lambda s: sorted(set(s)))
    by_tw = {k: g for k, g in d.groupby(["posteam", "week"])}
    for team, weeks in team_weeks.items():
        for i, W in enumerate(weeks):
            if i < 3:
                continue
            ref_weeks = weeks[max(0, i - 4):i]
            ref = pd.concat([by_tw[(team, w)] for w in ref_weeks])
            now = by_tw[(team, W)]
            played_now = set(now["gsis"])
            for gsis, g in ref.groupby("gsis"):
                pos = g["pos"].iloc[0]
                if gsis in played_now or len(g) < 2 or pos not in GROUP:
                    continue
                n, t = GROUP[pos]
                share = g[n].sum() / max(g[t].sum(), 1)
                if share < (RB_SHARE if pos == "RB" else REC_SHARE):
                    continue
                s_weeks = set(g["week"])
                mates = ref[ref["week"].isin(s_weeks) & (ref["gsis"] != gsis)]
                for m, mg in mates.groupby("gsis"):
                    if m not in played_now or len(mg) < 2:
                        continue
                    mpos = mg["pos"].iloc[0]
                    if pos == "RB" and mpos != "RB":
                        continue  # carries only matter for backs
                    nr = now[now["gsis"] == m].iloc[0]
                    base_share = mg[n].sum() / max(mg[t].sum(), 1)
                    rows.append({
                        "season": int(g["season"].iloc[0]), "week": W, "team": team,
                        "out_gsis": gsis, "out_pos": pos, "out_name": g["pname"].dropna().iloc[0] if g["pname"].notna().any() else gsis,
                        "vacated": share, "gsis": m, "pos": mpos, "name": nr["pname"], "espn_id": nr["espn_id"],
                        "base_share": base_share, "now_share": nr[n] / max(nr[t], 1),
                        "team_opps": mg[t].mean(), "base_ppr": mg["ppr"].mean(), "now_ppr": nr["ppr"],
                        "ppo": mg["ppr"].sum() / max(mg["opp_n"].sum(), 1), "base_opp_n": mg["opp_n"].mean(),
                    })
    return pd.DataFrame(rows)


def add_rules(x: pd.DataFrame, depth: pd.DataFrame | None) -> pd.DataFrame:
    x = x.copy()
    key = ["season", "week", "team", "out_gsis"]
    x["w_all"] = x["base_share"] / x.groupby(key)["base_share"].transform("sum").clip(lower=1e-9)
    same = x["pos"] == x["out_pos"]
    s_sum = x["base_share"].where(same, 0).groupby([x[k] for k in key]).transform("sum")
    x["w_same"] = np.where(same, x["base_share"] / s_sum.clip(lower=1e-9), 0.0)
    x["w_same"] = x["w_same"].fillna(0.0)
    x["w_depth"] = 0.0
    if depth is not None:
        dd = depth.rename(columns={"gsis_id": "gsis", "team": "team"})[["season", "week", "team", "gsis", "position", "depth"]]
        x = x.merge(dd[["season", "week", "team", "gsis", "depth"]], on=["season", "week", "team", "gsis"], how="left")
        top = x[same & x["depth"].notna()].sort_values("depth").groupby(key).head(1).index
        x.loc[top, "w_depth"] = 1.0
    x["realized_delta"] = x["now_share"] - x["base_share"]
    return x


def fit_k(tr: pd.DataFrame, wcol: str) -> float:
    """Least-squares absorption factor: realized_delta ~ k * vacated * w."""
    f = tr["vacated"] * tr[wcol]
    m = f > 0
    return float((f[m] * tr.loc[m, "realized_delta"]).sum() / max((f[m] ** 2).sum(), 1e-9))


def spearman(a, b) -> float:
    return float(pd.Series(np.asarray(a, float)).rank().corr(pd.Series(np.asarray(b, float)).rank()))


def main() -> int:
    pd.set_option("display.width", 200)
    raw = pd.concat([load_roles(s) for s in SEASONS], ignore_index=True)
    x = pd.concat([events(raw[raw["season"] == s]) for s in SEASONS], ignore_index=True)
    depth = []
    for s in ESPN_SEASONS:
        p = ROOT / "history" / "nflverse" / str(s) / "depth.csv"
        if p.exists():
            depth.append(pd.read_csv(p).assign(season=s))
    x = add_rules(x, pd.concat(depth, ignore_index=True) if depth else None)
    n_ev = x.groupby(["season", "week", "team", "out_gsis"]).ngroups
    print(f"{n_ev} starter-out games, {len(x)} beneficiary rows, seasons {SEASONS[0]}-{SEASONS[-1]}\n")

    rules = {"prorata": "w_all", "samepos": "w_same", "depth": "w_depth"}
    for name, w in rules.items():
        x[f"pred_{name}"] = np.nan
    for held in SEASONS:
        tr, te = x["season"] != held, x["season"] == held
        for name, w in rules.items():
            if name == "depth" and held not in ESPN_SEASONS:
                continue
            trr = x[tr & (x["season"].isin(ESPN_SEASONS) if name == "depth" else tr)]
            k = fit_k(trr, w)
            x.loc[te, f"pred_{name}"] = k * x.loc[te, "vacated"] * x.loc[te, w]

    print("Share inheritance: corr(pred, realized share change) by starter position")
    rows = []
    for (pos,), g in x.groupby(["out_pos"]):
        r = {"out_pos": pos, "n": len(g)}
        for name in rules:
            gg = g.dropna(subset=[f"pred_{name}"])
            r[name] = round(float(np.corrcoef(gg[f"pred_{name}"], gg["realized_delta"])[0, 1]), 3) if len(gg) > 10 else np.nan
        r["mean realized delta"] = round(float(g["realized_delta"].mean()), 3)
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))

    # points per opportunity, shrunk toward the position average with 20 pseudo-opportunities
    pos_ppo = x.groupby("pos").apply(lambda g: g["base_ppr"].sum() / max(g["base_opp_n"].sum(), 1), include_groups=False)
    x["ppo"] = (x["base_ppr"] + 20 * x["pos"].map(pos_ppo)) / (x["base_opp_n"] + 20)
    best = "samepos"
    x["pred_bump"] = x[f"pred_{best}"] * x["team_opps"] * x["ppo"]
    x["real_bump"] = x["now_ppr"] - x["base_ppr"]
    print(f"\nFantasy bump ({best} rule): realized ppr - base ppr, all seasons")
    for pos, g in x.groupby("out_pos"):
        q = pd.qcut(g["pred_bump"], 4, duplicates="drop")
        t = g.groupby(q, observed=True)["real_bump"].agg(["count", "mean"]).round(2)
        print(f"  {pos} starter out -> beneficiaries by predicted-bump quartile:\n{t.to_string()}")

    # vs ESPN's weekly projection (it already knows the starter is out)
    pr = []
    for s in ESPN_SEASONS:
        p = pd.read_csv(ROOT / "history" / "actuals" / str(s) / "espn_weekly_proj.csv").assign(season=s)
        pr.append(p)
    pr = pd.concat(pr)
    e = x[x["season"].isin(ESPN_SEASONS)].merge(pr, on=["season", "week", "espn_id"], how="inner")
    e = e[e["espn_proj"] > 0]
    print(f"\nvs ESPN weekly projection: {len(e)} beneficiary rows with an ESPN number (2021-25)")
    e["base_plus_bump"] = e["base_ppr"] + e["pred_bump"]
    out = []
    for held in ESPN_SEASONS:
        tr, te = e[e["season"] != held], e[e["season"] == held]
        X = np.c_[np.ones(len(tr)), tr["espn_proj"], tr["pred_bump"]]
        b = np.linalg.lstsq(X, tr["now_ppr"], rcond=None)[0]
        X0 = np.c_[np.ones(len(tr)), tr["espn_proj"]]
        b0 = np.linalg.lstsq(X0, tr["now_ppr"], rcond=None)[0]
        p1 = np.c_[np.ones(len(te)), te["espn_proj"], te["pred_bump"]] @ b
        p0 = np.c_[np.ones(len(te)), te["espn_proj"]] @ b0
        out.append({"season": held, "n": len(te), "MAE espn": np.abs(p0 - te["now_ppr"]).mean(),
                    "MAE espn+bump": np.abs(p1 - te["now_ppr"]).mean(), "MAE base+bump": np.abs(te["base_plus_bump"] - te["now_ppr"]).mean(),
                    "rho espn": spearman(p0, te["now_ppr"]), "rho espn+bump": spearman(p1, te["now_ppr"]), "bump coef": b[2]})
    o = pd.DataFrame(out).round(3)
    print(o.to_string(index=False))
    print("mean:", o.drop(columns=["season"]).mean().round(3).to_dict())

    # waiver relevance in Max's league
    ro = []
    for s in ESPN_SEASONS:
        r = pd.read_csv(ROOT / "history" / "actuals" / str(s) / "rosters_by_week.csv").assign(season=s)
        ro.append(r[["season", "week", "espn_id"]])
    ro = pd.concat(ro).drop_duplicates()
    ro["rostered"] = True
    w = x[x["season"].isin(ESPN_SEASONS)].copy()
    w["prev_week"] = w["week"] - 1
    w = w.merge(ro.rename(columns={"week": "prev_week"}), on=["season", "prev_week", "espn_id"], how="left")
    w["rostered"] = w["rostered"].fillna(False).astype(bool)
    big = w[w["pred_bump"] >= 3]
    print(f"\nBeneficiaries with predicted bump >= 3 ppr (2021-25): {len(big)}; on waivers the week before: "
          f"{(~big['rostered']).sum()}; their realized bump: rostered {big.loc[big['rostered'], 'real_bump'].mean():.2f}, "
          f"waivers {big.loc[~big['rostered'], 'real_bump'].mean():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
