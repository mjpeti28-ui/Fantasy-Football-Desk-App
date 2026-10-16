"""Do our forecasts beat ESPN's own weekly projections (what managers see in the app)?
Leave-one-season-out, rostered players, checkpoints weeks 3-10, 2021-2025."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import valuation as v  # noqa: E402
from backtest import multi_season as ms  # noqa: E402

OUT = ROOT / "history" / "backtest" / "multi"


def rho(a, b):
    a, b = pd.Series(list(a)), pd.Series(list(b))
    ok = a.notna() & b.notna()
    return float(a[ok].rank().corr(b[ok].rank())) if ok.sum() > 4 else np.nan


def build() -> pd.DataFrame:
    rows = pd.read_csv(OUT / "player_rows.csv")
    parts = []
    for yr in ms.SEASONS:
        p = ms.fit_params(rows[rows["season"] != yr])
        r = rows[rows["season"] == yr].drop(columns=["espn_now"], errors="ignore").copy()
        k = r["position"].map(p.k)
        prior = r["prior_ppg_raw"] * r["position"].map(p.prior_scale)
        r["ours_ppg"] = (k * prior + r["n"] * r["obs_ppg"]) / (k + r["n"])
        r["ours_now"] = r["ours_ppg"] * [p.availability[a]["now"][b] for a, b in zip(r["position"], r["status"])]
        proj = v.load_season(yr).espn_weekly  # cleaned (drops 2023 week 1)
        r = r.merge(proj.rename(columns={"espn_proj": "espn_now"}), on=["week", "espn_id"], how="left")
        # ESPN's per-game view as of the checkpoint: this week's projection (its only
        # no-look-ahead number); 0 when ESPN already knows he's out
        wk = v.load_season(yr).weekly
        nxt = wk[["week", "espn_id", "points", "played"]].rename(columns={"points": "next_points", "played": "next_played"})
        r = r.merge(nxt, on=["week", "espn_id"], how="left")
        parts.append(r)
    return pd.concat(parts, ignore_index=True)


def main():
    d = build()
    d = d[d["position"].isin(["QB", "RB", "WR", "TE"])]
    out = ["== Next week's points (all rostered players with a game that week; DNP = 0)"]
    nw = d.dropna(subset=["espn_now", "next_points"])
    res = []
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        pred = a * nw["ours_now"] + (1 - a) * nw["espn_now"]
        per = [rho(pred[g.index], g["next_points"]) for _, g in nw.groupby(["season", "week", "position"]) if len(g) > 5]
        res.append({"weight_on_ours": a, "rho": np.nanmean(per), "mae": float(np.abs(pred - nw["next_points"]).mean())})
    out.append(pd.DataFrame(res).round(3).to_string(index=False))
    out.append("")
    out.append("== Rest-of-season points per game when playing (future_ppg)")
    ros = d.dropna(subset=["espn_now", "future_ppg"])
    # ESPN projection is 0 for players already ruled out; use it only when > 0 for a per-game rate
    ros = ros[ros["espn_now"] > 0]
    res = []
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        pred = a * ros["ours_ppg"] + (1 - a) * ros["espn_now"]
        per = [rho(pred[g.index], g["future_ppg"]) for _, g in ros.groupby(["season", "week", "position"]) if len(g) > 5]
        res.append({"weight_on_ours": a, "rho": np.nanmean(per), "mae": float(np.abs(pred - ros["future_ppg"]).mean())})
    out.append(pd.DataFrame(res).round(3).to_string(index=False))
    out.append("")
    out.append("== by position (rest-of-season ppg rank corr): ours vs ESPN vs 50/50")
    rp = []
    for pos, g in ros.groupby("position"):
        f = lambda col: np.nanmean([rho(gg[col], gg["future_ppg"]) for _, gg in g.groupby(["season", "week"]) if len(gg) > 5])
        g = g.assign(blend=0.5 * g["ours_ppg"] + 0.5 * g["espn_now"])
        rp.append({"pos": pos, "ours": f("ours_ppg"), "espn": f("espn_now"), "blend": f("blend"), "n": len(g)})
    out.append(pd.DataFrame(rp).round(3).to_string(index=False))
    out.append("")
    out.append("== by season (ROS ppg rank corr, skill positions)")
    rs = []
    for yr, g in ros.groupby("season"):
        f = lambda col: np.nanmean([rho(gg[col], gg["future_ppg"]) for _, gg in g.groupby(["week", "position"]) if len(gg) > 5])
        g = g.assign(blend=0.5 * g["ours_ppg"] + 0.5 * g["espn_now"])
        rs.append({"season": yr, "ours": f("ours_ppg"), "espn": f("espn_now"), "blend": f("blend")})
    out.append(pd.DataFrame(rs).round(3).to_string(index=False))
    text = "\n".join(out)
    print(text)
    (OUT / "espn_vs_ours.txt").write_text(text)


if __name__ == "__main__":
    main()
