"""Does valuing upside (persistent level uncertainty) value individual roster spots better?

For each season (LOSO params), checkpoints 4/8/11, every team, every rostered player j:
  predicted marginal value = team_value(roster) - team_value(roster without j), weeks W..17,
     point model (levels off) and level model (levels on)
  realized marginal value  = realized lineup points with vs without j, weeks W..17, holes
     filled by the streamer actually available that week (stream target) and, separately,
     hindsight-best lineups
Grouped by the player's actual lineup slot at W (starter / bench) and depth role.
  python3 backtest/option_value.py 2021 [...]      -> option_value_<yr>.csv
  python3 backtest/option_value.py report
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import valuation as v  # noqa: E402
from backtest import currency_validation as cv  # noqa: E402
from backtest import multi_season as ms  # noqa: E402

OUT = ROOT / "history" / "backtest" / "multi"


def run(season: int, checkpoints=(4, 6, 8, 11), sims: int = 150) -> pd.DataFrame:
    rows = pd.read_csv(OUT / "player_rows.csv")
    p = ms.fit_all(rows, [y for y in ms.SEASONS if y != season])
    s = v.load_season(season)
    recs = []
    for W in checkpoints:
        ros_df = s.roster_at(W)
        slot = dict(zip(ros_df["espn_id"], ros_df["lineup_slot"]))
        rosters = cv.team_rosters(s, W)
        rostered = [i for ids in rosters.values() for i in ids]
        tbl = v.player_table(s, W, p, rostered=rostered)
        repl = v.waiver_replacement(tbl, rostered, scale=p.waiver_scale)
        st = pd.read_csv(OUT / "streamer_test.csv")
        st = st[st["season"] != season].groupby("position")
        mult = (st["stream2"].mean() / st["predicted"].mean()).to_dict()
        repl_sm = {k: x * mult.get(k, 1.0) for k, x in repl.items()}
        VAR = {"point": (repl, False), "level": (repl, True), "point_sm": (repl_sm, False), "level_sm": (repl_sm, True)}
        import os
        if os.environ.get("OV_ALT"):  # alternative level settings: slower recognition, half the sd
            VAR = {"point": (repl, False), "level_slow": (repl, "slow"), "level_half": (repl, "half")}
        weeks = range(W, v.FANTASY_LAST_WEEK + 1)
        roles = dict(zip(tbl.index, v.player_role(s, W, tbl.index)))
        for t, ids in rosters.items():
            ids = [i for i in ids if i in tbl.index]
            def tvv(pl, rp, lv):
                tt = tbl
                if lv == "slow":
                    tt = tbl.assign(learn_k=tbl["learn_k"] * 3)
                elif lv == "half":
                    tt = tbl.assign(level_sd=tbl["level_sd"] * 0.5)
                return v.team_value(pl, tt, s.slots, weeks, rp, sims=sims, seed=5, levels=bool(lv))
            base = {m: tvv(ids, rp, lv) for m, (rp, lv) in VAR.items()}
            real = {}
            for tgt in ("stream", "hindsight"):
                cv.TARGET = tgt
                real[tgt] = cv.realized_lineup_points(s, ids, weeks, repl)
            for j in ids:
                rest = [i for i in ids if i != j]
                rec = {"season": season, "W": W, "team": t, "espn_id": j, "position": tbl.at[j, "position"],
                       "role": roles.get(j), "slot": "bench" if slot.get(j) in ("BE", "IR") else "starter",
                       "est_ppg": tbl.at[j, "est_ppg"], "level_sd": tbl.at[j, "level_sd"]}
                for m, (rp, lv) in VAR.items():
                    rec["pred_" + m] = base[m] - tvv(rest, rp, lv)
                for tgt in ("stream", "hindsight"):
                    cv.TARGET = tgt
                    rec["real_" + tgt] = real[tgt] - cv.realized_lineup_points(s, rest, weeks, repl)
                recs.append(rec)
    return pd.DataFrame(recs)


def report():
    import os
    pat = "option_value_alt_*.csv" if os.environ.get("OV_ALT") else "option_value_2*.csv"
    d = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob(pat))], ignore_index=True)
    alts = [c[5:] for c in d.columns if c.startswith("pred_") and c != "pred_point"]
    pd.set_option("display.width", 220)
    print("players:", len(d), "| seasons", sorted(d["season"].unique()))
    for tgt in ("stream", "hindsight"):
        print(f"\n== realized target: {tgt} ==")
        rows = []
        for key, g in [("ALL", d)] + [(f"slot={k}", g) for k, g in d.groupby("slot")] + \
                      [(f"{a}/{b}", g) for (a, b), g in d.groupby(["slot", "role"])]:
            rec = {"group": key, "n": len(g), "real": g[f"real_{tgt}"].mean()}
            for m in ["point"] + alts:
                rec[f"pred_{m}"] = g[f"pred_{m}"].mean()
                rec[f"mae_{m}"] = (g[f"pred_{m}"] - g[f"real_{tgt}"]).abs().mean()
                rec[f"rho_{m}"] = np.nanmean([ms.spearman(h[f"pred_{m}"], h[f"real_{tgt}"])
                                              for _, h in g.groupby(["season", "W", "team"]) if len(h) > 3]) if len(g) > 50 else np.nan
            rows.append(rec)
        print(pd.DataFrame(rows).round(2).to_string(index=False))
        # paired: within team-checkpoint rank corr gain, and bench MAE gain, bootstrap over team-checkpoints
        per, mae = [], []
        for alt in alts:
            per, mae, allm = [], [], []
            for _, h in d.groupby(["season", "W", "team"]):
                per.append(ms.spearman(h[f"pred_{alt}"], h[f"real_{tgt}"]) - ms.spearman(h["pred_point"], h[f"real_{tgt}"]))
                allm.append(((h[f"pred_{alt}"] - h[f"real_{tgt}"]).abs() - (h["pred_point"] - h[f"real_{tgt}"]).abs()).mean())
                b = h[h["slot"] == "bench"]
                if len(b):
                    mae.append(((b[f"pred_{alt}"] - b[f"real_{tgt}"]).abs() - (b["pred_point"] - b[f"real_{tgt}"]).abs()).mean())
            rng = np.random.default_rng(0)
            for name, arr in (("rank corr gain", np.array(per)), ("MAE change (all)", np.array(allm)), ("MAE change (bench)", np.array(mae))):
                arr = arr[~np.isnan(arr)]
                bs = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(2000)]
                print(f"  {alt:9s} vs point, {name}: {arr.mean():+.3f} [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]")


if __name__ == "__main__":
    if sys.argv[1] == "report":
        report()
    else:
        for a in sys.argv[1:]:
            import os
            d = run(int(a))
            d.to_csv(OUT / (f"option_value_alt_{a}.csv" if os.environ.get("OV_ALT") else f"option_value_{a}.csv"), index=False)
            print(a, len(d))
