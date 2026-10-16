"""Availability model from injury reports, depth charts and absence streaks.

For every player-checkpoint (season, W) in usage_rows_*.csv we know the forecast inputs
and, afterwards, which of his remaining game weeks he actually played. We fit logistic
models for P(plays) of the next game and of each later game (with a horizon term),
leave-one-season-out, and compare against the current v2 availability.

Feature sets
  tue  : knowable when waivers run (Tue/Wed): status buckets, missed-games streak, share of
         games played so far, rostered, last week's injury report (status + body part),
         depth-chart rank.
  full : tue + ESPN's published 0 for this week + this week's (Friday) injury report and
         practice status -- what you know at lineup lock.

Metrics: Brier / log-loss per game (next game and later games), calibration by decile,
expected-vs-actual games (MAE), rest-of-season points rank corr, and a waiver replay.
  python3 backtest/availability.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backtest import multi_season as ms  # noqa: E402

OUT = ROOT / "history" / "backtest" / "multi"
NFLV = ROOT / "history" / "nflverse"
LAST = 17
BODY = {"knee": "knee", "ankle": "ankle", "hamstring": "hamstring", "concussion": "concussion",
        "foot": "foot", "achilles": "severe", "acl": "severe", "pectoral": "severe", "shoulder": "shoulder",
        "back": "other", "calf": "other", "groin": "other", "hip": "other"}
BODY_CATS = ["knee", "ankle", "hamstring", "concussion", "foot", "severe", "shoulder", "other"]


def _inj(season: int) -> pd.DataFrame:
    import nflverse_data as nd
    u = nd.usage_table(season)  # has report_status / practice_status joined on espn_id, week
    inj = pd.read_csv(NFLV / str(season) / "injuries.csv")
    ids = pd.read_csv(NFLV / "ids.csv").dropna(subset=["gsis_id", "espn_id"]).drop_duplicates("gsis_id")
    inj["espn_id"] = inj["gsis_id"].map(ids.set_index("gsis_id")["espn_id"])
    inj = inj.dropna(subset=["espn_id"]).drop_duplicates(["espn_id", "week"], keep="last")
    inj["espn_id"] = inj["espn_id"].astype("int64")
    inj["body"] = inj["report_primary_injury"].astype(str).str.lower().map(lambda s: next((v for k, v in BODY.items() if k in s), "other"))
    return inj[["espn_id", "week", "report_status", "practice_status", "body"]]


def features(season: int) -> pd.DataFrame:
    rows = pd.read_csv(OUT / f"usage_rows_{season}.csv")
    wk = pd.read_csv(ROOT / "history" / "actuals" / str(season) / "players_weekly.csv")
    wk = wk[(wk["kind"] == "actual") & wk["week"].between(1, LAST)]
    played = set(zip(wk.loc[wk["played"] == True, "espn_id"], wk.loc[wk["played"] == True, "week"]))  # noqa: E712
    inj = _inj(season)
    inj_key = {(int(r.espn_id), int(r.week)): r for r in inj.itertuples()}
    depth = pd.read_csv(NFLV / str(season) / "depth.csv").dropna(subset=["espn_id"])
    depth["espn_id"] = depth["espn_id"].astype("int64")
    dmin = depth.groupby(["espn_id", "week"])["depth"].min().to_dict()
    feats, games = [], []
    for r in rows.itertuples():
        pid, W, bye = int(r.espn_id), int(r.week), (int(r.bye) if pd.notna(r.bye) else -1)
        past = [w for w in range(1, W) if w != bye]
        streak = 0
        for w in reversed(past):
            if (pid, w) in played:
                break
            streak += 1
        n_past = len(past)
        frac = (sum((pid, w) in played for w in past) / n_past) if n_past else 1.0
        prev = inj_key.get((pid, W - 1)) if W - 1 != bye else inj_key.get((pid, W - 2))
        now = inj_key.get((pid, W))
        last_body = None
        for w in range(W - 1, 0, -1):
            x = inj_key.get((pid, w))
            if x is not None:
                last_body = x.body
                break
        dep = dmin.get((pid, W), dmin.get((pid, W - 1)))
        f = {
            "season": season, "week": W, "espn_id": pid, "position": r.position,
            "est_ppg": r.est_ppg, "rostered": float(r.rostered), "added_now": r.added_now,
            "future_points": r.future_points, "v2_total": r.v2_total, "espn_now": r.espn_now,
            "avail_now_v2": r.avail_now, "avail_v2": r.avail, "bye": bye, "name": r.name,
            "st_missed_last": float(r.status == "missed_last"), "st_ir": float(r.status == "ir"),
            "streak1": float(streak == 1), "streak2_3": float(2 <= streak <= 3), "streak4p": float(streak >= 4),
            "frac_played": frac, "no_games_yet": float(n_past > 0 and frac == 0),
            "prev_out": float(prev is not None and prev.report_status in ("Out", "Doubtful")),
            "prev_q": float(prev is not None and prev.report_status == "Questionable"),
            "depth1": float(dep == 1), "depth2": float(dep == 2), "depth3p": float(dep is not None and dep >= 3),
            "off_chart": float(dep is None),
            "espn0": float(r.espn_now == 0 and bye != W),
            "now_out": float(now is not None and now.report_status == "Out"),
            "now_doubt": float(now is not None and now.report_status == "Doubtful"),
            "now_q": float(now is not None and now.report_status == "Questionable"),
            "now_dnp": float(now is not None and isinstance(now.practice_status, str) and "Did Not" in now.practice_status),
            "now_ltd": float(now is not None and isinstance(now.practice_status, str) and "Limited" in now.practice_status),
        }
        recent = streak > 0 or f["prev_out"] or f["prev_q"] or f["now_out"] or f["now_doubt"] or f["now_q"]
        for c in BODY_CATS:
            f["body_" + c] = float(recent and last_body == c)
        feats.append(f)
        fut = [w for w in range(W, LAST + 1) if w != bye]
        for h, w in enumerate(fut):
            games.append((season, W, pid, h, float((pid, w) in played)))
    F = pd.DataFrame(feats)
    G = pd.DataFrame(games, columns=["season", "week", "espn_id", "h", "played"])
    return F, G


TUE = ["st_missed_last", "st_ir", "streak1", "streak2_3", "streak4p", "frac_played", "no_games_yet", "rostered",
       "prev_out", "prev_q", "depth1", "depth2", "depth3p", "off_chart"] + ["body_" + c for c in BODY_CATS]
FULL = TUE + ["espn0", "now_out", "now_doubt", "now_q", "now_dnp", "now_ltd"]
POS = ["pos_QB", "pos_RB", "pos_TE"]  # WR baseline


def logit_fit(X, y, w=None, l2=1.0, iters=30):
    X = np.column_stack([np.ones(len(X)), X])
    w = np.ones(len(y)) if w is None else w
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        g = X.T @ (w * (p - y)) + l2 * np.r_[0, b[1:]]
        H = (X * (w * p * (1 - p))[:, None]).T @ X + l2 * np.diag(np.r_[0, np.ones(len(b) - 1)])
        step = np.linalg.solve(H, g)
        b -= step
        if np.abs(step).max() < 1e-7:
            break
    return b


def logit_pred(b, X):
    X = np.column_stack([np.ones(len(X)), X])
    return 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))


def design(F: pd.DataFrame, G: pd.DataFrame, cols) -> tuple:
    D = G.merge(F, on=["season", "week", "espn_id"])
    for p in ("QB", "RB", "TE"):
        D["pos_" + p] = (D["position"] == p).astype(float)
    D["first"] = (D["h"] == 0).astype(float)
    D["logh"] = np.log1p(D["h"])
    base = cols + POS
    # separate intercept + slopes for the next game vs later games, plus decay with horizon
    X = np.column_stack([D[base].to_numpy(float) * D[["first"]].to_numpy(),
                         D[base].to_numpy(float) * (1 - D[["first"]].to_numpy()),
                         D["first"], D["logh"],
                         D[cols].to_numpy(float) * D[["logh"]].to_numpy()])
    return D, X


def loso(F, G, cols, relevant_min=3.0):
    D, X = design(F, G, cols)
    D["p"] = np.nan
    for held in sorted(D["season"].unique()):
        tr = (D["season"] != held) & (D["est_ppg"] >= relevant_min)
        b = logit_fit(X[tr.to_numpy()], D.loc[tr, "played"].to_numpy(float))
        te = (D["season"] == held).to_numpy()
        D.loc[te, "p"] = logit_pred(b, X[te])
    return D


def v2_game_probs(D):
    return np.where(D["h"] == 0, D["avail_now_v2"], D["avail_v2"])


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def logloss(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def report(D, label):
    rel = D[D["est_ppg"] >= 3]
    out = []
    for grp, g in (("all relevant", rel), ("injury-flagged", rel[(rel["st_missed_last"] + rel["st_ir"] + rel["prev_out"] + rel["prev_q"]
                                                                   + rel["now_out"] + rel["now_doubt"] + rel["now_q"] + rel["espn0"]) > 0])):
        for part, x in (("next game", g[g["h"] == 0]), ("later games", g[g["h"] > 0])):
            y = x["played"].to_numpy(float)
            out.append({"model": label, "group": grp, "part": part, "n": len(x),
                        "brier_v2": brier(v2_game_probs(x), y), "brier_new": brier(x["p"].to_numpy(), y),
                        "ll_v2": logloss(v2_game_probs(x), y), "ll_new": logloss(x["p"].to_numpy(), y)})
    return pd.DataFrame(out)


def calibration(D):
    rel = D[D["est_ppg"] >= 3].copy()
    rel["v2p"] = v2_game_probs(rel)
    out = {}
    for c in ("v2p", "p"):
        rel["bin"] = pd.cut(rel[c], [0, .2, .4, .6, .7, .8, .85, .9, .95, 1.0001], include_lowest=True)
        out[c] = rel.groupby("bin", observed=True).agg(pred=(c, "mean"), actual=("played", "mean"), n=("played", "size"))
    return out


def ros_eval(D, F):
    """Expected games -> rest-of-season points, compared with v2."""
    eg = D.groupby(["season", "week", "espn_id"])["p"].sum().rename("exp_games_new")
    eg_v2 = D.assign(v=v2_game_probs(D)).groupby(["season", "week", "espn_id"])["v"].sum().rename("exp_games_v2")
    act = D.groupby(["season", "week", "espn_id"])["played"].sum().rename("games")
    R = F.set_index(["season", "week", "espn_id"]).join([eg, eg_v2, act]).reset_index()
    R["pts_new"] = R["est_ppg"] * R["exp_games_new"]
    R["pts_v2"] = R["est_ppg"] * R["exp_games_v2"]
    rel = R[R["est_ppg"] >= 3]
    res = {"games MAE v2": float((rel["exp_games_v2"] - rel["games"]).abs().mean()),
           "games MAE new": float((rel["exp_games_new"] - rel["games"]).abs().mean()),
           "pts MAE v2": float((rel["pts_v2"] - rel["future_points"]).abs().mean()),
           "pts MAE new": float((rel["pts_new"] - rel["future_points"]).abs().mean())}
    for c in ("pts_v2", "pts_new"):
        res["ros rank corr " + c] = float(np.mean([ms.spearman(x[c], x["future_points"])
                                                  for _, x in rel.groupby(["season", "week", "position"]) if len(x) > 5]))
    return R, res


def waiver_replay(R, rule_cols):
    recs = []
    fa = R[R["rostered"] == 0]
    for (yr, W, pos), g in fa.groupby(["season", "week", "position"]):
        k = int(g["added_now"].sum())
        if k == 0:
            continue
        rec = {"season": yr, "k": k, "position": pos, "managers": g.loc[g["added_now"], "future_points"].mean()}
        for c in rule_cols:
            rec[c] = g.nlargest(k, c)["future_points"].mean()
        recs.append(rec)
    r = pd.DataFrame(recs)
    cols = ["managers"] + rule_cols
    w = r.assign(**{c: r[c] * r["k"] for c in cols})
    tot = w.groupby("position")[cols + ["k"]].sum()
    tot.loc["ALL"] = tot.sum()
    return r, tot[cols].div(tot["k"], axis=0).round(1)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    FS, GS = [], []
    for y in ms.SEASONS:
        f, g = features(y)
        FS.append(f); GS.append(g)
    F, G = pd.concat(FS, ignore_index=True), pd.concat(GS, ignore_index=True)
    F.to_csv(OUT / "avail_features.csv", index=False)
    print("rows", len(F), "game rows", len(G))
    results = {}
    for label, cols in (("tue", TUE), ("full", FULL)):
        D = loso(F, G, cols)
        print(report(D, label).round(4).to_string(index=False))
        cal = calibration(D)
        print("calibration v2:\n", cal["v2p"].round(3).to_string(), "\ncalibration new:\n", cal["p"].round(3).to_string())
        R, res = ros_eval(D, F)
        print({k: round(v, 3) for k, v in res.items()})
        R = R.rename(columns={"pts_new": f"pts_{label}"})
        results[label] = R
    R = results["tue"].merge(results["full"][["season", "week", "espn_id", "pts_full"]], on=["season", "week", "espn_id"])
    R.to_csv(OUT / "avail_ros.csv", index=False)
    r, tab = waiver_replay(R, ["v2_total", "pts_v2", "pts_tue", "pts_full"])
    print("\nWaiver replay (pts per pickup, W..17):\n", tab.to_string())
    r.to_csv(OUT / "avail_waiver_replay.csv", index=False)
