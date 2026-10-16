"""Matchups: how a player has been used, what coverages and defenses he sees, and what each defense
and its players contribute. Descriptive, never feeds the valuation (the opponent-defense backtest of
2026-10-10 found nothing beyond ESPN's number: claude/opp-defense-findings-2026-10.md).

Data: matchups_data.py (nflverse play-by-play, FTN charting, FTN participation; see the licences there).
Honesty rules baked in:
  * coverage tags exist for 2023-25 only (published after each season), so splits by coverage pool those
    seasons; the current season contributes blitz / play-action / depth / usage from play-by-play and FTN.
  * every split carries its sample size and a reliability (the share of the player-to-player spread in
    that split that is signal, measured from a split-half check); noisy splits are shrunk toward the
    position average and labelled.
"""
from __future__ import annotations

import datetime as dt
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from . import matchups_data as md

ROOT = Path(__file__).resolve().parents[2]
HIST = md.HISTORY
SHELL = {"COVER_0": "C01", "COVER_1": "C01", "COVER_2": "C2", "2_MAN": "C2", "COVER_3": "C3", "COVER_4": "C46", "COVER_6": "C46"}
SHELL_LABEL = {"C01": "Cover 0/1", "C2": "Cover 2", "C3": "Cover 3", "C46": "Cover 4/6", "OTH": "Other"}
SHELLS = ["C01", "C2", "C3", "C46", "OTH"]
POSITIONS = ["QB", "RB", "WR", "TE"]
ROUTE_LABEL = {"GO": "Go", "HITCH/CURL": "Hitch/curl", "IN/DIG": "In/dig", "POST": "Post", "CORNER": "Corner", "QUICK OUT": "Quick out",
               "DEEP OUT": "Deep out", "SHALLOW CROSS/DRAG": "Shallow cross", "SLANT": "Slant", "SWING": "Swing", "SCREEN": "Screen",
               "TEXAS/ANGLE": "Texas/angle", "WHEEL": "Wheel", "FLAT": "Flat", "ANGLE": "Angle"}
K_ALLOWED = 40.0  # targets of prior when judging what a defense allows to a position
_lock = threading.RLock()
_C: dict = {}


def current_season() -> int:
    t = dt.date.today()
    return t.year if t.month >= 3 else t.year - 1


def _ids() -> pd.DataFrame:
    i = pd.read_csv(ROOT / "history" / "nflverse" / "ids.csv").dropna(subset=["gsis_id"]).drop_duplicates("gsis_id").set_index("gsis_id")
    i["espn_id"] = pd.to_numeric(i["espn_id"], errors="coerce")
    return i


def _depth(a: pd.Series) -> pd.Series:
    return pd.cut(a, [-100, 0, 9, 19, 100], labels=["behind", "short", "mid", "deep"])


def _ppr(d: pd.DataFrame) -> pd.Series:
    return d["complete_pass"].fillna(0) * (1 + 0.1 * d["yards_gained"].fillna(0)) + 6 * d["pass_touchdown"].fillna(0)


def _stamp() -> tuple:
    d = md.cache_dir()
    return tuple((p.name, int(p.stat().st_mtime)) for p in sorted(d.glob("*.csv.gz")))


def tables() -> dict:
    """Plays, on-field tables and league baselines; rebuilt only when the cached season files change."""
    with _lock:
        key = _stamp()
        if _C.get("key") == key and _C.get("t"):
            return _C["t"]
        cur = current_season()
        ids = _ids()
        hist = []
        for y in HIST:
            d = md.load(y)
            d = d[d["db"] == 1].copy()
            d["pidx"] = np.arange(len(d)) + y * 1_000_000
            hist.append(d)
        H = pd.concat(hist, ignore_index=True)
        H["man"] = np.where(H["defense_man_zone_type"].isna(), np.nan, (H["defense_man_zone_type"] == "MAN_COVERAGE").astype(float))
        H["shell"] = H["defense_coverage_type"].map(SHELL).where(H["defense_coverage_type"].isna(), H["defense_coverage_type"].map(SHELL).fillna("OTH"))
        H["blitz"] = (H["n_blitzers"] >= 1).astype(float)
        H["ppr"] = _ppr(H)
        C = md.load(cur)
        C["blitz"] = (C["n_blitzers"] >= 1).astype(float)
        C["ppr"] = _ppr(C)
        C["pos"] = C["receiver"].map(ids["position"])
        C["rpos"] = C["rusher"].map(ids["position"])
        H["pos"] = H["receiver"].map(ids["position"])
        # on-field (offense) for history dropbacks: one row per player per play
        oo = H[["pidx", "season", "week", "man", "shell", "blitz", "is_play_action", "offense_players"]].dropna(subset=["offense_players"]).copy()
        oo["gsis"] = oo["offense_players"].str.split(";")
        oo = oo.explode("gsis").drop(columns="offense_players")
        oo["pos"] = oo["gsis"].map(ids["position"])
        oo = oo[oo["pos"].isin(POSITIONS)]
        tg = H[H["receiver"].notna() & (H["pass_attempt"] == 1)][["pidx", "receiver", "epa", "ppr", "complete_pass", "yards_gained", "air_yards", "route"]]
        tg = tg.rename(columns={"receiver": "gsis"})
        oo = oo.merge(tg, on=["pidx", "gsis"], how="left")
        oo["tgt"] = oo["epa"].notna().astype(int)
        # quarterbacks: the passer on the play (not on-field membership)
        t = {"cur": cur, "ids": ids, "H": H, "C": C, "oo": oo}
        t["lg"] = _league(H, C)
        t["eb"] = _eb(oo)
        t["persist"] = _persist(H)
        t["oo_idx"] = oo.groupby("gsis").indices
        t["lg_rt"] = {pos: oo[(oo["tgt"] == 1) & (oo["pos"] == pos) & oo["route"].notna()]["route"].value_counts(normalize=True) for pos in ("RB", "WR", "TE")}
        t["pos_tr"] = {pos: {"man": float(oo[(oo["pos"] == pos) & (oo["man"] == 1)]["tgt"].mean()), "zone": float(oo[(oo["pos"] == pos) & (oo["man"] == 0)]["tgt"].mean())}
                       for pos in ("RB", "WR", "TE")}
        tgC = C[(C["pass_attempt"] == 1) & C["receiver"].notna()]
        base = {}
        for pos in ("RB", "WR", "TE"):
            x = tgC[tgC["pos"] == pos]
            if not len(x):
                continue
            dp, loc = _depth(x["air_yards"]), x["pass_location"].value_counts(normalize=True)
            base[pos] = {"adot": float(x["air_yards"].mean()), "epa_t": float(x["epa"].mean()), "catch_rate": float(x["complete_pass"].mean()),
                         "catchable": float(x["is_catchable_ball"].mean()), "pa_share": float(x["is_play_action"].mean()),
                         "screen_share": float(x["is_screen_pass"].mean()), "blitz_share": float(x["blitz"].mean()),
                         "depth": {k: float((dp == k).mean()) for k in ("behind", "short", "mid", "deep")},
                         "loc": {k: float(loc.get(k, 0.0)) for k in ("left", "middle", "right")}}
        t["base"] = base
        # how far a starter's receiving PPR swings from one game to the next (the yardstick for any matchup nudge)
        gg = H[H["receiver"].notna() & (H["pass_attempt"] == 1)].groupby(["season", "week", "receiver"]).agg(ppr=("ppr", "sum"), n=("ppr", "size")).reset_index()
        gg = gg[gg["n"] >= 4]
        gg["pos"] = gg["receiver"].map(ids["position"])
        t["game_sd"] = {pos: float(gg[gg["pos"] == pos]["ppr"].std()) for pos in ("RB", "WR", "TE")}
        t["_pc"] = {}
        _C.update(key=key, t=t)
        return t


def _league(H: pd.DataFrame, C: pd.DataFrame) -> dict:
    h = H[H["defense_coverage_type"].notna()]
    lg = {"man": float(h["man"].mean()), "blitz": float(h["blitz"].mean()), "pa": float(h["is_play_action"].mean()),
          "pressure": float(h["was_pressure"].mean()), "pr5": float((h["n_pass_rushers"] >= 5).mean()),
          "ttt": float(h["time_to_throw"].mean()), "shell": {s: float((h["shell"] == s).mean()) for s in SHELLS}}
    for y in HIST:
        hy = h[h["season"] == y]
        lg[str(y)] = {"man": float(hy["man"].mean()), "blitz": float(hy["blitz"].mean()), "pressure": float(hy["was_pressure"].mean()),
                      "pa": float(hy["is_play_action"].mean()), "pr5": float((hy["n_pass_rushers"] >= 5).mean()),
                      "shell": {s: float((hy["shell"] == s).mean()) for s in SHELLS}}
    # The charted man/zone mix moves a lot from one season to the next for the whole league (2023-25: 41%, 48%, 31%),
    # so a team is always judged against its own season's league; this season's baseline is last season's.
    lg["base"] = {"man": lg[str(HIST[-1])]["man"], "shell": lg[str(HIST[-1])]["shell"], "season": HIST[-1]}
    c = C[C["db"] == 1]
    lg["cur"] = {"blitz": float(c["blitz"].mean()), "pa": float(c["is_play_action"].mean()), "pr5": float((c["n_pass_rushers"] >= 5).mean()),
                 "epa_db": float(c["epa"].mean()), "success": float(c["success"].mean()), "sack": float(c["sack"].mean()),
                 "comp": float(c.loc[c["pass_attempt"] == 1, "complete_pass"].mean()), "n": int(len(c))}
    r = C[C["run"] == 1]
    lg["cur_run"] = {"epa": float(r["epa"].mean()), "success": float(r["success"].mean()), "ypc": float(r["rushing_yards"].mean()),
                     "explosive": float((r["rushing_yards"] >= 10).mean()), "n": int(len(r))}
    return lg


# ---------- empirical-Bayes shrinkage for player splits (priors measured from the same table) ----------
def _eb(oo: pd.DataFrame) -> dict:
    """Per position: signal variance of (a) a split's target rate relative to the player's overall rate and
    (b) PPR per target relative to overall, from method of moments. Reliability of one player's split =
    tau2 / (tau2 + sampling variance)."""
    out = {}
    for pos in ("WR", "TE", "RB"):
        o = oo[oo["pos"] == pos]
        g = o.groupby("gsis").agg(db=("pidx", "size"), tgt=("tgt", "sum"))
        keep = g[(g["db"] >= 250)].index
        o = o[o["gsis"].isin(keep)]
        res = {}
        for name, mask_fn in (("man", lambda x: x["man"] == 1), ("zone", lambda x: x["man"] == 0)):
            res[name] = None
        # man-minus-zone target rate
        a = o[o["man"] == 1].groupby("gsis").agg(db=("pidx", "size"), tgt=("tgt", "sum"))
        b = o[o["man"] == 0].groupby("gsis").agg(db=("pidx", "size"), tgt=("tgt", "sum"))
        j = a.join(b, lsuffix="_m", rsuffix="_z", how="inner")
        j = j[(j["db_m"] >= 60) & (j["db_z"] >= 60)]
        pm, pz = j["tgt_m"] / j["db_m"], j["tgt_z"] / j["db_z"]
        p = (j["tgt_m"] + j["tgt_z"]) / (j["db_m"] + j["db_z"])
        s2 = p * (1 - p) * (1 / j["db_m"] + 1 / j["db_z"])
        d = pm - pz
        tau2 = max(0.0, float(d.var() - s2.mean()))
        res["tr_mz"] = {"tau2": tau2, "n": int(len(j)), "mean": float(d.mean())}
        # PPR per target, man minus zone: sampling variance from per-target spread
        t = o[o["tgt"] == 1]
        sd = float(t["ppr"].std())
        ta = t[t["man"] == 1].groupby("gsis").agg(n=("ppr", "size"), m=("ppr", "mean"))
        tb = t[t["man"] == 0].groupby("gsis").agg(n=("ppr", "size"), m=("ppr", "mean"))
        k = ta.join(tb, lsuffix="_m", rsuffix="_z", how="inner")
        k = k[(k["n_m"] >= 8) & (k["n_z"] >= 8)]
        dd = k["m_m"] - k["m_z"]
        s2e = sd ** 2 * (1 / k["n_m"] + 1 / k["n_z"])
        res["ppr_mz"] = {"tau2": max(0.0, float(dd.var() - s2e.mean())), "sd": sd, "n": int(len(k)), "mean": float(dd.mean())}
        out[pos] = res
    return out


def _rel(tau2: float, s2: float) -> float:
    return float(tau2 / (tau2 + s2)) if (tau2 + s2) > 0 else 0.0


# ---------- team (defense) side ----------
def _team_hist(H: pd.DataFrame, abbr: str) -> list[dict]:
    rows = []
    for y in HIST:
        h = H[(H["defteam"] == abbr) & (H["season"] == y) & H["defense_coverage_type"].notna()]
        if not len(h):
            continue
        rows.append({"season": y, "n": int(len(h)), "man": float(h["man"].mean()),
                     "shell": {s: float((h["shell"] == s).mean()) for s in SHELLS},
                     "blitz": float(h["blitz"].mean()), "pr5": float((h["n_pass_rushers"] >= 5).mean()),
                     "pressure": float(h["was_pressure"].mean()), "pa": float(h["is_play_action"].mean()),
                     "ttt": float(h["time_to_throw"].mean()),
                     "epa_db": float(h["epa"].mean()), "success": float(h["success"].mean())})
    return rows


def _persist(H: pd.DataFrame) -> dict:
    """Year-to-year regression slope of a team's trait on last season's, stacked over the history seasons."""
    out = {}
    cols = {"man": "man", "blitz": "blitz", "pa": "is_play_action", "pressure": "was_pressure"}
    for k, c in cols.items():
        xs, ys = [], []
        for y in HIST[:-1]:
            a = H[(H["season"] == y)].groupby("defteam")[c].mean()
            b = H[(H["season"] == y + 1)].groupby("defteam")[c].mean()
            a, b = a.align(b, join="inner")
            xs += list((a - a.mean())); ys += list((b - b.mean()))
        x, y_ = np.array(xs), np.array(ys)
        out[k] = float((x * y_).sum() / (x * x).sum()) if len(x) else 0.0
    for s in SHELLS:
        xs, ys = [], []
        for y in HIST[:-1]:
            a = H[(H["season"] == y)].assign(v=lambda d: (d["shell"] == s).astype(float)).groupby("defteam")["v"].mean()
            b = H[(H["season"] == y + 1)].assign(v=lambda d: (d["shell"] == s).astype(float)).groupby("defteam")["v"].mean()
            a, b = a.align(b, join="inner")
            xs += list((a - a.mean())); ys += list((b - b.mean()))
        x, y_ = np.array(xs), np.array(ys)
        out["shell_" + s] = float((x * y_).sum() / (x * x).sum()) if len(x) else 0.0
    return out


def defense(abbr: str) -> dict | None:
    t = tables()
    H, C, lg, ids = t["H"], t["C"], t["lg"], t["ids"]
    if abbr not in set(C["defteam"].dropna()) and abbr not in set(H["defteam"].dropna()):
        return None
    hist = _team_hist(H, abbr)
    pers = t["persist"]
    last = hist[-1] if hist else None
    exp = None
    if last:
        b = lg["base"]
        sh = {s: max(0.0, b["shell"][s] + pers["shell_" + s] * (last["shell"][s] - b["shell"][s])) for s in SHELLS}
        tot = sum(sh.values()) or 1.0  # each shell is regressed on its own, so put the mix back on 100%
        exp = {"man": b["man"] + pers["man"] * (last["man"] - b["man"]), "shell": {s: v / tot for s, v in sh.items()}}
    c = C[(C["defteam"] == abbr) & (C["db"] == 1)]
    g = int(c["week"].nunique())
    cur = None
    if len(c):
        att = c[c["pass_attempt"] == 1]
        w = g / (g + 5.0)  # this season's blitz/PA rate vs last season's: 5 games of prior
        cur = {"games": g, "n": int(len(c)), "epa_db": float(c["epa"].mean()), "success": float(c["success"].mean()),
               "comp": float(att["complete_pass"].mean()), "ypa": float(att["yards_gained"].mean()), "sack": float(c["sack"].mean()),
               "hit": float(c["qb_hit"].mean()), "blitz": float(c["blitz"].mean()), "pr5": float((c["n_pass_rushers"] >= 5).mean()),
               "pa": float(c["is_play_action"].mean()), "screen": float(c["is_screen_pass"].mean()),
               "blitz_exp": float(w * c["blitz"].mean() + (1 - w) * (last["blitz"] if last else lg["blitz"]))}
    r = C[(C["defteam"] == abbr) & (C["run"] == 1)]
    run = {"n": int(len(r)), "epa": float(r["epa"].mean()), "success": float(r["success"].mean()), "ypc": float(r["rushing_yards"].mean()),
           "explosive": float((r["rushing_yards"] >= 10).mean())} if len(r) else None
    # what it allows, by position of the target and by depth (current season; shrunk toward the league)
    allowed = {}
    tg = C[(C["pass_attempt"] == 1) & C["receiver"].notna()]
    lgt = tg.groupby("pos")["ppr"].agg(["mean", "size"])
    mine = tg[tg["defteam"] == abbr]
    for pos in ("RB", "WR", "TE"):
        m = mine[mine["pos"] == pos]
        n = len(m)
        base = float(lgt.loc[pos, "mean"]) if pos in lgt.index else 0.0
        val = (m["ppr"].sum() + K_ALLOWED * base) / (n + K_ALLOWED)
        allowed[pos] = {"tgt": int(n), "share": float(n / max(1, len(mine))), "league_share": float(lgt.loc[pos, "size"] / lgt["size"].sum()) if pos in lgt.index else 0.0,
                        "ppr_t": float(val), "league_ppr_t": base, "raw_ppr_t": float(m["ppr"].mean()) if n else None}
    depth = {}
    tgd = tg.assign(dp=_depth(tg["air_yards"]))
    for k, grp in tgd[tgd["defteam"] == abbr].groupby("dp", observed=True):
        lgd = tgd[tgd["dp"] == k]
        depth[str(k)] = {"tgt": int(len(grp)), "epa_t": float(grp["epa"].mean()), "league_epa_t": float(lgd["epa"].mean()),
                         "comp": float(grp["complete_pass"].mean()), "league_comp": float(lgd["complete_pass"].mean())}
    faced = []
    if len(mine):
        f = mine.groupby("receiver").agg(tgt=("ppr", "size"), rec=("complete_pass", "sum"), yds=("yards_gained", "sum"), td=("pass_touchdown", "sum"), ppr=("ppr", "sum"))
        for g, r in f.sort_values("ppr", ascending=False).head(8).iterrows():
            faced.append({"gsis": g, "name": str(ids["name"].get(g, g)), "pos": str(ids["position"].get(g, "?")), "tgt": int(r["tgt"]), "rec": int(r["rec"]),
                          "yds": float(r["yds"]), "td": int(r["td"]), "ppr": float(r["ppr"])})
    return {"abbr": abbr, "history": hist, "expected": exp, "persist": pers, "current": cur, "run": run, "allowed": allowed, "depth": depth,
            "faced": faced, "league": lg, "shell_label": SHELL_LABEL}


# ---------- player side ----------
def _bucket_stats(g: pd.DataFrame) -> dict:
    n_t = int(g["tgt"].sum())
    return {"db": int(len(g)), "tgt": n_t, "tr": float(n_t / len(g)) if len(g) else None,
            "ypt": float(g.loc[g["tgt"] == 1, "yards_gained"].mean()) if n_t else None,
            "ppr_t": float(g.loc[g["tgt"] == 1, "ppr"].mean()) if n_t else None,
            "epa_t": float(g.loc[g["tgt"] == 1, "epa"].mean()) if n_t else None,
            "comp": float(g.loc[g["tgt"] == 1, "complete_pass"].mean()) if n_t else None}


def player(gsis: str, opp: str | None = None) -> dict | None:
    t = tables()
    k = (gsis, opp)
    if k not in t["_pc"]:
        t["_pc"][k] = _player(t, gsis, opp)
    return t["_pc"][k]


def _player(t: dict, gsis: str, opp: str | None) -> dict | None:
    ids, oo, C, H, lg, eb = t["ids"], t["oo"], t["C"], t["H"], t["lg"], t["eb"]
    if gsis not in ids.index:
        return None
    pos = str(ids.loc[gsis, "position"])
    out = {"gsis": gsis, "name": str(ids.loc[gsis, "name"]), "pos": pos}
    if pos == "QB":
        return _qb(out, gsis, t, opp)
    if pos not in ("RB", "WR", "TE"):
        return None
    # this season (play-by-play + FTN): usage and quality of the targets
    me = C[(C["receiver"] == gsis) & (C["pass_attempt"] == 1)]
    team = me["posteam"].mode().iat[0] if len(me) else None
    tt = C[(C["posteam"] == team) & (C["db"] == 1) & (C["pass_attempt"] == 1)] if team else C.iloc[0:0]
    games = int(C.loc[(C["posteam"] == team), "week"].nunique()) if team else 0
    cur = None
    if len(me):
        dp = _depth(me["air_yards"])
        loc = me["pass_location"].value_counts(normalize=True)
        cur = {"team": team, "games": games, "tgt": int(len(me)), "team_tgt": int(len(tt)), "share": float(len(me) / max(1, len(tt))),
               "rec": int(me["complete_pass"].sum()), "yds": float(me["yards_gained"].sum()), "td": int(me["pass_touchdown"].sum()),
               "adot": float(me["air_yards"].mean()), "ppr": float(me["ppr"].sum()),
               "depth": {k: float((dp == k).mean()) for k in ("behind", "short", "mid", "deep")},
               "loc": {k: float(loc.get(k, 0.0)) for k in ("left", "middle", "right")},
               "pa_share": float(me["is_play_action"].mean()), "team_pa_share": float(tt["is_play_action"].mean()),
               "screen_share": float(me["is_screen_pass"].mean()), "team_screen_share": float(tt["is_screen_pass"].mean()),
               "blitz_share": float(me["blitz"].mean()), "team_blitz_share": float(tt["blitz"].mean()),
               "contested": float(me["is_contested_ball"].sum()), "drops": float(me["is_drop"].sum()),
               "catchable": float(me["is_catchable_ball"].mean()), "catch_rate": float(me["complete_pass"].mean()),
               "epa_t": float(me["epa"].mean()), "base": t["base"].get(pos)}
        # runs, for backs
    rush = C[(C["rusher"] == gsis) & (C["run"] == 1)]
    if pos == "RB" and len(rush):
        out["rush"] = {"car": int(len(rush)), "yds": float(rush["rushing_yards"].sum()), "ypc": float(rush["rushing_yards"].mean()),
                       "epa": float(rush["epa"].mean()), "success": float(rush["success"].mean()),
                       "explosive": float((rush["rushing_yards"] >= 10).mean()), "td": int(rush["rush_touchdown"].sum())}
    out["current"] = cur
    # against coverage, 2023-25 (on-field denominators)
    o = oo.iloc[t["oo_idx"].get(gsis, [])]
    out["hist"] = None
    if len(o) >= 60:
        by_season = {int(y): int(len(s)) for y, s in o.groupby("season")}
        tot = _bucket_stats(o)
        pm = o[o["man"] == 1]; pz = o[o["man"] == 0]
        e = eb.get(pos, {})
        tr_mz = e.get("tr_mz", {"tau2": 0.0, "mean": 0.0, "n": 0})
        pe = e.get("ppr_mz", {"tau2": 0.0, "sd": 1.0, "mean": 0.0, "n": 0})
        sm, sz = _bucket_stats(pm), _bucket_stats(pz)
        p_all = tot["tr"] or 0.0
        s2 = p_all * (1 - p_all) * (1 / max(1, sm["db"]) + 1 / max(1, sz["db"]))
        rel_tr = _rel(tr_mz["tau2"], s2)
        d_tr = (sm["tr"] - sz["tr"]) if sm["tr"] is not None and sz["tr"] is not None else None
        mean_d = tr_mz["mean"]
        d_tr_sh = None if d_tr is None else mean_d + rel_tr * (d_tr - mean_d)
        d_pe = (sm["ppr_t"] - sz["ppr_t"]) if sm["ppr_t"] is not None and sz["ppr_t"] is not None else None
        s2e = pe["sd"] ** 2 * (1 / max(1, sm["tgt"]) + 1 / max(1, sz["tgt"]))
        rel_pe = _rel(pe["tau2"], s2e)
        d_pe_sh = None if d_pe is None else pe["mean"] + rel_pe * (d_pe - pe["mean"])
        shells = {}
        for s in SHELLS:
            gs = o[o["shell"] == s]
            st = _bucket_stats(gs)
            # target rate vs this shell relative to the rest, shrunk with the man/zone reliability as a stand-in
            shells[s] = st
        bl = {"blitz": _bucket_stats(o[o["blitz"] == 1]), "no_blitz": _bucket_stats(o[o["blitz"] == 0]),
              "pa": _bucket_stats(o[o["is_play_action"] == 1]), "no_pa": _bucket_stats(o[o["is_play_action"] == 0])}
        rt = o[(o["tgt"] == 1) & o["route"].notna()]
        lg_rt = t["lg_rt"][pos]
        routes = [{"route": k, "label": ROUTE_LABEL.get(k, k.title()), "n": int(v), "share": float(v / len(rt)), "league_share": float(lg_rt.get(k, 0.0)),
                   "ppr_t": float(rt[rt["route"] == k]["ppr"].mean()), "epa_t": float(rt[rt["route"] == k]["epa"].mean())}
                  for k, v in rt["route"].value_counts().items() if v >= 3]
        out["hist"] = {"seasons": by_season, "total": tot, "man": sm, "zone": sz,
                       "tr_diff": d_tr, "tr_diff_shrunk": d_tr_sh, "tr_rel": rel_tr, "ppr_diff": d_pe, "ppr_diff_shrunk": d_pe_sh, "ppr_rel": rel_pe,
                       "shells": shells, "other": bl, "routes": routes[:10],
                       "pos_mean_tr": t["pos_tr"][pos]}
    if opp:
        out["lens"] = _lens(out, opp)
    return out


def _lens(p: dict, opp: str) -> dict | None:
    dfn = defense(opp)
    h = p.get("hist")
    if not dfn or not dfn.get("expected") or not h or h.get("tr_diff_shrunk") is None:
        return None
    lg = dfn["league"]
    d_man = dfn["expected"]["man"] - lg["base"]["man"]
    tr = h["total"]["tr"] or 0.0
    # a typical game has about 36 dropbacks; a starter is on the field for most of them
    db_game = 36.0
    d_t = h["tr_diff_shrunk"] * d_man * db_game
    d_ppr = (h["ppr_diff_shrunk"] or 0.0) * d_man * (tr * db_game)
    return {"opp": opp, "opp_man_exp": dfn["expected"]["man"], "league_man": lg["base"]["man"], "d_targets": float(d_t), "d_ppr": float(d_ppr + d_t * (h["total"]["ppr_t"] or 0.0)),
            "tr_rel": h["tr_rel"], "ppr_rel": h["ppr_rel"], "game_sd": tables()["game_sd"].get(p["pos"]), "tgt_game": float(tr * db_game)}


def _qb(out: dict, gsis: str, t: dict, opp: str | None) -> dict:
    H, C = t["H"], t["C"]
    me = C[(C["passer"] == gsis) & (C["db"] == 1)]
    out["current"] = None
    if len(me):
        att = me[me["pass_attempt"] == 1]
        out["current"] = {"db": int(len(me)), "epa_db": float(me["epa"].mean()), "success": float(me["success"].mean()), "comp": float(att["complete_pass"].mean()),
                          "ypa": float(att["yards_gained"].mean()), "sack": float(me["sack"].mean()), "adot": float(att["air_yards"].mean()),
                          "pa": float(me["is_play_action"].mean()), "blitz_epa": float(me.loc[me["blitz"] == 1, "epa"].mean()), "noblitz_epa": float(me.loc[me["blitz"] == 0, "epa"].mean()),
                          "blitz_n": int((me["blitz"] == 1).sum()), "td": int(me["pass_touchdown"].sum()), "int": int(me["interception"].sum())}
    h = H[(H["passer"] == gsis) & H["defense_coverage_type"].notna()]
    out["hist"] = None
    if len(h) >= 100:
        def st(x):
            a = x[x["pass_attempt"] == 1]
            return {"db": int(len(x)), "epa_db": float(x["epa"].mean()), "se": float(x["epa"].std() / np.sqrt(max(1, len(x)))) if len(x) > 1 else None,
                    "comp": float(a["complete_pass"].mean()) if len(a) else None,
                    "ypa": float(a["yards_gained"].mean()) if len(a) else None, "sack": float(x["sack"].mean()), "pressure": float(x["was_pressure"].mean())}
        out["hist"] = {"total": st(h), "man": st(h[h["man"] == 1]), "zone": st(h[h["man"] == 0]),
                       "shells": {s: st(h[h["shell"] == s]) for s in SHELLS}, "blitz": st(h[h["blitz"] == 1]), "no_blitz": st(h[h["blitz"] == 0]),
                       "pa": st(h[h["is_play_action"] == 1]), "no_pa": st(h[h["is_play_action"] == 0]),
                       "seasons": {int(y): int(len(s)) for y, s in h.groupby("season")}}
    return out


# ---------- defenders ----------
EV = {"sack": ["sack_player_id"], "half": ["half_sack_1_player_id", "half_sack_2_player_id"], "hit": ["qb_hit_1_player_id", "qb_hit_2_player_id"],
      "tfl": ["tackle_for_loss_1_player_id", "tackle_for_loss_2_player_id"], "int": ["interception_player_id"],
      "pbu": ["pass_defense_1_player_id", "pass_defense_2_player_id"], "solo": ["solo_tackle_1_player_id"],
      "ast": ["assist_tackle_1_player_id", "assist_tackle_2_player_id", "assist_tackle_3_player_id", "assist_tackle_4_player_id"],
      "ff": ["forced_fumble_player_1_player_id"]}


def _events(P: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for k, cols in EV.items():
        for c in cols:
            if c in P:
                x = P[P[c].notna()][["defteam", c]].rename(columns={c: "gsis"})
                x["ev"] = k
                parts.append(x)
    e = pd.concat(parts, ignore_index=True)
    return e.pivot_table(index=["defteam", "gsis"], columns="ev", aggfunc="size", fill_value=0).reset_index()


TEAM_FIX = {"KCC": "KC", "GBP": "GB", "LVR": "LV", "NEP": "NE", "NOS": "NO", "SFO": "SF", "TBB": "TB", "SDC": "LAC", "JAC": "JAX", "LAR": "LA",
            "RAM": "LA", "STL": "LA", "OAK": "LV", "WSH": "WAS", "ARZ": "ARI"}


def _on_roster(abbr: str) -> set:
    """Players the nflverse id table currently lists with this team (it moves when trades and cuts land)."""
    i = tables()["ids"]
    tm = i["team"].map(lambda x: TEAM_FIX.get(x, x))
    return set(i.index[tm == abbr])


def defenders(abbr: str) -> dict:
    t = tables()
    ids, C, H = t["ids"], t["C"], t["H"]
    out = {"abbr": abbr, "current": [], "last": []}
    for key, P in (("current", C), ("last", H[H["season"] == HIST[-1]])):
        e = _events(P[P["defteam"] == abbr]) if len(P[P["defteam"] == abbr]) else pd.DataFrame()
        if not len(e):
            continue
        for c in EV:
            if c not in e:
                e[c] = 0
        e["sacks"] = e["sack"] + 0.5 * e["half"]
        e["tkl"] = e["solo"] + e["ast"]
        e["pos"] = e["gsis"].map(ids["position"]).fillna("?")
        e["name"] = e["gsis"].map(ids["name"]).fillna(e["gsis"])
        e["score"] = 4 * e["sacks"] + 1.5 * e["hit"] + 1.5 * e["tfl"] + 3 * e["int"] + 1.5 * e["pbu"] + 0.25 * e["tkl"] + 2 * e["ff"]
        e = e.sort_values("score", ascending=False).head(14)
        here = _on_roster(abbr)
        out[key] = [{"gsis": r.gsis, "name": r["name"], "pos": r.pos, "sacks": float(r.sacks), "hit": int(r.hit), "tfl": int(r.tfl), "int": int(r["int"]),
                     "pbu": int(r.pbu), "tkl": int(r.tkl), "ff": int(r.ff), "still": (r.gsis in here) if here else None} for _, r in e.iterrows()]
    # last season: snaps on the field (pass plays) and team EPA allowed with / without him
    h = H[(H["season"] == HIST[-1]) & (H["defteam"] == abbr) & H["defense_players"].notna()]
    if len(h):
        ex = h[["pidx", "epa", "success", "defense_players", "defense_positions"]].copy()
        ex["g"] = ex["defense_players"].str.split(";"); ex["ps"] = ex["defense_positions"].str.split(";")
        ex = ex.explode(["g", "ps"])
        tot_n, tot_epa = len(h), float(h["epa"].sum())
        sd = float(h["epa"].std())
        rows = {}
        for g, s in ex.groupby("g"):
            on = len(s)
            if on < 0.25 * tot_n:
                continue
            on_epa = float(s["epa"].mean()); off_n = tot_n - on
            off_epa = (tot_epa - float(s["epa"].sum())) / off_n if off_n > 0 else None
            se = sd * (1 / np.sqrt(on) + 1 / np.sqrt(max(1, off_n)))
            if off_n < 40:  # on the field for essentially every snap: there is no "without him" to compare
                off_epa, se = None, None
            rows[g] = {"on": on, "off": off_n, "on_epa": on_epa, "off_epa": off_epa, "pos": s["ps"].mode().iat[0] if len(s) else None, "se": se}
        here = _on_roster(abbr)
        out["onoff"] = [{"gsis": g, "name": str(ids["name"].get(g, g)), "pct": v["on"] / tot_n, "still": (g in here) if here else None, **v}
                        for g, v in sorted(rows.items(), key=lambda kv: -kv[1]["on"])[:18]]
        out["onoff_dropbacks"] = int(tot_n)
        out["onoff_season"] = HIST[-1]
    return out


def abbrs() -> list[str]:
    return sorted(set(tables()["C"]["defteam"].dropna()))


def board() -> dict:
    t = tables()
    teams = {}
    lg = t["lg"]
    for a in abbrs():
        d = defense(a)
        last = d["history"][-1] if d["history"] else None
        teams[a] = {"man": d["expected"]["man"] if d["expected"] else None, "blitz": d["current"]["blitz_exp"] if d["current"] else None,
                    "epa_db": d["current"]["epa_db"] if d["current"] else None, "run_epa": d["run"]["epa"] if d["run"] else None,
                    "games": d["current"]["games"] if d["current"] else 0, "pressure": last["pressure"] if last else None}
    return {"season": t["cur"], "hist": list(HIST), "teams": teams, "league": lg,
            "attribution": "FTN Data via nflverse (CC BY-SA 4.0); play-by-play from nflverse",
            "alias": {"LAR": "LA", "WSH": "WAS", "JAC": "JAX", "ARZ": "ARI", "LVR": "LV"}}


# ---------- the screen's views ----------
ALIAS = {"LAR": "LA", "WSH": "WAS", "JAC": "JAX", "ARZ": "ARI", "LVR": "LV"}


def norm(a: str | None) -> str | None:
    return None if a is None else ALIAS.get(str(a).upper(), str(a).upper())


def schedule(season: int, week: int) -> dict:
    """team -> {opp, home, date, time} for one week of the nflverse schedule."""
    f = ROOT / "history" / "nflverse" / "schedules.csv"
    try:
        sc = pd.read_csv(f)
    except Exception:
        return {}
    sc = sc[(sc["season"] == season) & (sc["week"] == week) & (sc["game_type"] == "REG")]
    out = {}
    for _, r in sc.iterrows():
        out[norm(r["home_team"])] = {"opp": norm(r["away_team"]), "home": True, "date": str(r["gameday"]), "time": r.get("gametime")}
        out[norm(r["away_team"])] = {"opp": norm(r["home_team"]), "home": False, "date": str(r["gameday"]), "time": r.get("gametime")}
    return out


def _gsis_by_espn() -> dict:
    i = tables()["ids"]
    x = i[i["espn_id"].notna()]
    return {int(e): g for g, e in x["espn_id"].items()}


def _brief(p: dict | None) -> dict | None:
    """What a list row needs from a player file (full file loads on click)."""
    if not p:
        return None
    pos, cur, h = p["pos"], p.get("current"), p.get("hist")
    out = {"pos": pos}
    if pos == "QB":
        if cur:
            out.update(db=cur["db"], epa_db=cur["epa_db"], adot=cur["adot"], pa=cur["pa"])
        if h:
            out.update(epa_man=h["man"]["epa_db"], epa_zone=h["zone"]["epa_db"], n_man=h["man"]["db"], n_zone=h["zone"]["db"])
        return out
    if cur:
        out.update(games=cur["games"], tgt=cur["tgt"], share=cur["share"], adot=cur["adot"], epa_t=cur["epa_t"])
    if h:
        out.update(tr_man=h["man"]["tr"], tr_zone=h["zone"]["tr"], n_man=h["man"]["db"], n_zone=h["zone"]["db"], diff=h["tr_diff"],
                   diff_shrunk=h["tr_diff_shrunk"], rel=h["tr_rel"], hist_db=h["total"]["db"])
    return out


def view(snap: dict, team: int = 3) -> dict:
    """Matchups board: all 32 defenses in brief, this week's opponents, Max's players and the other side's starters."""
    t = tables()
    b = board()
    season, week = int(snap["season"]), int(snap["week"])
    sch = schedule(season, week)
    g_by = _gsis_by_espn()
    tw = snap["team_week"]

    def row(o: dict, side: str, slot: str | None) -> dict:
        tm = norm(o.get("team"))
        game = o.get("game") or {}
        opp = None if game.get("bye") else norm(game.get("opp"))
        g = g_by.get(int(o["id"]))
        pl = player(g, opp) if (g and o.get("pos") in POSITIONS) else None
        lens = pl.get("lens") if pl else None
        return {"id": int(o["id"]), "gsis": g, "name": o["name"], "pos": o["pos"], "team": tm, "opp": opp, "home": game.get("home"), "day": game.get("day"),
                "date": game.get("date"), "time": game.get("time"), "bye": bool(game.get("bye")), "slot": slot or o.get("slot"), "side": side,
                "injury": o.get("injury"), "headshot": o.get("headshot"), "brief": _brief(pl), "lens": lens}

    mine = [row(o, "me", o.get("slot")) for o in tw["players"] if o.get("pos") in POSITIONS]
    theirs = [row(o, "opp", o.get("slot")) for o in (tw.get("matchup") or {}).get("opp_starters", []) if o.get("pos") in POSITIONS]
    return {**b, "week": week, "schedule": sch, "mine": mine, "theirs": theirs,
            "opp_name": (tw.get("matchup") or {}).get("opp_name"), "team_name": tw.get("team_name")}


def defense_view(abbr: str) -> dict | None:
    abbr = norm(abbr)
    d = defense(abbr)
    if not d:
        return None
    return {"defense": d, "defenders": defenders(abbr)}


def player_view(espn_id: int, opp: str | None = None, season: int | None = None, week: int | None = None) -> dict | None:
    t = tables()
    g = _gsis_by_espn().get(int(espn_id))
    if not g:
        return None
    i = t["ids"]
    team = norm(TEAM_FIX.get(i.loc[g, "team"], i.loc[g, "team"]))
    p0 = player(g, None)
    if p0 and p0.get("current") and p0["current"].get("team"):
        team = p0["current"]["team"]
    game = None
    if opp is None and season and week:
        game = schedule(season, week).get(team)
        opp = game["opp"] if game else None
    p = player(g, norm(opp)) if opp else p0
    if not p:
        return None
    return {**p, "espn_id": int(espn_id), "team": team, "opp": norm(opp), "game": game}


def warm(snap: dict | None = None) -> None:
    md.load(current_season())  # refreshes the file when it is more than 6 h old; tables() rebuilds when the file changes
    tables()
    board()


if __name__ == "__main__":
    import json, sys, time
    t0 = time.time(); tables(); print("tables", round(time.time() - t0, 1), "s")
    t0 = time.time(); b = board(); print("board", round(time.time() - t0, 1), "s", len(b["teams"]), "teams")
    print(json.dumps(defense("DAL"), default=float)[:600])
