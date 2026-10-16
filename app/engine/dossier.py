"""Player file: everything Desk knows about one player, for the full-screen dossier.

Pieces (each backtested where it makes a claim; see roles_fit.py and app/DESIGN.md):
role (style + tier + traits + trends), fingerprint (every usage metric with its shrunk estimate,
percentile and how much of it is signal yet), role map (style axes for the whole position),
game log (real box score, ESPN points and projection), comps (similar player-seasons 2016-25 and
what happened next), and "the read" (plain-language takeaways with the evidence behind each).
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

try:
    from . import comps as C, roles as R
except ImportError:
    import comps as C, roles as R


def _f(x, nd=1):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return round(float(x), nd)
    except (TypeError, ValueError):
        return None


def _age(birth, season):
    try:
        return round((pd.Timestamp(f"{season}-09-01") - pd.Timestamp(birth)).days / 365.25, 1)
    except Exception:
        return None


def _map(model, pos, cur_agg, prev_agg, d_w, me, owners, g2e):
    """Role map: shrunk style axes for every qualified player at the position, his weekly trail,
    last season's point, and each style group's 1-sd ellipse on these two axes."""
    st = model["style"][pos]
    pff_axes = model.get("axes_pff", {}).get(pos)
    if pff_axes and pos in model.get("style_free", {}) and not cur_agg[cur_agg.pos == pos].reindex(columns=pff_axes).notna().all(axis=1).any():
        st = model["style_free"][pos]  # no PFF alignment this season: free-data groups on the free axes
    ax, ay = pff_axes if pff_axes and all(a in st["features"] for a in pff_axes) else model["axes"][pos]
    f = st["features"]; ix, iy = f.index(ax), f.index(ay)
    groups = []
    for nm, mu, cov in zip(st["names"], st["means"], st["covs"]):
        mx = mu[ix] * st["scale"][ix] + st["mean"][ix]; my = mu[iy] * st["scale"][iy] + st["mean"][iy]
        c = np.array([[cov[ix][ix] * st["scale"][ix] ** 2, cov[ix][iy] * st["scale"][ix] * st["scale"][iy]],
                      [cov[iy][ix] * st["scale"][ix] * st["scale"][iy], cov[iy][iy] * st["scale"][iy] ** 2]])
        groups.append({"name": nm, "x": mx, "y": my, "cov": c.round(6).tolist()})
    col, mn = {"RB": ("opp_sh", .05), "WR": ("tgt_sh", .06), "TE": ("tgt_sh", .05), "QB": ("db_sh", .5)}[pos]
    peers = []
    for g, r in cur_agg[cur_agg.pos == pos].iterrows():
        if r["games"] < 2 or (r.get(col) or 0) < mn:
            continue
        p = prev_agg.loc[g] if g in prev_agg.index else None
        pg = float(p["games"]) if p is not None else 0
        x = R.shrink(model, pos, ax, r.get(ax), r["games"], None if p is None else p.get(ax), pg)
        y = R.shrink(model, pos, ay, r.get(ay), r["games"], None if p is None else p.get(ay), pg)
        if x is None or y is None:
            continue
        eid = g2e.get(g)
        peers.append({"id": eid, "x": x, "y": y, "vol": _f(r.get(col), 3), "name": r.get("pname"), "owner": owners.get(eid)})
    trail = []
    if (ax in R.PFF_STYLE.get(pos, {}) or ay in R.PFF_STYLE.get(pos, {})) and me in cur_agg.index:
        # PFF alignment is season to date only: one point, no weekly trail
        r = cur_agg.loc[me]; p = prev_agg.loc[me] if me in prev_agg.index else None
        pg = float(p["games"]) if p is not None else 0
        x = R.shrink(model, pos, ax, r.get(ax), r["games"], None if p is None else p.get(ax), pg)
        y = R.shrink(model, pos, ay, r.get(ay), r["games"], None if p is None else p.get(ay), pg)
        if x is not None and y is not None and d_w is not None and len(d_w):
            trail.append({"week": int(d_w.iloc[-1]["week"]), "x": x, "y": y})
    elif d_w is not None and len(d_w):
        p = prev_agg.loc[me] if me in prev_agg.index else None
        pg = float(p["games"]) if p is not None else 0
        for i in range(1, len(d_w) + 1):
            a = R.aggregate(d_w.iloc[:i], ("gsis", "pos")).iloc[0]
            x = R.shrink(model, pos, ax, a.get(ax), a["games"], None if p is None else p.get(ax), pg)
            y = R.shrink(model, pos, ay, a.get(ay), a["games"], None if p is None else p.get(ay), pg)
            if x is not None and y is not None:
                trail.append({"week": int(d_w.iloc[i - 1]["week"]), "x": x, "y": y})
    last = None
    if me in prev_agg.index:
        p = prev_agg.loc[me]
        if p["games"] >= 4 and not pd.isna(p.get(ax)) and not pd.isna(p.get(ay)):
            last = {"x": float(p[ax]), "y": float(p[ay])}
    labels = {k: R.METRICS[pos].get(k, (None,))[0] for k in (ax, ay)}
    fallback = {"rec_mix": "Touches that are catches", "pd_ratio": "Passing-down tilt (1 = even)", "behind_r": "Targets at or behind the line",
                "adot": "Average depth of target", "tgt_per_snap": "Targets per snap", "des_rush_g": "Designed runs per game", "p_adot": "Average throw depth",
                "slot_rate": "Pass snaps in the slot (PFF)", "inline_rate": "Pass snaps inline (PFF)"}
    return {"x": ax, "y": ay, "xlabel": labels[ax] or fallback.get(ax, ax), "ylabel": labels[ay] or fallback.get(ay, ay),
            "groups": groups, "peers": peers, "trail": trail, "last_season": last,
            "fmt": {a: "pct100" if a in R.PFF_STYLE.get(pos, {}) else R.METRICS[pos].get(a, (None, "num2"))[1] for a in (ax, ay)},
            "seasons": st.get("seasons", model["seasons"])}


def _style_info(model, pos, st):
    """Facts about the style model this player was typed with (PFF or the free-data fallback)."""
    m = model["style"][pos]
    if st and not st.get("pff", True) and pos in model.get("style_free", {}):
        m = model["style_free"][pos]
    out = {k: m.get(k) for k in ("names", "centroids", "transitions", "meaning", "early_acc", "yoy_same", "kappa")}
    out["seasons"] = m.get("seasons", model["seasons"]); out["pff"] = any(c in R.PFF_STYLE.get(pos, {}) for c in m["features"])
    return out


def _read(pos, name, prof, fp, comps, est_ppg, model, games, espn_wk, logs):
    """Plain-language takeaways, each tagged signal / caution / outlook, with where to look."""
    out, first = [], name.split(" ")[0]
    st = prof.get("style")
    if st and games < 1:
        out.append({"kind": "role", "anchor": "map", "text": f"Hasn't played this season. Last season he was {('an ' if st['name'][0] in 'AEIOU' else 'a ')}{st['name'].lower()} "
                    f"({round(st['p'] * 100)}%); everything below leans on that."})
    elif st:
        sure = "" if st["p"] >= 0.75 else f", leaning {st['alt'].lower()}" if st["alt_p"] >= 0.25 else ""
        sm = model["style_free"][pos] if not st.get("pff", True) and pos in model.get("style_free", {}) else model["style"][pos]
        acc = sm.get("early_acc", {}).get("G3" if games < 6 else "G6", {}).get("acc")
        out.append({"kind": "role", "anchor": "map", "text": f"{prof['tier']}, {st['name'].lower()}{sure} ({round(st['p'] * 100)}% sure after {games} games"
                    + (f"; labels this early match the full-season call {round(acc * 100)}% of the time" if acc and games < 8 else "") + ")."})
    reliable = [m for m in fp if (m["rel"] or 0) >= 0.6 and m["pct"] is not None and m["group"] in ("Role", "Style") and m["key"] != "snap"]
    if reliable:
        top = max(reliable, key=lambda m: abs(m["pct"] - 50))
        if abs(top["pct"] - 50) >= 30:
            side = "top" if top["pct"] >= 50 else "bottom"; v = round(top["pct"] if side == "top" else 100 - top["pct"])
            out.append({"kind": "signal", "anchor": "fingerprint", "text": f"Real already: {top['label'].lower()} {R.fmt_val(top['fmt'], top['est'])}, {side} {max(1, 100 - v)}% of {pos}s. "
                        f"This stat settles in ~{top['stab_games']:.0f} game{'s' if top['stab_games'] >= 1.5 else ''}, so {games} games is plenty."})
    loud = [m for m in fp if m["raw"] is not None and (m["rel"] or 0) < 0.35 and m["group"] in ("Efficiency", "Scoring") and m["key"] != "ppr_g"]
    loud = [m for m in loud if (R.pct_of(model, pos, m["key"], m["raw"]) or 50) >= 85 or (R.pct_of(model, pos, m["key"], m["raw"]) or 50) <= 15]
    if loud:
        m = max(loud, key=lambda m: abs((R.pct_of(model, pos, m["key"], m["raw"]) or 50) - 50))
        hi = (R.pct_of(model, pos, m["key"], m["raw"]) or 50) >= 50
        out.append({"kind": "caution", "anchor": "fingerprint", "text": f"Mostly noise so far: {m['label'].lower()} of {R.fmt_val(m['fmt'], m['raw'])} is {'near the top' if hi else 'near the bottom'} of the league, "
                    f"but it takes ~{m['stab_games']:.0f} games to mean much. Best guess: {R.fmt_val(m['fmt'], m['est'])}. Expect it to drift {'down' if hi else 'up'}."})
    for t in prof.get("trends", []):
        if t["kind"] in ("snap_up", "snap_down", "share_spike", "share_up", "share_down") and t.get("hold") is not None:
            e = t.get("next_vs_espn")
            espn_note = ""
            if e is not None and t.get("next_lo") is not None:
                if t["next_lo"] > 0 or t["next_hi"] < 0:
                    espn_note = f" Players with this tag scored {e:+.1f} vs ESPN's projection the next week (2021–25), a real gap."
                else:
                    espn_note = " ESPN's projection already accounts for it: tagged players scored about as projected the next week."
            out.append({"kind": "change", "anchor": "usage", "text": f"{t['label']}: {t['text']} Changes like this held {round(t['hold'] * 100)}% of the time.{espn_note}"})
            break
    if comps:
        q = comps["q"]; sp = model.get("starter_ppg", {}).get(pos)
        bt = model["comps"]["bt"].get(pos, {})
        line = (f"Players with his profile after week {comps['W']} averaged {q['p50']} PPR the rest of the way; the middle 80% landed between {q['p10']} and {q['p90']} "
                f"(ranges like this held the real outcome {round(bt.get('cov80', .8) * 100)}% of the time).")
        if est_ppg is not None:
            line += f" Our model says {est_ppg:.1f}."
        out.append({"kind": "outlook", "anchor": "comps", "text": line})
        extra = f"{round(comps['kept_role'] * 100)}% of them kept their snap share and they played {round(comps['avail'] * 100)}% of their remaining games."
        if sp:
            extra += f" Weeks of 15+ came {round(comps['weeks']['ge15'] * 100)}% of the time, under 5 points {round(comps['weeks']['lt5'] * 100)}%."
        out.append({"kind": "outlook", "anchor": "comps", "text": extra})
    return out


def build(espn_id: int, ctx: dict) -> dict | None:
    """ctx: season, through_week, cur (weekly roles table), prev, ids, names, model, espn_pts (week->pts),
    espn_proj (week->proj), ledger (dict or None), owner (dict or None), owners (espn_id->owner), week."""
    model, cur, prev, ids = ctx["model"], ctx["cur"], ctx["prev"], ctx["ids"]
    e2g = ids.dropna(subset=["gsis_id", "espn_id"]).drop_duplicates("espn_id").set_index(ids.dropna(subset=["gsis_id", "espn_id"]).drop_duplicates("espn_id")["espn_id"].astype(int))["gsis_id"]
    g2e = {g: int(e) for e, g in e2g.items()}
    g = e2g.get(int(espn_id))
    meta = ids[ids.gsis_id == g].iloc[0] if g is not None and (ids.gsis_id == g).any() else None
    d_w = cur[cur.gsis == g].sort_values("week") if g is not None else pd.DataFrame()
    p_w = prev[prev.gsis == g].sort_values("week") if (g is not None and prev is not None) else pd.DataFrame()
    pos = (d_w.pos.iloc[0] if len(d_w) else p_w.pos.iloc[0] if len(p_w) else (meta.position if meta is not None else None))
    if pos not in ("QB", "RB", "WR", "TE"):
        return None
    season, through = ctx["season"], int(cur["week"].max())
    name = ctx["names"].get(g) or (d_w.pname.dropna().iloc[-1] if len(d_w) and "pname" in d_w and d_w.pname.notna().any() else str(espn_id))
    cur_agg = R.aggregate(cur, ("gsis", "pos")).set_index("gsis")
    if "pname" in cur:
        cur_agg["pname"] = cur.dropna(subset=["pname"]).groupby("gsis")["pname"].last()
    prev_agg = R.aggregate(prev, ("gsis", "pos")).set_index("gsis") if prev is not None else pd.DataFrame()
    from . import roles_pff as rp
    pff = R.pff_tables(season, cur, prev, ids)
    cur_agg = rp.join(cur_agg, pff[0])
    if len(prev_agg):
        prev_agg = rp.join(prev_agg, pff[1])
    c = cur_agg.loc[g] if g in cur_agg.index else None
    p = prev_agg.loc[g] if (g in prev_agg.index and prev_agg.loc[g]["games"] >= 3) else None
    games = float(c["games"]) if c is not None else 0.0
    if len(d_w):
        prof, fp = R.profile_one(model, g, d_w, c, p, R._context(cur), ctx["names"], through)
    else:  # hasn't played this season: describe him from last season
        fp = R.fingerprint(model, pos, None, 0, p)
        prof = {"pos": pos, "tier": "No snaps this season", "style": R.style_of(model, pos, None, p, 0), "traits": [], "trends": [], "series": None}
    # peer rank this season (by raw value, players with 2+ games and some volume)
    peers = cur_agg[(cur_agg.pos == pos) & (cur_agg.games >= 2)]
    for m in fp:
        col = peers[m["key"]].dropna() if m["key"] in peers else pd.Series(dtype=float)
        if m["raw"] is not None and len(col) > 10:
            better = (col > m["raw"]).sum() if not m["lower_better"] else (col < m["raw"]).sum()
            m["rank"] = int(better + 1); m["of"] = int(len(col))
    # game log
    logs = []
    team_weeks = sorted(set(int(x) for x in cur[cur.posteam == (d_w.posteam.iloc[-1] if len(d_w) else None)]["week"])) if len(d_w) else []
    rows = {int(r.week): r for r in d_w.itertuples()}
    for wk in sorted(set(team_weeks) | set(ctx["espn_pts"]) | set(ctx["espn_proj"])):
        if wk > through + 1 or wk < 1:
            continue
        r = rows.get(wk)
        logs.append({"week": wk, "played": r is not None, "opp": getattr(r, "opp", None) if r is not None else None,
                     "snap": _f(r.snap, 0) if r is not None else None,
                     "tgt": int(r.tgt) if r is not None else None, "rec": int(r.recs) if r is not None else None,
                     "rec_yd": int(r.rec_yd) if r is not None else None, "rec_td": int(r.rec_td) if r is not None else None,
                     "car": int(r.car) if r is not None else None, "rush_yd": int(r.rush_yd) if r is not None else None,
                     "rush_td": int(r.rush_td) if r is not None else None,
                     "cmp": int(r.cmp) if r is not None else None, "att": int(r.pass_att) if r is not None else None,
                     "pass_yd": int(r.pass_yd) if r is not None else None, "pass_td": int(r.pass_td) if r is not None else None,
                     "ints": int(r.ints) if r is not None else None, "fl": int(r.fl) if r is not None else None,
                     "ppr": _f(r.ppr) if r is not None else None, "espn_pts": _f(ctx["espn_pts"].get(wk)), "espn_proj": _f(ctx["espn_proj"].get(wk)),
                     "upcoming": wk > through})
    # comps at his current checkpoint
    comp = None
    if c is not None and games >= 2:
        row = {k: c.get(k) for k in ["rec_mix", "pd_ratio", "adot", "behind_r", "deep_r", "tgt_per_snap", "des_rush_g", "scr_rate", "p_adot", "snap"]}
        row["ppr_g"] = float(c["ppr_g"]); row["games"] = games
        row["share"] = c.get("opp_sh") if pos == "RB" else c.get("db_sh") if pos == "QB" else c.get("tgt_sh")
        row["prev_ppg"] = float(p["ppr_g"]) if p is not None else np.nan; row["prev_games"] = float(p["games"]) if p is not None else 0
        row["age"] = _age(meta.birthdate, season) if meta is not None and "birthdate" in meta and pd.notna(meta.birthdate) else np.nan
        dy = meta.draft_year if meta is not None and "draft_year" in meta and pd.notna(meta.get("draft_year")) else None
        row["exp"] = season - int(dy) if dy else (3 if p is not None else 0)
        sn = d_w.snap.values
        row["trend"] = float(sn[-2:].mean() - sn.mean()) if len(sn) >= 2 else 0.0
        comp = C.find(pos, row, min(through, 14))
        if comp:
            comp["me"] = {"ppg": round(row["ppr_g"], 1), "age": row["age"], "exp": row["exp"], "prev_ppg": None if np.isnan(row["prev_ppg"]) else round(row["prev_ppg"], 1)}
            comp["bt"] = model["comps"]["bt"].get(pos)
            comp["starter_ppg"] = model.get("starter_ppg", {}).get(pos)
    owners = ctx.get("owners", {})
    led = ctx.get("ledger") or {}
    est = led.get("est_ppg")
    out = {
        "id": int(espn_id), "name": name, "pos": pos, "team": d_w.posteam.iloc[-1] if len(d_w) else (meta.team if meta is not None else None),
        "season": season, "through_week": through, "week": ctx.get("week"),
        "age": _age(meta.birthdate, season) if meta is not None and pd.notna(meta.get("birthdate")) else None,
        "draft": None if meta is None or pd.isna(meta.get("draft_year")) else {"year": int(meta.draft_year), "round": None if pd.isna(meta.get("draft_round")) else int(meta.draft_round),
                                                                              "pick": None if pd.isna(meta.get("draft_pick")) else int(meta.draft_pick)},
        "owner": owners.get(int(espn_id)), "ledger": led, "espn_week": _f(ctx["espn_proj"].get(ctx.get("week"))),
        "season_line": None if c is None else {"games": int(games), "ppr_g": _f(c["ppr_g"]), "snap": _f(c["snap"], 0),
                                                "espn_g": _f(np.mean([v for v in ctx["espn_pts"].values() if v is not None])) if ctx["espn_pts"] else None,
                                                "rank": int((peers.ppr_g > c["ppr_g"]).sum() + 1) if len(peers) else None, "of": int(len(peers))},
        "last_season": None if p is None else {"games": int(p["games"]), "ppr_g": _f(p["ppr_g"]), "snap": _f(p["snap"], 0)},
        "role": prof, "fingerprint": fp,
        "style_info": _style_info(model, pos, prof.get("style")),
        "map": _map(model, pos, cur_agg, prev_agg, d_w if len(d_w) else None, g, owners, g2e),
        "gamelog": logs, "comps": comp,
    }
    out["read"] = _read(pos, name, prof, fp, comp, est, model, int(games), out["espn_week"], logs)
    return out


def engine_build(espn_id: int) -> dict | None:
    """Collects the context from the engine's data and builds the file."""
    from . import data as d, views
    season, cur, prev, ids = R.tables()
    inp = d.load()
    names = dict(zip(ids["gsis_id"], ids["name"]))
    pw = inp.players_weekly[inp.players_weekly["espn_id"] == espn_id]
    pts = {int(r.week): float(r.points) for r in pw.itertuples() if bool(getattr(r, "played", True)) and int(r.week) >= 1}
    pr = inp.espn_proj[inp.espn_proj["espn_id"] == espn_id]
    proj = {int(r.week): float(r.espn_proj) for r in pr.itertuples() if int(r.week) >= 1}
    lp = inp.ledger_players.set_index("espn_id")
    led = None
    if espn_id in lp.index:
        L = lp.loc[espn_id]
        led = {"est_ppg": _f(L.get("est_ppg")), "avail_now": _f(L.get("avail_now"), 3), "avail_ros": _f(L.get("avail"), 3),
               "ecr_rank": _f(L.get("ecr_rank"), 0), "status": L.get("status") if isinstance(L.get("status"), str) else None,
               "bye": int(L["bye"]) if pd.notna(L.get("bye")) else None}
    wk = inp.rosters["week"].max()
    ro = inp.rosters[inp.rosters["week"] == wk]
    owners = {int(r.espn_id): {"team_id": int(r.team_id), "name": inp.team_name(int(r.team_id)),
                               "injury": r.injury_status if isinstance(getattr(r, "injury_status", None), str) else None} for r in ro.itertuples()}
    ctx = {"season": season, "cur": cur, "prev": prev, "ids": ids, "names": names, "model": R.load_model(),
           "espn_pts": pts, "espn_proj": proj, "ledger": led, "owners": owners, "week": inp.week}
    out = build(int(espn_id), ctx)
    if out:
        ab = views._pro_team(inp).get(int(espn_id)) or out.get("team")
        out["headshot"] = views.headshot(int(espn_id), out["pos"], ab)
        out["pro_team"] = ab
        out["game"] = views._nfl_game(inp, ab, inp.week)
    return out
