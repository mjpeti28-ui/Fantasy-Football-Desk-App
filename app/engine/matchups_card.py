"""The Matchups card: one player against one defense, written for a start/sit call.

What it is built on (backtest/matchup_mechanisms.py, results in matchups_credit.json, 2023-25):
  * Each defense is rated from every play (matchups_rating.py). That rating does predict the next game's offense EPA.
  * ESPN's weekly projection already moves with it, by a measured amount per position ("priced in"). The amount is a share of the
    player's own projection (a 16-point back has more to move than a 6-point one); that fits ESPN's behaviour as well as one number per position.
  * Beyond ESPN's number the same rating explains nothing usable, so the card never adjusts the projection;
    it says how much of the number is the matchup and what the number cannot see.
  * The man/zone lens is a real player trait but a defense's coverage in a given game is not foreseeable
    (and the current season has no coverage tags), so it is shown as a sensitivity, tagged "too small".
Tags on each row: priced (measured, inside ESPN's number), small (real but moves a starter by a fraction of a target),
context (descriptive only, not tested against ESPN), role (from the Roles screen).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import data as d, matchups as m, matchups_data as md, matchups_rating as mr

CREDIT = Path(__file__).with_name("matchups_credit.json")
DEAD = {"OUT", "INJURY_RESERVE", "SUSPENSION"}
SPEC = {"QB": ["pass_r"], "RB": ["pass_r", "run_r"], "WR": ["pass_r"], "TE": ["pass_r"]}
KIND = {"pass_r": "pass", "run_r": "run"}
DB_GAME = 36.0
_EMPTY = pd.DataFrame({"epa": pd.Series(dtype=float), "n": pd.Series(dtype=float)})
_J: dict = {"mt": None, "v": {}}
_G: dict = {"key": None, "v": None}


def credit_table() -> dict:
    mt = CREDIT.stat().st_mtime if CREDIT.exists() else None
    if _J["mt"] != mt:
        _J["v"] = json.loads(CREDIT.read_text()) if mt else {}
        _J["mt"] = mt
    return _J["v"]


def sg(x: float, nd: int = 1) -> str:
    t = f"{abs(x):.{nd}f}"
    return t if float(t) == 0 else ("+" if x > 0 else "−") + t


def ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def scaled(cj: dict, pos: str, c: str, espn: float | None) -> dict:
    """Points per unit of rating, per sd and its range for a player projected at `espn` (or the position's typical projection when he has none).
    Uses the share-of-projection fit; falls back to the flat per-position number if the credit file predates it."""
    cr = cj["credit"][pos][c]
    if "pct_per_unit" not in cr:
        return {"per_unit": cr["per_unit"], "per_sd": cr["per_sd"], "ci": cr["ci_per_sd"], "pct": None, "pct_ci": None, "base": None}
    base = float(espn) if espn else float(cj["credit"][pos].get("mean_proj") or 10.0)
    return {"per_unit": cr["pct_per_unit"] * base, "per_sd": cr["pct_per_sd"] * base, "ci": [cr["ci_pct_per_sd"][0] * base, cr["ci_pct_per_sd"][1] * base],
            "pct": cr["pct_per_sd"], "pct_ci": cr["ci_pct_per_sd"], "base": base}


# ---------- defense grades ----------
def grades() -> dict:
    """abbr -> {'pass': {...}, 'run': {...}}. rating = EPA allowed per play vs the league, higher = softer; rank 1 = toughest."""
    t = m.tables()
    key = m._stamp()
    if _G["key"] == key and _G["v"] is not None:
        return _G["v"]
    C, cur = t["C"], int(t["cur"])
    try:
        P = md.load(cur - 1)
    except Exception:
        P = None
    cj = credit_table()
    out: dict = {}
    for kind in ("pass", "run"):
        pr = mr.team_table(P, kind) if P is not None else _EMPTY
        cu = mr.team_table(C, kind)
        r = mr.rate(pr, cu)
        r = r - r.mean()
        sd_ref = float((cj.get("sd_ref") or {}).get(kind) or r.std() or 1.0)
        rank = r.rank(method="min").astype(int)
        for a in r.index:
            n = float(cu["n"].get(a, 0.0)) if len(cu) else 0.0
            k = int(rank[a])
            out.setdefault(a, {})[kind] = {"r": float(r[a]), "z": float(r[a] / sd_ref), "rank": k, "of": int(len(r)),
                                          "label": "Tough" if k <= 11 else "Soft" if k >= 22 else "Middle", "w_now": n / (n + mr.K_PLAYS), "plays": int(n)}
    _G.update(key=key, v=out)
    return out


def annotate_board(b: dict) -> dict:
    """Add the pass/run grades to each defense in the board."""
    g = grades()
    for a, row in b["teams"].items():
        x = g.get(a)
        if x:
            row["pass_grade"], row["run_grade"] = x["pass"], x["run"]
    return b


# ---------- one player ----------
def _proj(eid: int, snap: dict) -> dict:
    tw = snap.get("team_week") or {}
    pools = list(tw.get("players") or []) + list((tw.get("matchup") or {}).get("opp_starters") or [])
    for o in pools:
        if int(o["id"]) == eid:
            g = o.get("game") or {}
            return {"espn": o.get("espn_week"), "model": o.get("est_ppg"), "injury": o.get("injury") or "ACTIVE", "bye": bool(g.get("bye")), "listed": True}
    inp = d.load(int(snap["season"]))
    r = inp.espn_proj[(inp.espn_proj["week"] == int(snap["week"])) & (inp.espn_proj["espn_id"] == eid)]
    lp = inp.ledger_players
    est = lp[lp["espn_id"] == eid]["est_ppg"]
    return {"espn": float(r["espn_proj"].iloc[0]) if len(r) else None, "model": float(est.iloc[0]) if len(est) else None, "injury": None, "bye": False, "listed": False}


def _role(eid: int) -> dict | None:
    try:
        from . import roles
        p = roles.board()["players"].get(str(eid))
    except Exception:
        return None
    if not p:
        return None
    tr = (p.get("trends") or [None])[0]
    style = p.get("style") or {}
    return {"tier": p.get("tier"), "style": style.get("name"), "trend": tr.get("label") if tr else None, "trend_text": tr.get("text") if tr else None}


def _row(key, label, him, them, effect, tag, tip) -> dict:
    return {"key": key, "label": label, "him": him, "them": them, "effect": effect, "tag": tag, "tip": tip}


def build(pv: dict, snap: dict) -> dict:
    """pv: matchups.player_view() output. Returns the card; see the module docstring for what each tag means."""
    pos, eid, opp = pv["pos"], int(pv["espn_id"]), pv.get("opp")
    cj = credit_table()
    pr = _proj(eid, snap)
    first = pv["name"].split(" ")[0]
    dead = pr["bye"] or (pr["injury"] in DEAD) or not pr["espn"]
    try:
        sched = (m.schedule(int(snap["season"]), int(snap["week"])).get(pv.get("team")) or {}).get("opp")
    except Exception:
        sched = None
    through = bool(opp) and opp != sched  # "Look through": a defense he is not playing this week
    card = {"espn_id": eid, "name": pv["name"], "pos": pos, "team": pv.get("team"), "opp": opp, "proj": pr, "dead": bool(dead), "through": through, "sched": sched,
            "week": int(snap["week"]), "season": int(snap["season"]), "rows": [], "priced": None, "bottom": None,
            "miss_sd": (cj.get("miss_sd") or {}).get(pos), "role": _role(eid)}
    if not opp or (pr["bye"] and not through):
        card["bottom"] = {"tone": "bye", "headline": "Bye week.", "lines": ["There is no defense to read him against this week. Use “Look through” to see him against any team."]}
        return card
    if pos not in SPEC or not cj.get("credit"):
        card["bottom"] = {"tone": "none", "headline": "No matchup card for this position.", "lines": []}
        return card
    g = grades().get(opp)
    if not g:
        card["bottom"] = {"tone": "none", "headline": f"No ratings for {opp} yet.", "lines": []}
        return card
    card["defense"] = {"abbr": opp, "pass": g["pass"], "run": g["run"]}
    miss = card["miss_sd"] or 7.5
    # ---- priced in: ESPN's own movement with the defense, per position, measured ----
    parts, rows = [], []
    for c in SPEC[pos]:
        kind = KIND[c]
        gr = g[kind]
        cr, lo = scaled(cj, pos, c, pr["espn"]), cj["left_over"][pos][c]
        pts = cr["per_unit"] * gr["r"]
        parts.append({"kind": kind, "pts": pts, "per_sd": cr["per_sd"], "ci": cr["ci"], "pct": cr["pct"], "pct_ci": cr["pct_ci"], "left": lo["per_sd"], "left_ci": lo["ci_per_sd"], "z": gr["z"]})
        what = "pass" if kind == "pass" else "run"
        if cr["pct"] is not None:
            worth = (f"In 2023–25 ESPN\u2019s projection for {pos}s moved {cr['pct'] * 100:.1f}% per sd of softer {what} defense (range {sg(cr['pct_ci'][0] * 100)}% to {sg(cr['pct_ci'][1] * 100)}%), "
                     f"so a bigger role has more to move: {sg(cr['per_sd'])} pts per sd on his {cr['base']:.1f} (range {sg(cr['ci'][0])} to {sg(cr['ci'][1])}).")
        else:
            worth = f"In 2023–25 a {what} defense one standard deviation (sd) softer was worth {sg(cr['per_sd'])} pts to {pos}s inside ESPN\u2019s projection (range {sg(cr['ci'][0])} to {sg(cr['ci'][1])})."
        tip = (f"Rated on every play: EPA allowed per {'dropback' if kind == 'pass' else 'rush'}, this season so far (counts {gr['w_now']:.0%} now) blended with last season. "
               f"Higher = softer; rank 1 = toughest of {gr['of']}. {worth} After ESPN\u2019s number the same rating explained {sg(lo['per_sd'])} pts per sd "
               f"(range {sg(lo['ci_per_sd'][0])} to {sg(lo['ci_per_sd'][1])}) for the position: nothing usable. A defense\u2019s rating does predict the next game\u2019s offense EPA "
               f"(slope {cj['mechanism']['slope']:.2f}, range {cj['mechanism']['ci'][0]:.2f} to {cj['mechanism']['ci'][1]:.2f}), which is why ESPN already uses it.")
        rows.append(_row(kind, f"{'Pass' if kind == 'pass' else 'Run'} defense", None, f"{gr['label']}, {ordinal(gr['rank'])} of {gr['of']}",
                         f"{sg(pts)} pts", "priced", tip))
    card["priced"] = {"pts": sum(p["pts"] for p in parts), "parts": parts}
    # ---- coverage mix (receivers and backs) ----
    lens, h = pv.get("lens"), pv.get("hist")
    lj = cj.get("lens") or {}
    lens_line = None
    if pos in ("WR", "TE", "RB") and lens and h and h.get("tr_diff_shrunk") is not None:
        per10 = h["tr_diff_shrunk"] * 0.10 * DB_GAME
        lens_line = (f"Coverage mix: {first} draws {abs(h['tr_diff_shrunk']) * 100:.1f} points {'more' if h['tr_diff_shrunk'] >= 0 else 'less'} target rate against man than zone "
                     f"(pulled toward the {pos} norm; {h['tr_rel'] * 100:.0f}% of a gap this size is real). {opp} is expected to play man {lens['opp_man_exp'] * 100:.0f}% of the time against "
                     f"{lens['league_man'] * 100:.0f}% for a typical defense, which moves him {sg(lens['d_targets'], 2)} targets.")
        tip = (f"Each 10 points of man rate moves him {sg(per10, 2)} targets. Measured 2024–25, out of sample: when the game’s real coverage is known this gap is calibrated "
               f"(slope {lj.get('slope_known', float('nan')):.2f}, range {lj.get('ci_known', [float('nan')] * 2)[0]:.2f} to {lj.get('ci_known', [float('nan')] * 2)[1]:.2f}). "
               f"But a defense’s man rate is predictable to only about {lj.get('sd_expected_effect', float('nan')):.2f} targets (a game swings ±{lj.get('sd_targets_miss', float('nan')):.1f}), "
               f"and the current season has no coverage tags to update it. A real trait, not a forecast.")
        rows.append(_row("lens", "Coverage mix", f"man − zone {sg(h['tr_diff_shrunk'] * 100)} pts ({h['tr_rel'] * 100:.0f}% real)",
                         f"man {lens['opp_man_exp'] * 100:.0f}% (typical {lens['league_man'] * 100:.0f}%)", f"{sg(lens['d_targets'], 2)} targets vs his usual", "small", tip))
    # ---- pressure (quarterbacks) ----
    if pos == "QB":
        dd = m.defense(opp)
        last = dd["history"][-1] if dd and dd["history"] else None
        qh = pv.get("hist")
        if last and qh:
            lgp = dd["league"][str(last["season"])]["pressure"]
            exp_p = lgp + dd["persist"]["pressure"] * (last["pressure"] - lgp)
            rows.append(_row("pressure", "Pass rush", f"pressured on {qh['total']['pressure'] * 100:.0f}% of dropbacks, sacked {qh['total']['sack'] * 100:.1f}%",
                             f"{opp} pressure {exp_p * 100:.0f}% expected (league {lgp * 100:.0f}%)", None, "context",
                             "Pressure carries over from last season at about half strength. It is not tested separately: it is part of what the pass-defense rating above already measures."))
    # ---- a back who lives on early-down carries: the one style that measurably responds more to a defense ----
    sr = cj.get("style_run")
    ro = card["role"]
    if pos == "RB" and sr and ro and ro.get("style") == "Early-down runner" and "Early-down runner" in (sr.get("styles") or {}):
        e, o1, o2 = sr["styles"]["Early-down runner"], sr["styles"]["All-purpose back"], sr["styles"]["Passing-down back"]
        other = (o1["per_sd"] + o2["per_sd"]) / 2
        eras = sr.get("eras") or {}
        lab = {"e16_19": "2016–19", "e20_22": "2020–22", "e23_25": "2023–25"}
        rep = ", ".join(f"{lab.get(k, k)} {sg(v['Early-down runner'])}" for k, v in eras.items())
        gr = g["run"]
        rows.append(_row("style_run", "Early-down runner", "run defense matters more for this style", f"{gr['label']}, {ordinal(gr['rank'])} of {gr['of']}", None, "context",
                         f"Measured on {sr['n_games']:,} player-games, {sr['seasons'].replace('-', '–')}: per sd of softer run defense, early-down runners score {sg(e['per_sd'])} pts (± {e['se']:.2f}) against "
                         f"{sg(o1['per_sd'])} for all-purpose and {sg(o2['per_sd'])} for passing-down backs, about {e['per_sd'] / other:.0f}\u00d7 as much. It appeared in every era ({rep}) and holds among backs with a real workload. "
                         f"Whether ESPN prices the extra sensitivity is not known (what is left after its number is within noise), so this explains the number and does not change it."))
    # ---- role ----
    if ro and (ro.get("tier") or ro.get("trend")):
        rows.append(_row("role", "Role", f"{ro.get('tier') or ''}{' · ' + ro['style'] if ro.get('style') else ''}", ro.get("trend_text") or ro.get("trend"), None, "role",
                         "From the Roles screen. Usage trends describe what changed; in backtests they added almost nothing once ESPN’s number and the consensus rankings were known."))
    card["rows"] = rows
    # ---- bottom line ----
    if through:
        real = ""
        gs = grades().get(sched) if sched else None
        if gs:
            ps = sum(scaled(cj, pos, c, pr["espn"])["per_unit"] * gs[KIND[c]]["r"] for c in SPEC[pos])
            real = f" His real game is against {sched}, where the same measure is {sg(ps)}."
        elif not sched:
            real = " He has no game this week."
        run = f" The run defense is {g['run']['label'].lower()} ({ordinal(g['run']['rank'])})." if len(parts) > 1 else ""
        card["bottom"] = {"tone": "whatif", "headline": f"Against {opp}, the matchup would be worth {sg(card['priced']['pts'])} points to {first}.",
                          "lines": [f"{opp}\u2019s pass defense is {g['pass']['label'].lower()} ({ordinal(g['pass']['rank'])} of {g['pass']['of']}, 1 = toughest).{run}{real}",
                                    "ESPN\u2019s number on this card is for his real game. This view shows how much a defense like this adds or takes away inside a projection; it is not a forecast for a game that is not on the schedule."]}
        return card
    if dead:
        why = "is out" if pr["injury"] in DEAD else "has no ESPN projection"
        card["bottom"] = {"tone": "out", "headline": f"{first} {why} this week.", "lines": ["Nothing in the matchup matters until he is cleared. Check the injury report."]}
        return card
    p0 = parts[0]
    zero = p0["left_ci"][0] <= 0 <= p0["left_ci"][1]
    run = (f" The run defense is {g['run']['label'].lower()} ({ordinal(g['run']['rank'])}), worth {sg(parts[1]['pts'])}." if len(parts) > 1 else "")
    lines = [f"{opp}\u2019s pass defense is {g['pass']['label'].lower()} ({ordinal(g['pass']['rank'])} of {g['pass']['of']}, 1 = toughest). About {sg(card['priced']['pts'])} of his {pr['espn']:.1f} "
             f"is that matchup, and a normal week misses by \u00b1{miss:.1f}." + run,
             (f"Once ESPN\u2019s number was counted, the rating still explained {sg(p0['left'])} pts per sd in 2023\u201325 (range {sg(p0['left_ci'][0])} to {sg(p0['left_ci'][1])}): "
              + ("nothing to add or subtract." if zero else "outside noise in the backtest, so weigh it as a small lean."))]
    inj = pr["injury"]
    lines.append(("He is listed " + inj.replace("_", " ").lower() + ". " if inj and inj not in ("ACTIVE", "NORMAL") else "") +
                 "Not in the number: injury news, teammates who are out, weather. Check those before kickoff.")
    card["bottom"] = {"tone": "go", "headline": f"Go with ESPN’s {pr['espn']:.1f}. The matchup is already in it.", "lines": lines}
    return card


# ---------- Compare: the same card, cut into short rows ----------
DEPTH_LABEL = {"behind": "behind the line", "short": "short (0–9 yds)", "mid": "intermediate (10–19)", "deep": "deep (20+)"}


def _rk(u: dict | None) -> str | None:
    return f"{ordinal(u['rank'])} of {u['of']}" if u else None


def compare(pv: dict, card: dict, pff: dict | None) -> dict:
    """Short rows for the Compare screen. Tags: priced (inside ESPN's number), small (real but tiny), context (descriptive, tested null as a predictor)."""
    pos, opp, first = pv["pos"], pv.get("opp"), pv["name"].split(" ")[0]
    out = {"opp": opp, "read": None, "rows": []}
    rows = out["rows"]
    add = lambda key, label, text, sub, tag, tip: rows.append({"key": key, "label": label, "text": text, "sub": sub, "tag": tag, "tip": tip})
    bt = card.get("bottom") or {}
    if card.get("dead") or not card.get("defense") or card.get("through"):
        out["read"] = bt.get("headline")
        return out
    g, pr, priced = card["defense"], card["proj"], card.get("priced") or {}
    parts = {p["kind"]: p for p in priced.get("parts", [])}
    # ---- the read: which unit faces him and how much of ESPN's number it is ----
    main = "run" if pos == "RB" and "run" in parts and abs(parts["run"]["pts"]) > abs(parts["pass"]["pts"]) else "pass"
    gm = g[main]
    tone = {"Tough": "a tough", "Soft": "a soft", "Middle": "an average"}[gm["label"]]
    out["read"] = (f"{tone.capitalize()} {main} defense ({ordinal(gm['rank'])} of {gm['of']}, 1 = toughest). "
                   f"About {sg(priced.get('pts', 0.0))} of his {pr['espn']:.1f} is the matchup, already in ESPN’s number.")
    for kind in SPEC.get(pos, []):
        k = KIND[kind]
        if k in parts:
            add(k, f"{'Pass' if k == 'pass' else 'Run'} defense", f"{g[k]['label']}, {ordinal(g[k]['rank'])} of {g[k]['of']}", f"{sg(parts[k]['pts'])} pts inside ESPN",
                "priced", next((r["tip"] for r in card["rows"] if r["key"] == k), None))
    dd = m.defense(opp) if opp else None
    cur = pv.get("current") or {}
    # ---- what this defense gives up to his position, and where he catches the ball ----
    if dd and pos in (dd.get("allowed") or {}):
        a = dd["allowed"][pos]
        add("allowed", f"Allows to {pos}s", f"{a['ppr_t']:.2f} PPR per target (league {a['league_ppr_t']:.2f})",
            f"{pos}s draw {a['share'] * 100:.0f}% of targets vs it (league {a['league_share'] * 100:.0f}%)", "context",
            f"This season, {a['tgt']} targets to {pos}s against {opp}, pulled toward the league average so a few big games don't dominate. "
            "Points allowed by position barely predict the next game once ESPN's number is known.")
    if dd and cur.get("depth") and dd.get("depth"):
        k = max(cur["depth"], key=cur["depth"].get)
        x = dd["depth"].get(k)
        if x and x["tgt"] >= 10:
            add("depth", "Where he's targeted", f"{cur['depth'][k] * 100:.0f}% {DEPTH_LABEL[k]}, aDOT {cur['adot']:.1f}",
                f"{opp} there: {x['comp'] * 100:.0f}% completions (league {x['league_comp'] * 100:.0f}%), EPA/target {sg(x['epa_t'], 2)} (league {sg(x['league_epa_t'], 2)})", "context",
                "His most common target depth this season, and how this defense has done on throws at that depth. aDOT/YAC fit vs the defense tested null against ESPN (2023-25).")
    # ---- PFF: the units in the trenches and the defenders he will likely see ----
    if pff:
        them, us = pff.get("them") or {}, pff.get("us") or {}
        wk = f"PFF grades through week {pff['through']}" if pff.get("through") else "PFF grades"
        if pos == "RB" and them.get("run_defense"):
            add("trench", "Run D vs his line", f"their run D {_rk(them['run_defense'])}", f"his run blocking {_rk(us.get('run_block')) or '–'}", "context",
                f"{wk}, snap-weighted unit grades, 1 = best. Run defense vs run blocking tested null against ESPN (2023-25).")
        elif pos in ("QB", "WR", "TE") and them.get("pass_rush"):
            add("trench", "Pass rush vs his line", f"their rush {_rk(them['pass_rush'])}, coverage {_rk(them.get('coverage')) or '–'}",
                f"his pass blocking {_rk(us.get('pass_block')) or '–'}", "context",
                f"{wk}, snap-weighted unit grades, 1 = best. Pass rush vs pass block and coverage grade tested null against ESPN (2023-25); ESPN already prices coverage.")
        if pos in ("WR", "TE") and pff.get("likely"):
            spot = pff["likely"]
            al = pff.get("align") or {}
            who = (pff.get("slot") if spot == "slot" else pff.get("outside")) or []
            pct = al.get(spot if spot != "outside" else "wide")
            if spot == "inline":
                text, sub = f"inline {pct}% of pass plays", "mostly blocks or works the seams; no single defender"
            else:
                d0 = who[0] if who else None
                text = f"{'slot' if spot == 'slot' else 'outside'} {pct}% of pass plays"
                sub = (f"likely {d0['name']}: coverage grade {d0['cov_grade'] or '–'}, {d0['yds_cov'] or '–'} yds per coverage snap" if d0 else "–")
            add("seen", "Who he'll likely see", text, sub, "context",
                "Where he lines up (PFF) and the defender who plays that spot most for them. PFF has no shadow flag, so this is the usual matchup, not a promise. CB-vs-WR matchups tested null against ESPN.")
        elif pos == "QB" and pff.get("rushers"):
            r0 = pff["rushers"][0]
            add("seen", "Top pass rusher", f"{r0['name']} ({r0['pos']})", f"{r0['pressures']} pressures, {r0['prp'] or '–'}% pressure rate", "context",
                "The defender with the most pressures this season (PFF). Pressure matchups tested null against ESPN.")
        elif pos == "RB" and pff.get("run"):
            r0 = pff["run"][0]
            add("seen", "Top run defender", f"{r0['name']} ({r0['pos']})", f"{r0['stops']} stops, run grade {r0['run_grade'] or '–'}", "context",
                "The defender with the most run stops this season (PFF). Run-defense matchups tested null against ESPN.")
    # ---- rows the card already wrote ----
    for r in card["rows"]:
        if r["key"] == "lens":
            add("lens", "Man vs zone", r["them"], f"{r['effect']} ({first}: {r['him']})", "small", r["tip"])
        elif r["key"] == "pressure":
            add("pressure", "Pressure", r["them"], f"{first}: {r['him']}", "context", r["tip"])
        elif r["key"] == "style_run":
            add("style_run", "Early-down runner", "run D matters more for this style", r["them"], "context", r["tip"])
    return out
