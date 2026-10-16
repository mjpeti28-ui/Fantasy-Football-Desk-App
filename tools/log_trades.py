"""Log trade ideas with their predictions before the week's games, so the trade finder's
*live* picks can be scored later (did the side we favored actually gain, under the same
frozen-roster rules as backtest/currency_validation.py?).

  python3 tools/log_trades.py            # default find top 15 + the watchlist
  python3 tools/log_trades.py --no-find  # watchlist only

Watchlist: history/ledger/<season>/trade_watchlist.json, a list of
{"with": team_id, "give": [names], "get": [names], "note": "..."} for trades we're
considering or actually proposed. Output: history/ledger/<season>/wk<NN>_trades.csv
(appends; one row per trade per capture) with our blend's prediction plus the one-source
stress tests (ours / ESPN / experts), for both sides.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # before tools/, whose trade_value.py is the CLI

import pandas as pd  # noqa: E402

import trade_value as tv  # noqa: E402
import valuation as v  # noqa: E402

SEASON, TEAM = 2026, 3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-find", action="store_true")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args(argv)
    from env import load_local_env
    load_local_env()
    ctx = tv.load_context(SEASON)
    d = v.ROOT / "history" / "ledger" / str(SEASON)
    d.mkdir(parents=True, exist_ok=True)
    trades = []  # (origin, note, partner, give ids, get ids)
    wl = d / "trade_watchlist.json"
    if wl.exists():
        for t in json.loads(wl.read_text()):
            try:
                trades.append(("watchlist", t.get("note", ""), int(t["with"]),
                               tv.resolve(ctx, TEAM, t["give"]), tv.resolve(ctx, int(t["with"]), t["get"])))
            except Exception as e:  # noqa: BLE001  (player moved / name changed)
                print(f"skip watchlist item {t}: {e}")
    if not a.no_find:
        names = {ctx.team_name(t): t for t in ctx.rosters}
        f = tv.find(ctx, TEAM, top=a.top)
        for _, r in f.iterrows():
            o = names[r["with"]]
            trades.append(("find", "", o, tv.resolve(ctx, TEAM, r["give"].split(", ")),
                           tv.resolve(ctx, o, r["get"].split(", "))))
    ctxs = {"blend": ctx, **{s: tv.load_context(SEASON, params=tv.source_params(s)) for s in ("ours", "espn", "ecr")}}
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    rows = []
    for origin, note, other, give, get in trades:
        row = {"captured_utc": stamp, "season": SEASON, "week": ctx.week, "origin": origin, "note": note,
               "team": TEAM, "with": other, "give_ids": " ".join(map(str, give)), "get_ids": " ".join(map(str, get)),
               "give": ", ".join(ctx.tbl.at[i, "name"] for i in give), "get": ", ".join(ctx.tbl.at[i, "name"] for i in get)}
        for name, c in ctxs.items():
            ev = tv.evaluate(c, TEAM, give, other, get, sims=300)
            me, th = ev["teams"][TEAM], ev["teams"][other]
            row[f"me_{name}"], row[f"them_{name}"] = me["delta_total"], th["delta_total"]
            if name == "blend":
                row["me_playoffs"], row["p_right"] = me["delta_playoffs"], me["p_direction_right"]
                row["me_drop"], row["them_drop"] = ", ".join(me["would_drop"]), ", ".join(th["would_drop"])
        rows.append(row)
    out = d / f"wk{ctx.week:02d}_trades.csv"
    df = pd.DataFrame(rows)
    if out.exists():
        df = pd.concat([pd.read_csv(out), df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"logged {len(rows)} trades -> {out.relative_to(v.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
