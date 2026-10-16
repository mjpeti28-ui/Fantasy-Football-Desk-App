"""Weekly routine: refresh every input, snapshot it, record the model's forecasts, score past
forecasts, and print a briefing. Run Tuesdays after Monday Night Football (and optionally
again Wednesday/Thursday for a pre-game snapshot).

  python3 tools/weekly_run.py                 # full run
  python3 tools/weekly_run.py --no-refresh    # reuse current inputs
  python3 tools/weekly_run.py --snapshot-only # just save a timestamped copy of the inputs
  python3 tools/weekly_run.py --pregame       # Sunday morning: refresh, snapshot, re-record this
                                              # week's forecasts + trade shortlist before kickoff

Steps (each is independent; a failure is reported and the rest still run):
 1. refresh  : ESPN season data (espn_history), ESPN weekly projections + source scoring
               (track_projections score), FantasyPros ROS consensus (fetch_fantasypros),
               nflverse stats/injuries/depth/lines/ECR archive
 2. snapshot : history/snapshots/<season>/<UTC stamp>/ -- ESPN weekly projections, FantasyPros
               rankings, injuries, depth
 3. timing   : for last week, compare the latest pre-game snapshot of ESPN's projections with
               ESPN's final numbers (how many 0s were already known midweek)
 4. health   : tools/health_check.py
 5. ledger   : record this week's player forecasts + team values + season odds, score all
               earlier ledgers (history/ledger/<season>/)
 6. briefing : season odds, your waiver shortlist, top trade ideas, input freshness
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
PY = sys.executable
SEASON = 2026
TEAM = 3


def run(label, args, timeout=900):
    t0 = time.time()
    try:
        r = subprocess.run([PY, *args], cwd=REPO, capture_output=True, text=True, timeout=timeout)
        ok = r.returncode == 0
        tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
        print(f"[{'ok' if ok else 'FAIL'}] {label} ({time.time() - t0:.0f}s)" + ("" if ok else " :: " + " | ".join(tail)))
        return ok, r.stdout
    except subprocess.TimeoutExpired:
        print(f"[FAIL] {label}: timed out after {timeout}s")
        return False, ""


def refresh():
    run("ESPN season data", ["espn_history.py", str(SEASON)])
    run("ESPN weekly projections + source scoring", ["tools/track_projections.py", "score"])
    run("FantasyPros ROS consensus", ["tools/fetch_fantasypros.py"])
    run("FantasyPros ROS projected points (API, ~25 calls)", ["tools/fetch_fantasypros_projections.py"])
    run("nflverse stats + injuries", ["nflverse_data.py", str(SEASON)])
    run("nflverse schedules/depth/ECR archive", ["nflverse_data.py", "--extras", str(SEASON)])


def snapshot() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    d = REPO / "history" / "snapshots" / str(SEASON) / stamp
    d.mkdir(parents=True, exist_ok=True)
    src = {"espn_weekly_proj.csv": REPO / "history" / "actuals" / str(SEASON) / "espn_weekly_proj.csv",
           "FantasyPros_ROS_rankings.csv": REPO / "FantasyPros_ROS_rankings.csv",
           "FantasyPros_ROS_projections.csv": REPO / "FantasyPros_ROS_projections.csv",
           "injuries.csv": REPO / "history" / "nflverse" / str(SEASON) / "injuries.csv",
           "depth.csv": REPO / "history" / "nflverse" / str(SEASON) / "depth.csv"}
    for name, f in src.items():
        if f.exists():
            shutil.copy2(f, d / name)
    try:
        n = espn_live_state(d / "espn_live.csv")
        print(f"[ok] ESPN live state: {n} players (injury tags, ownership, rosters, this week's projection)")
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] ESPN live state: {e}")
    print(f"[ok] snapshot -> {d.relative_to(REPO)}")
    return d


def espn_live_state(path: Path) -> int:
    """ESPN's *current* injury tag, ownership and team for the ~1,500 most-owned players, plus
    this week's lineup slots. ESPN only ever returns today's tag (for past weeks too), so the
    only way to fit anything on ESPN tags later is to save them at the time -- ideally Sunday
    morning before kickoff."""
    import json

    import pandas as pd

    import espn_client as ec
    import espn_history as eh
    import valuation as v

    wk = int(v.load_season(SEASON).rosters["week"].max())
    c = eh._client(SEASON)
    flt = {"players": {"limit": 1500, "sortPercOwned": {"sortPriority": 1, "sortAsc": False}}}
    k = c.fetch_endpoint(view=["kona_player_info"], params={"scoringPeriodId": wk},
                         headers={"X-Fantasy-Filter": json.dumps(flt)})
    want = f"11{SEASON}{wk}"
    rows = []
    for ent in k.get("players", []):
        x = ent.get("player", {})
        own = x.get("ownership", {}) or {}
        proj = next((float(st.get("appliedTotal") or 0.0) for st in x.get("stats", []) if st.get("id") == want), None)
        rows.append({"week": wk, "espn_id": x.get("id"), "name": x.get("fullName"),
                     "position": eh.POS.get(x.get("defaultPositionId")), "pro_team_id": x.get("proTeamId"),
                     "on_team_id": ent.get("onTeamId"), "status": ent.get("status"),
                     "injury_status": x.get("injuryStatus"), "injured": x.get("injured"),
                     "pct_owned": own.get("percentOwned"), "pct_started": own.get("percentStarted"),
                     "pct_change": own.get("percentChange"), "espn_proj": proj})
    df = pd.DataFrame(rows)
    st = ec.fetch_league_state()
    slot = {r.player_id: r.lineup_slot for t in st.teams.values() for r in t.roster}
    df["lineup_slot"] = df["espn_id"].map(slot)
    df["captured_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    df.to_csv(path, index=False)
    return len(df)


def timing_check():
    """Were ESPN's zeros for last week already there in a pre-game snapshot?"""
    import pandas as pd
    import valuation as v

    s = v.load_season(SEASON)
    done = sorted(s.weekly["week"].unique())
    if not done:
        return
    last = int(max(done))
    final = pd.read_csv(REPO / "history" / "actuals" / str(SEASON) / "espn_weekly_proj.csv")
    final = final[final["week"] == last].set_index("espn_id")["espn_proj"]
    snaps = sorted((REPO / "history" / "snapshots" / str(SEASON)).glob("*/espn_weekly_proj.csv"))
    snaps += sorted((REPO / "history" / "actuals" / str(SEASON)).glob(f"espn_weekly_proj_snapshot_*_wk{last}.csv"))
    sched = pd.read_csv(REPO / "history" / "nflverse" / "schedules.csv")
    kick = pd.to_datetime(sched[(sched["season"] == SEASON) & (sched["week"] == last)]["gameday"]).min()
    pre = None
    for f in snaps:
        import re
        m = re.search(r"(\d{8})", f.parent.name if f.name == "espn_weekly_proj.csv" else f.name)
        if not m or pd.Timestamp(m.group(1)) >= kick:  # only snapshots taken before the week's first game
            continue
        x = pd.read_csv(f)
        if "week" in x and (x["week"] == last).any():
            pre = (f, x[x["week"] == last].set_index("espn_id")["espn_proj"])
    if pre is None:
        print(f"[--] timing: no pre-game snapshot of week {last} projections")
        return
    f, early = pre
    both = pd.concat([early.rename("early"), final.rename("final")], axis=1).dropna()
    rost = set(s.roster_at(last)["espn_id"])
    b = both[both.index.isin(rost) | (both["early"] > 5)]
    zero_final = b["final"] == 0
    known = (b["early"] == 0) & zero_final
    print(f"[ok] timing wk{last} ({f.parent.name if f.parent.name != str(SEASON) else f.name}): "
          f"{int(zero_final.sum())} relevant players ended at 0; {int(known.sum())} were already 0 in the snapshot "
          f"({known.sum() / max(zero_final.sum(), 1):.0%}); mean |change| {float((b['final'] - b['early']).abs().mean()):.2f} pts")


def _archive_ledger():
    """Keep the earlier (e.g. Tuesday/Friday) ledger for this week before a pre-game rewrite."""
    import valuation as v
    wk = int(v.load_season(SEASON).rosters["week"].max())
    d = REPO / "history" / "ledger" / str(SEASON)
    arch = d / "archive" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    for f in d.glob(f"wk{wk:02d}_*.csv"):
        arch.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, arch / f.name)


def ledger_step():
    import pandas as pd

    import ledger
    import season_sim as ss
    import trade_value as tv
    import valuation as v
    sys.path.insert(0, str(REPO / "tools"))
    import season_odds as so

    ctx = tv.load_context(SEASON)
    repl = ctx.replacement
    weeks = range(ctx.week, v.FANTASY_LAST_WEEK + 1)
    tvals = {t: v.team_value(ids, ctx.tbl, ctx.season.slots, weeks, repl, sims=150, seed=7) for t, ids in ctx.rosters.items()}
    odds = None
    try:
        odds = ss.simulate(ctx.season, ctx.week, ctx.params, so.ratios(), rosters=ctx.rosters, n_sims=2000, seed=1,
                           tbl=ctx.tbl, replacement=repl)[["seed1", "top2", "playoffs", "title", "reg_pts"]]
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] season odds: {e}")
    wkly = ctx.season.weekly
    pre = set(wkly.loc[(wkly["week"] == ctx.week) & wkly["played"], "espn_id"].astype(int))
    f = ledger.write(SEASON, ctx.week, ctx.tbl, ctx.rosters, tvals, odds, already_played=pre)
    if pre:
        print(f"[ok] {len(pre)} players already played week {ctx.week}; their week-{ctx.week} forecast is blanked")
    print(f"[ok] ledger -> {f.relative_to(REPO)}")
    try:
        f2 = ledger.write_sources(SEASON, ctx.week, ctx.tbl)
        print(f"[ok] projection sources -> {f2.relative_to(REPO)}")
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] projection sources: {e}")
    res = ledger.score(SEASON)
    if len(res):
        pd.set_option("display.width", 220)
        print("\nFORECAST SCORECARD (earlier ledgers vs what happened)")
        print(res.round(3).to_string(index=False))
    cw = ledger.calibration_watch(SEASON)
    if len(cw):
        print("\nCALIBRATION WATCH (realized / expected so far, shrunk; flag = outside 0.85-1.15; not applied to forecasts)")
        print(cw.round(3).to_string(index=False))
    src = ledger.score_sources(SEASON)
    if len(src):
        print("\nPROJECTION SOURCES, per-game rest of season vs realized (mean over ledger weeks; same players)")
        print(src.groupby("source")[["rank_corr", "mae", "bias"]].mean().round(3).sort_values("rank_corr", ascending=False).to_string())
    return ctx, tvals, odds


def briefing(ctx, tvals, odds):
    import pandas as pd

    names = {t: ctx.team_name(t).strip() for t in ctx.rosters}
    print(f"\nBRIEFING -- {SEASON} week {ctx.week}")
    t = pd.DataFrame({"ros_value": pd.Series(tvals)})
    if odds is not None:
        t = t.join(odds)
    t.index = [names[i] for i in t.index]
    print(t.sort_values("ros_value", ascending=False).round(3).to_string())
    run_out = run("waiver shortlist", ["tools/waivers.py"], timeout=600)
    if run_out[0]:
        print(run_out[1])
    ok, out = run("trade ideas", ["tools/trade_value.py", "find", "--team", str(TEAM)], timeout=900)
    if ok:
        print("\n".join(out.splitlines()[:8]))
    # input freshness
    for f in ("FantasyPros_ROS_rankings.csv", f"history/nflverse/{SEASON}/injuries.csv", f"history/nflverse/{SEASON}/depth.csv",
              f"history/actuals/{SEASON}/espn_weekly_proj.csv"):
        p = REPO / f
        if p.exists():
            age = (time.time() - p.stat().st_mtime) / 3600
            print(f"  input {f}: {age:.0f}h old" + ("  <-- STALE" if age > 96 else ""))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-refresh", action="store_true")
    ap.add_argument("--snapshot-only", action="store_true")
    ap.add_argument("--pregame", action="store_true",
                    help="Sunday before kickoff: refresh, snapshot, re-write this week's ledger (the previous "
                         "version is archived) and log the trade shortlist")
    a = ap.parse_args(argv)
    from env import load_local_env
    load_local_env()
    if a.pregame:
        if not a.no_refresh:
            refresh()
        snapshot()
        _archive_ledger()
        ledger_step()
        run("trade shortlist ledger", ["tools/log_trades.py"])
        return 0
    if a.snapshot_only:
        # midweek: pull ESPN's current projections for the upcoming week + injury reports, then save them
        import valuation as v
        wk = int(v.load_season(SEASON).rosters["week"].max())
        run(f"ESPN week {wk} projections", ["espn_history.py", "proj", str(SEASON), str(wk), str(wk)])
        run("nflverse injuries", ["nflverse_data.py", str(SEASON)])
        snapshot()
        return 0
    if not a.no_refresh:
        refresh()
    snapshot()
    try:
        timing_check()
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] timing check: {e}")
    run("health check", ["tools/health_check.py"])
    _archive_ledger()  # a rerun later in the week (e.g. Desk's Full refresh) keeps the earlier forecasts
    ctx, tvals, odds = ledger_step()
    briefing(ctx, tvals, odds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
