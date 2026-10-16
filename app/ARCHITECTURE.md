# Desk — architecture (draft 2026-10-07)

A local macOS app for deliberating on fantasy decisions with the v2 model's numbers and evidence
side by side. Read-only against ESPN for now. Nothing here changes `weekly_run.py` or the CLI tools.

## Shape

```
Tauri window (macOS)  ──spawns──▶  Python engine (localhost:8765, FastAPI)
  React + TS UI                       app/engine/*  ──imports──▶  v2 modules (valuation, trade_value,
  fetches /api/*                                                    season_sim, ledger) — unchanged
```

- **UI:** `app/ui/` — Vite + React + TypeScript, Tauri v2 shell in `app/ui/src-tauri/`.
  Charts are hand-built SVG plus Observable Plot where it saves time; cmdk for the command palette.
  Editorial look: serif headlines, sans UI, tabular numerals; dark default with light toggle.
- **Engine:** `app/engine/` — a thin, read-only layer. It never writes to model params or the ledger.
  - `data.py`: loads cached artifacts (ledger week files, actuals, ESPN weekly projections, schedules,
    ratio distributions). Fast: no simulation, no network.
  - `views.py`: pure functions that return JSON-ready dicts per screen (`team_week`, `power_table`,
    later `player_card`, `trade_eval`, `waivers`).
  - `server.py`: FastAPI routes over `views` plus `/api/refresh` (background job, progress polling).
  - `export.py`: Markdown briefs for "copy to Claude".
  - `journal.py`: decision journal (JSONL in `history/journal/`), scored later against outcomes.
  - `snapshot.py`: CLI that dumps every view to one JSON file (used for design mockups and for
    instant startup before the server answers).

## Data flow and freshness
1. On open, the UI renders the last snapshot immediately, then asks the engine for live views.
2. Live views read the ledger the Tuesday `weekly_run` already produced (team odds, player
   expectations) plus the latest ESPN weekly projections — milliseconds, no simulation.
3. **Refresh** (button, and once on open if data is older than ~12h): pull ESPN rosters/projections
   (`espn_history.py proj`, `track_projections.py score` equivalents), rebuild `trade_value.load_context`,
   rerun the season sim. Runs in the background; every panel shows its data age.
4. Heavy weekly work (nflverse, FantasyPros, ledger scoring) stays in `weekly_run.py`.

## Decisions taken with Max (2026-10-07)
- Tauri desktop app, laptop only. Editorial / data-journalism aesthetic, dark default.
- Dense tables with drill-down drawers. Point estimate first, range on hover.
- Annotated panels: one-sentence takeaway + labelled chart.
- Player headshots (ESPN CDN).
- First screens: **My Team** (centerpiece: this week's lineup) and **League** (centerpiece: power table).
- Extras: decision journal, news feed (source TBD, must be legal to fetch), command palette.
- Claude: no in-app API; "copy as Markdown" export on every screen/player/trade.
- ESPN writes later (lineup, add/drop via `lm-api-writes` `/transactions/`), with dry-run diff,
  confirm, re-read verification. Trades only if the request is captured from DevTools.
- Check-ins: design system first, then each screen.

## Honesty rules baked into the UI
- Trade edges < 10 pts render as "noise", < 20 as "thin" (backtest: 47-61% direction-right).
- Start/sit uses ESPN's weekly number (best next-week source in backtests); ours is shown beside it.
- Title odds always last and labelled near-lottery; seed/playoff odds lead.
- Usage/xFP appear as evidence only; they are not model inputs (they failed every test).

## Running it
One-time (macOS): install Node (e.g. `brew install node`) and Rust (`curl https://sh.rustup.rs -sSf | sh`),
then `cd app/ui && npm install`. The repo's `.venv` needs fastapi + uvicorn (already in requirements.txt).
Run: `app/run.sh` — writes a fresh sample snapshot, then `npm run tauri dev`. The Tauri shell starts the
engine (`uvicorn app.engine.server:app` on 127.0.0.1:8765) and stops it on quit. First Rust build takes a
few minutes. Browser-only alternative: `python -m uvicorn app.engine.server:app --port 8765` in one terminal,
`cd app/ui && npm run dev` in another, open http://localhost:5173.
Checks done in the cloud: `tsc`, `vite build`, and `cargo check` of the Tauri shell (Linux) all pass;
the macOS window itself is untested until it runs on the Mac.
