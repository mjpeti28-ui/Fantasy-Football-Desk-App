# Legacy: the v1 Streamlit app and API (parked 2026-10-05)

Status: **not maintained.** Kept in place so it can be revived later; nothing was deleted or moved.
There is no `streamlit-v1-final` tag in this repo (history was squashed to one commit), so the
files at the repo root are the only copy.

What's legacy (v1: VOR + combined z-score on PFF/FantasyPros projections):
- `streamlit_app.py`, `api/` (FastAPI: `uvicorn api.main:app`), `main.py`
- `trading.py` (v1 trade finder), `playoffs.py`, `simulation.py`, `epw.py`, `power_archetypes.py`,
  `power_snapshots.py`, `optimizer.py`, `reports.py`, `tools/backfill_snapshots.py`

What's current (v2, points-based, backtested on 2021-2025): `valuation.py`, `avail_model.py`,
`trade_value.py`, `season_sim.py`, `ledger.py`, and `tools/` (`trade_value.py`, `waivers.py`,
`season_odds.py`, `weekly_run.py`, `log_trades.py`, `health_check.py`). None of these import the
legacy modules (checked 2026-10-05), so the legacy code can be changed or removed without
touching the weekly flow. Exceptions: `backtest/currency_validation.py` and
`backtest/season_2025.py` still `import main` (v1 baselines), so moving `main.py` breaks them. Shared helpers both use: `espn_client.py`, `data.py`, `scoring.py`,
`env.py`, `config.py`.

Why parked: two valuation paths gave two different numbers for the same player, and v1's
trade objective is a z-score change rather than points. Backtests found v1's bench weight hurt
team prediction and VOR ~= raw points for cross-position calls (see project doc
`valuation-v2-findings.md`).

To revive it: the cleanest route is a new UI on top of the v2 modules (`trade_value.find /
evaluate`, `waivers`, `season_sim`) rather than re-wiring v1. If you just want the old app:
`pip install streamlit altair openpyxl` and `streamlit run streamlit_app.py` from the repo root
(untested since parking).
