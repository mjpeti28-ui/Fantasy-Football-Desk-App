#!/usr/bin/env bash
# Start Desk: (once) move the repo out of iCloud and set up its local env, stop any Desk left
# over from an earlier run, then open the Tauri window (it starts a fresh engine, which warms
# the slow views itself). After the first run you can also open Desk from Spotlight/Launchpad:
# this script installs ~/Applications/Desk.app, which runs this script for you.
#
# Why the iCloud part: the repo lived in an iCloud-synced Desktop with Optimize Mac Storage on,
# and macOS had evicted ~19,000 of ~19,600 files in .venv and nearly all of node_modules, so every
# launch re-downloaded pandas and numpy. app/relocate.sh moves the repo to ~/Developer (once);
# the Python env, Rust build and .pyc files live in ~/.local/share/desk either way.
set -euo pipefail
cd "$(dirname "$0")/.."
HERE="$(pwd)"
LOCAL="${DESK_LOCAL:-$HOME/.local/share/desk}"
mkdir -p "$LOCAL"

# ---- started from an old copy the move left behind? hand over to the real repo ----
# A Terminal tab that was sitting in the repo keeps pointing at the renamed old copy, and running
# app/run.sh there used to copy that old copy into ~/Developer all over again (10/8: four extra
# copies). Old copies carry MOVED.txt or "(old copy in iCloud)" in their name; never run from one.
old_copy() { [ -f MOVED.txt ] || case "$(basename "$(pwd -P)")" in *"(old copy in iCloud)"*) return 0 ;; *) return 1 ;; esac; }
if old_copy; then
  REAL="$(cat "$LOCAL/repo" 2>/dev/null || true)"
  [ -n "$REAL" ] && [ -f "$REAL/app/run.sh" ] || REAL="$HOME/Developer/PFF-fantasy-ranking"
  if [ -f "$REAL/app/run.sh" ] && [ "$(cd "$REAL" && pwd -P)" != "$(pwd -P)" ]; then
    echo "This folder is an old copy left behind by the move. Starting Desk from $REAL instead."
    cd "$REAL" && exec bash app/run.sh "$@"
  fi
  echo "This folder is an old copy left behind by the move, and the real repo isn't at $REAL."
  echo "Run app/run.sh from the real repo (or set DESK_LOCAL/repo). Not starting Desk from stale files."
  exit 1
fi

# ---- once: move the repo out of iCloud, then carry on from the new place ----
if [ -f app/relocate.sh ] && [ -z "${DESK_RELOCATED:-}" ]; then
  rc=0; bash app/relocate.sh || rc=$?
  if [ "$rc" -eq 0 ]; then
    NEW="$(cat "$LOCAL/relocated-to" 2>/dev/null || true)"   # relocate.sh records where the repo went
    [ -f "$NEW/app/run.sh" ] || NEW="$HOME/Developer/$(basename "$HERE")"
    echo "continuing from $NEW"
    DESK_RELOCATED=1 exec bash "$NEW/app/run.sh" "$@"
  elif [ "$rc" -eq 1 ]; then
    echo "(the move didn't happen; Desk will start from here as before)"
  fi
fi
cd "$(pwd -P)"
# remember where the real repo is (the launcher and old-copy hand-off read this)
if ! old_copy; then printf '%s\n' "$(pwd -P)" > "$LOCAL/repo"; printf '%s\n' "$(pwd -P)" > "$LOCAL/relocated-to"; fi
in_icloud() { case "$(pwd -P)" in "$HOME/Desktop/"*|"$HOME/Documents/"*|"$HOME/Library/Mobile Documents/"*) return 0 ;; *) return 1 ;; esac; }

stamp() { date +%Y%m%d%H%M%S; }
aside() {  # move $1 to $1.icloud-old (or a dated name if that's taken); prints the new name
  local to="$1.icloud-old"; [ -e "$to" ] || [ -L "$to" ] && to="$1.icloud-old-$(stamp)"
  mv "$1" "$to" && echo "$to"
}

# ---- Python env, outside the repo ----
setup_python() {
  [ -f "$LOCAL/venv/.desk-ok" ] && return 0
  local base=""
  if [ -x .venv/bin/python ]; then base=$(.venv/bin/python -c 'import sys; print(sys._base_executable)' 2>/dev/null || true); fi
  if [ -z "$base" ] || [ ! -x "$base" ]; then
    base=$(command -v python3.13 || command -v python3.12 || command -v python3.11 || command -v python3)
  fi
  echo "one-time setup: building Desk's Python env in $LOCAL/venv with $base (a minute or two)…"
  rm -rf "$LOCAL/venv"
  if "$base" -m venv "$LOCAL/venv" \
     && "$LOCAL/venv/bin/python" -m pip install -q --disable-pip-version-check -r app/requirements-desk.txt \
     && "$LOCAL/venv/bin/python" -c "import pandas, numpy, fastapi, uvicorn, httpx, dotenv, yaml"; then
    touch "$LOCAL/venv/.desk-ok"
  else
    echo "  Python setup failed (offline?). Using the repo's .venv this time; run app/run.sh again later."
    rm -rf "$LOCAL/venv"
    return 1
  fi
}
if setup_python; then
  if [ ! -L .venv ]; then
    [ -e .venv ] && echo "  old .venv kept as $(aside .venv) (safe to delete)"
    ln -s "$LOCAL/venv" .venv
  fi
  PY="$LOCAL/venv/bin/python"
else
  PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
fi
export DESK_PYTHON="$PY"
export PYTHONPYCACHEPREFIX="$LOCAL/pycache"
export CARGO_TARGET_DIR="$LOCAL/target"

# ---- UI packages ----
( cd app/ui
  # an earlier version linked node_modules to node_modules.nosync; npm replaces such links anyway
  if [ -L node_modules ] && [ ! -d node_modules/.bin ]; then rm -f node_modules; fi
  [ -d node_modules/.bin ] || { echo "installing UI packages…"; npm install --no-audit --no-fund --loglevel=error; } )
mkdir -p app/cache.nosync
[ -d "$LOCAL/target" ] || echo "first launch with the new build folder: Rust rebuilds the window shell once (a few minutes)"

# ---- Desk.app in ~/Applications, so Spotlight and Launchpad can start Desk ----
bash app/install-launcher.sh || echo "(couldn't install ~/Applications/Desk.app; app/run.sh still works)"

# ---- stop anything left from an earlier run ----
pkill -f "uvicorn app.engine.server:app" 2>/dev/null && echo "stopped an old Desk engine" || true
# a Desk window or dev server left from an earlier run holds port 5173; stop it if it's ours
for pid in $(lsof -ti tcp:5173 2>/dev/null); do
  if ps -o command= -p "$pid" | grep -q "vite"; then kill "$pid" && echo "stopped an old Desk dev server ($pid)"; fi
done
pkill -f "target/debug/desk" 2>/dev/null && echo "closed an old Desk window" || true
sleep 1

# ---- still in iCloud (move skipped): pull back evicted model inputs in the background ----
if in_icloud; then
  Y=$(date +%Y)
  ( nice find history/actuals/"$Y" history/actuals/"$((Y - 1))" history/ledger/"$Y" history/nflverse/"$Y" history/nflverse/"$((Y - 1))" \
      history/nflverse/ids.csv history/nflverse/schedules.csv history/journal history/backtest/multi/ratio_dist.json \
      schedule.csv ./*.py tools app/engine .git/objects/pack \
      -type f -not -path '*/__pycache__/*' -print0 2>/dev/null | nice xargs -0 cat >/dev/null 2>&1 || true ) &
fi

# The sample is only the fallback shown while the engine boots, so refresh it in the background.
( nice "$PY" -m app.engine.snapshot --out app/ui/public/sample-snapshot.json >/dev/null 2>&1 || echo "sample refresh failed (harmless; the app uses the engine)" ) &

echo "opening Desk; the engine precomputes waivers, roles, shifts and league odds as it starts"
cd app/ui
npm run tauri dev
