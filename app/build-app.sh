#!/usr/bin/env bash
# Build Desk as a real macOS app and install it as ~/Applications/"Desk (built).app".
#
# Why: `app/run.sh` runs the window through `tauri dev`, which is a bare program with no app bundle,
# so macOS can't list it (Spotlight and Launchpad never see it, and Claude's computer use can't be
# granted it). This makes an actual bundle (com.maxpetite.desk) that they can find.
#
# The built copy carries the UI as of the build, so run this again after UI changes (app/run.sh
# stays the way to develop). It starts its own engine from this repo, like the dev window does.
# The first build compiles the window shell's Rust in release-less "debug" mode and reuses the
# same build folder as app/run.sh, so it takes a few minutes once and under a minute after that.
set -euo pipefail
cd "$(dirname "$0")/.."
LOCAL="${DESK_LOCAL:-$HOME/.local/share/desk}"
export CARGO_TARGET_DIR="$LOCAL/target"
DEST="$HOME/Applications/Desk (built).app"

cd app/ui
[ -d node_modules/.bin ] || { echo "installing UI packages…"; npm install --no-audit --no-fund --loglevel=error; }
echo "building Desk.app (UI + window shell)…"
npm run tauri -- build --debug --bundles app

SRC="$CARGO_TARGET_DIR/debug/bundle/macos/Desk.app"
[ -d "$SRC" ] || { echo "build finished but $SRC isn't there; look above for the Tauri error."; exit 1; }
mkdir -p "$HOME/Applications"
rm -rf "$DEST"
cp -R "$SRC" "$DEST"
touch "$DEST"
LSREG=/System/Library/Frameworks/CoreServices.framework/Versions/A/Frameworks/LaunchServices.framework/Versions/A/Support/lsregister
[ -x "$LSREG" ] && "$LSREG" -f "$DEST" || true
echo "installed: $DEST"
echo "open it from Spotlight (type Desk) or: open \"$DEST\""
