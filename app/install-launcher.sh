#!/usr/bin/env bash
# Installs (or updates) ~/Applications/Desk.app, a launcher that runs app/run.sh from this repo
# with your login shell's PATH, so Desk opens from Spotlight, Launchpad or the Dock.
set -euo pipefail
cd "$(dirname "$0")/.."
here="$(pwd -P)"; app="$HOME/Applications/Desk.app"; src="app/launcher/Desk.app"
[ -d "$src" ] || exit 0
if [ -d "$app" ] && cmp -s "$src/Contents/MacOS/desk-launch" "$app/Contents/MacOS/desk-launch" \
   && cmp -s "$src/Contents/Info.plist" "$app/Contents/Info.plist" \
   && [ "$(cat "$app/Contents/Resources/repo-path" 2>/dev/null)" = "$here" ]; then
  exit 0
fi
mkdir -p "$HOME/Applications"
rm -rf "$app"
cp -R "$src" "$app"
cp app/ui/src-tauri/icons/icon.icns "$app/Contents/Resources/AppIcon.icns" 2>/dev/null || true
printf '%s\n' "$here" > "$app/Contents/Resources/repo-path"
chmod +x "$app/Contents/MacOS/desk-launch"
touch "$app"
echo "installed ~/Applications/Desk.app: open Desk from Spotlight or Launchpad (log: ~/Library/Logs/Desk.log)"
