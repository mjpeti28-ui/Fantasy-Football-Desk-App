#!/bin/bash
# Double-click in Finder: moves this repo out of iCloud to ~/Developer (see app/relocate.sh),
# installs ~/Applications/Desk.app, and starts Desk from the new place.
cd "$(dirname "$0")/.." || exit 1
bash app/relocate.sh; rc=$?
if [ "$rc" -eq 1 ]; then echo; echo "Nothing was moved. Press return to close."; read -r _; exit 1; fi
if [ "$rc" -eq 0 ]; then NEW="$(cat "${DESK_LOCAL:-$HOME/.local/share/desk}/relocated-to")"; else NEW="$(pwd -P)"; fi
cd "$NEW" && bash app/install-launcher.sh
echo; echo "Starting Desk from $NEW (first start rebuilds the window shell, about a minute)…"
open "$HOME/Applications/Desk.app"
sleep 2
