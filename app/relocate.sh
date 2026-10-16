#!/usr/bin/env bash
# One-time move of this repo out of iCloud, called by app/run.sh (or run it by hand).
#
# The repo lived in an iCloud-synced Desktop with Optimize Mac Storage on, so macOS kept evicting
# files (they turn into placeholders that download again on the next read; a VM can't read them
# at all). This copies the repo to ~/Developer/<name>, reading every file natively so anything
# evicted downloads first, checks that every file arrived with the same size, then:
#   - renames the old folder to "<name> (old copy in iCloud)"  (delete it once you're happy)
#   - leaves a small real folder (just MOVED.txt) at the old path. NOT a link: with a link there,
#     Claude's connected-folder mount for that path failed ("failed to mount"), breaking every chat
#     attached to the old path (10/8). A real folder mounts fine and MOVED.txt says where the repo went.
# Rebuildables are not copied: old venvs/node_modules kept aside earlier, the old Rust build.
#
# Exit codes: 0 moved, 3 nothing to do, 1 failed (the repo is left exactly where it was).
# Opt out: DESK_NO_RELOCATE=1 app/run.sh     Different home: DESK_HOME=~/Code app/run.sh
set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd -P)"
NAME="$(basename "$SRC")"
DEST_PARENT="${DESK_HOME:-$HOME/Developer}"
DEST="$DEST_PARENT/$NAME"
CLOUD="$HOME/Library/Mobile Documents/com~apple~CloudDocs"

[ "${DESK_NO_RELOCATE:-}" = "1" ] && exit 3
# never "move" an old copy left behind by an earlier move (see app/run.sh): that made duplicates
case "$NAME" in *"(old copy in iCloud)"*) exit 3 ;; esac
[ -f "$SRC/MOVED.txt" ] && exit 3
case "$SRC" in
  "$HOME/Desktop/"*)   [ -d "$CLOUD/Desktop" ] || exit 3 ;;
  "$HOME/Documents/"*) [ -d "$CLOUD/Documents" ] || exit 3 ;;
  "$HOME/Library/Mobile Documents/"*) ;;
  *) exit 3 ;;
esac
RESUME=""

# what we copy: everything except rebuildables and scratch
EXCLUDES=(
  "/.venv" "/.venv.icloud-old*" "/app/ui/node_modules.*" "/app/ui/src-tauri/target"
  "/_to_delete" "__pycache__" ".pytest_cache" ".DS_Store"
)
# the same things, skipped when counting and checking (find syntax)
SKIP=( '(' -path ./.venv -o -path './.venv.icloud-old*' -o -path './app/ui/node_modules.*' -o -path ./app/ui/src-tauri/target
       -o -path ./_to_delete -o -name __pycache__ -o -name .pytest_cache -o -name .DS_Store ')' -prune -o )
if stat -f '%z' / >/dev/null 2>&1; then   # macOS
  listing() { ( cd "$1" && find . "${SKIP[@]}" '(' -type f -o -type l ')' -print0 | xargs -0 stat -f '%z %N' | LC_ALL=C sort ); }
else                                       # GNU (only when testing this script on Linux)
  listing() { ( cd "$1" && find . "${SKIP[@]}" '(' -type f -o -type l ')' -printf '%s %p\n' | LC_ALL=C sort ); }
fi

# A folder already at the destination is only reused if it's an interrupted copy of this repo:
# it carries our marker, or every file in it also exists here (rsync then just fills in the rest).
if [ -e "$DEST" ] || [ -L "$DEST" ]; then
  if [ -f "$DEST/.desk-relocating" ] || { [ -d "$DEST" ] && [ -z "$(comm -23 <(cd "$DEST" && find . "${SKIP[@]}" -type f -print | LC_ALL=C sort) <(cd "$SRC" && find . "${SKIP[@]}" -type f -print | LC_ALL=C sort) | grep -v '^./.desk-relocating$' | head -1)" ]; }; then
    RESUME=1
  else
    echo "relocate: $DEST already exists and isn't a copy of this repo, so the repo stays where it is (set DESK_HOME to pick another place)."
    exit 3
  fi
fi

echo "relocate: moving $NAME out of iCloud to $DEST${RESUME:+ (picking up the earlier, interrupted copy)}"
echo "  counting files (this reads the folder tree, not the files)…"
TOTAL_FILES=$(listing "$SRC" | wc -l | tr -d ' ')
TOTAL_KB=$(listing "$SRC" | awk '{s += $1} END {printf "%d", s / 1024}')
FREE_KB=$(df -Pk "$HOME" | awk 'NR == 2 {print $4}')
echo "  $TOTAL_FILES files, $((TOTAL_KB / 1024)) MB; $((FREE_KB / 1024)) MB free on disk"
if [ "$FREE_KB" -lt $((TOTAL_KB + 1048576)) ]; then
  echo "relocate: not enough free space (need the copy plus 1 GB to spare). Leaving the repo where it is."
  exit 1
fi

# stop anything of Desk's that could write into the repo while it's copied
pkill -f "uvicorn app.engine.server:app" 2>/dev/null || true
pkill -f "target/debug/desk" 2>/dev/null || true
for pid in $(lsof -ti tcp:5173 2>/dev/null); do ps -o command= -p "$pid" | grep -q vite && kill "$pid" 2>/dev/null || true; done

# an interrupted earlier run can leave its rsync going in the background
pkill -f "rsync.*${DEST}/" 2>/dev/null || true

# Download everything iCloud evicted first, 32 files at a time: reading a placeholder makes macOS
# fetch it, and one at a time that's about a file a second.
PROG="$(mktemp "${TMPDIR:-/tmp}/desk-relocate.XXXXXX")"
echo "  downloading anything iCloud evicted, 32 files at a time…"
( cd "$SRC" && find . "${SKIP[@]}" -type f -print0 | xargs -0 -n 40 -P 32 sh -c 'cat "$@" >/dev/null 2>&1; echo $# >> "$0"' "$PROG" ) & PF=$!
while kill -0 "$PF" 2>/dev/null; do
  sleep 5
  kill -0 "$PF" 2>/dev/null && echo "    $(awk '{s += $1} END {print s + 0}' "$PROG") of $TOTAL_FILES files read"
done
wait "$PF" || true
rm -f "$PROG"

mkdir -p "$DEST"
touch "$DEST/.desk-relocating"
echo "  copying…"
copy() {
  if command -v rsync >/dev/null 2>&1; then
    local args=(-a)
    for e in "${EXCLUDES[@]}"; do args+=("--exclude=$e"); done
    rsync "${args[@]}" --exclude=/.desk-relocating "$SRC/" "$DEST/"
  else
    mkdir -p "$DEST"
    local args=()
    for e in "${EXCLUDES[@]}"; do args+=("--exclude=.${e#/}" "--exclude=*/${e#/}"); done
    ( cd "$SRC" && tar "${args[@]}" -cf - . ) | ( cd "$DEST" && tar -xpf - )
  fi
}
copy & CP=$!
while kill -0 "$CP" 2>/dev/null; do
  sleep 5
  kill -0 "$CP" 2>/dev/null && echo "    $(find "$DEST" \( -type f -o -type l \) 2>/dev/null | wc -l | tr -d ' ') of $TOTAL_FILES files"
done
if ! wait "$CP"; then
  echo "relocate: the copy stopped partway. The repo is unchanged where it was; the next run of app/run.sh picks the copy up where it left off."
  exit 1
fi

# in case this rsync ignored an anchored exclude, drop what shouldn't have come along
rm -rf "$DEST/.venv" "$DEST"/.venv.icloud-old* "$DEST"/app/ui/node_modules.* "$DEST/app/ui/src-tauri/target" "$DEST/_to_delete" 2>/dev/null || true
rm -f "$DEST/.desk-relocating"
echo "  checking every file arrived with the same size…"
if ! diff <(listing "$SRC") <(listing "$DEST") >/tmp/desk-relocate-diff.txt; then
  echo "relocate: the copy doesn't match the original (first differences below; full list in /tmp/desk-relocate-diff.txt)."
  head -10 /tmp/desk-relocate-diff.txt
  echo "  The repo is unchanged where it was; the next run of app/run.sh copies again and fills in the differences."
  touch "$DEST/.desk-relocating"
  exit 1
fi
if [ -d "$DEST/.git" ] && ! git -C "$DEST" rev-parse --verify HEAD >/dev/null 2>&1; then
  echo "relocate: git can't read the copied repository; leaving everything where it was (the next run tries again)."
  touch "$DEST/.desk-relocating"
  exit 1
fi
# the Python env link (its target lives outside the repo, so it survives the move as is)
if [ -L "$SRC/.venv" ]; then ln -sfn "$(readlink "$SRC/.venv")" "$DEST/.venv"; fi

OLD="$SRC (old copy in iCloud)"
[ -e "$OLD" ] && OLD="$OLD $(date +%Y%m%d%H%M%S)"
mv "$SRC" "$OLD"
cat > "$OLD/MOVED.txt" <<EOF
This folder is the old copy of $NAME, kept in case anything went missing.
The repo now lives at $DEST (moved $(date) by app/relocate.sh, because iCloud kept evicting
its files). "$SRC" is now a small marker folder (MOVED.txt only), not a link.
Once Desk and your other tools work from the new place, delete this folder to free iCloud space.
EOF
# A real folder, not a link: a link at a path Claude has connected makes its mount fail.
mkdir "$SRC"
cat > "$SRC/MOVED.txt" <<EOF
$NAME has moved.

The repo now lives at $DEST. app/relocate.sh moved it there on $(date +%Y-%m-%d) because
iCloud kept evicting its files.

This folder is only a marker. It is a real, empty folder (not a link) on purpose: a link at this
path stops Claude's connected-folder mount from working. If you are a chat or tool that was
connected to this path, connect $DEST instead.

The original iCloud copy is next to this folder as "$(basename "$OLD")".
Delete it once you are happy with the new place.
EOF
mkdir -p "${DESK_LOCAL:-$HOME/.local/share/desk}" && printf '%s\n' "$DEST" > "${DESK_LOCAL:-$HOME/.local/share/desk}/relocated-to"
echo "relocate: done. The repo is at $DEST"
echo "  The old path is now a marker folder (MOVED.txt); the old copy is \"$(basename "$OLD")\" next to it (delete it when you're happy)."
echo "  If Claude chats were connected to the old path, connect $DEST in them instead."
exit 0
