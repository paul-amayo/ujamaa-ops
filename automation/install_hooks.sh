#!/bin/bash
# Install the version-controlled hooks into .git/hooks (which git does not
# track). Re-run after a fresh clone. Idempotent; backs up anything it replaces.
set -e
cd "$(dirname "$0")/.."
SRC=automation/hooks; DST=.git/hooks
for h in "$SRC"/*; do
  n=$(basename "$h")
  [ -e "$DST/$n" ] && [ ! -L "$DST/$n" ] && mv "$DST/$n" "$DST/$n.bak.$(date +%s)"
  ln -sf "$(pwd)/$h" "$DST/$n"
  chmod +x "$h"
  echo "installed $n -> $DST/$n"
done
