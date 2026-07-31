#!/usr/bin/env bash
# Guarded quarantine move — the ONLY sanctioned way to reclaim disk during the
# autonomous week. Moves (never deletes) into _quarantine, preserving the full
# original path so restores are unambiguous. Usage: quarantine.sh <path> [...]
set -e
Q=/home/paperspace/data/_quarantine
for p in "$@"; do
  [ -e "$p" ] || { echo "skip (missing): $p"; continue; }
  dest="$Q/$(dirname "$p")"
  mkdir -p "$dest"
  mv -n "$p" "$dest/"
  echo "quarantined: $p -> $dest/"
done
df -h / | tail -1
