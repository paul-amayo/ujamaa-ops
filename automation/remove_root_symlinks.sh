#!/bin/bash
# Remove EVERY symlink under the surveys so prod assets can only be addressed
# where they physically live.
#
#   usage: remove_root_symlinks.sh <survey-dir> [--apply]
#          remove_root_symlinks.sh --all [--apply]
#
# Dry-run by default. Nothing is removed without --apply.
#
# WHY (Paul, 2026-08-18): "the survey root is where poisoned or stale
# artifacts have been saved, i want all prod assets to physically live in
# prod". The shims let stale references resolve silently; removing them turns
# silent staleness into a loud failure that names its call site.
#
# THE PIPELINE WILL BREAK. Intended: each break is a script addressing a
# survey by a shimmed path instead of by prod/.
#
# Safety:
#   - unlinking a symlink NEVER deletes its target, so no data is lost —
#     only reachability changes;
#   - only symlinks are touched, never a real file or directory;
#   - automation/root_symlink_audit.py must have written the restore
#     manifest first; automation/restore_root_symlinks.py undoes all of it.
set -uo pipefail

MANIFEST=/home/paperspace/logs/root_symlink_manifest.json
APPLY=0
TARGETS=()
for a in "$@"; do
    case "$a" in
        --apply) APPLY=1 ;;
        --all)   while IFS= read -r s; do TARGETS+=("$s"); done \
                     < <(python3 -c "import json;print('\n'.join(json.load(open('$MANIFEST'))['surveys']))") ;;
        *)       TARGETS+=("$a") ;;
    esac
done
[ ${#TARGETS[@]} -eq 0 ] && { echo "usage: $0 <survey-dir>|--all [--apply]"; exit 2; }
[ -f "$MANIFEST" ] || { echo "NO RESTORE MANIFEST at $MANIFEST — run automation/root_symlink_audit.py first"; exit 2; }

# Manifest must be current, or a link removed now could not be restored.
STALE=0
for S in "${TARGETS[@]}"; do
    S=${S%/}
    ON_DISK=$(find "$S" -type l 2>/dev/null | wc -l)
    IN_MAN=$(python3 -c "
import json,sys
m=json.load(open('$MANIFEST'))['surveys']
print(len(m.get('$S', [])))" 2>/dev/null || echo 0)
    if [ "$ON_DISK" != "$IN_MAN" ]; then
        echo "MANIFEST STALE for $(basename "$S"): $ON_DISK links on disk, $IN_MAN recorded"
        STALE=1
    fi
done
[ "$STALE" = "1" ] && { echo "re-run automation/root_symlink_audit.py before applying"; exit 2; }

TOT=0
for S in "${TARGETS[@]}"; do
    S=${S%/}
    [ -d "$S" ] || { echo "skip $S (not a directory)"; continue; }
    N=$(find "$S" -type l 2>/dev/null | wc -l)
    ROOT=$(find "$S" -maxdepth 1 -type l 2>/dev/null | wc -l)
    if [ "$APPLY" = "1" ]; then
        while IFS= read -r -d '' p; do
            unlink "$p" && TOT=$((TOT+1))
        done < <(find "$S" -type l -print0 2>/dev/null)
        LEFT=$(find "$S" -type l 2>/dev/null | wc -l)
        echo "  $(basename "$S"): removed $N symlinks ($ROOT at root); $LEFT remain"
    else
        echo "  $(basename "$S"): would remove $N symlinks ($ROOT at root)"
        TOT=$((TOT+N))
    fi
done

echo
if [ "$APPLY" = "1" ]; then
    echo "removed $TOT symlinks in total"
    echo "restore: python3 /home/paperspace/code/automation/restore_root_symlinks.py"
else
    echo "DRY RUN — $TOT symlinks would be removed. Re-run with --apply."
fi
