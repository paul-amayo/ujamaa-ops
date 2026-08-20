#!/bin/bash
# Safe process-group kill for the week queue (or any script by path pattern).
# Exists because `pkill -f` and hand-rolled `kill -TERM -$(ps|grep|awk)` have
# matched the CALLING shell's own eval string twice (2026-08-20: killed my own
# PGID at 18:10 while the real queue survived and kept burning slots).
# Usage: kill_queue.sh [pattern]   (default: week_prod_queue_20260814.sh)
set -u
PAT=${1:-week_prod_queue_20260814.sh}
ME=$$
MYPG=$(ps -o pgid= -p $ME | tr -dc 0-9)
# Real instances: bash running the script by path — NOT shells whose args
# merely CONTAIN the pattern inside an eval/snapshot wrapper.
MAPFILE=$(ps -eo pgid,pid,args | awk -v pat="$PAT" -v mypg="$MYPG" \
    '$3 == "bash" && index($0, pat) && $1 != mypg && $0 !~ /snapshot-bash|eval / {print $1}' | sort -u)
if [ -z "$MAPFILE" ]; then
    echo "no live instance of $PAT found (own PGID $MYPG excluded)"
    exit 1
fi
for PG in $MAPFILE; do
    echo "killing PGID $PG"
    kill -TERM -"$PG"
done
sleep 2
LEFT=$(ps -eo pgid,args | awk -v pat="$PAT" 'index($0, pat) && $2 == "bash" {print $1}' | sort -u | wc -l)
echo "instances remaining: $LEFT"
[ "$LEFT" -eq 0 ]
