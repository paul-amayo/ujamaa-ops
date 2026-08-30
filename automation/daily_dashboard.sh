#!/bin/bash
# Daily dashboard refresh (cron, 06:15 UTC). Rebuilds lab_notebook/dashboard.html
# from the live sources (roadmap, lab notebook, ~/logs, git, disk, ruff+pytest
# code health) and appends one line per run to ~/logs/daily_dashboard.log.
# flock: never overlap with a manual rebuild or a previous slow run.
# lockfile is DELIBERATELY volatile (stale locks must die with the boot)
exec 9>/tmp/ujamaa_dashboard.lock
flock -n 9 || { echo "$(date -Is) SKIP (locked)" >> /home/paperspace/logs/daily_dashboard.log; exit 0; }
cd /home/paperspace/code
LOG=/home/paperspace/logs/daily_dashboard.log
OUT=$(timeout 900 python3 automation/build_dashboard.py 2>&1)
RC=$?
NB=$(ls -t lab_notebook/2026-*.md | head -1)
ENTRIES=$(grep -c '^## ' "$NB" 2>/dev/null || echo 0)
TODAY=$(grep -c "^## $(date +%Y-%m-%d)" "$NB" 2>/dev/null || echo 0)
if [ $RC -eq 0 ]; then
  echo "$(date -Is) OK  | $(basename $NB): $ENTRIES entries ($TODAY today)" >> $LOG
else
  echo "$(date -Is) FAIL rc=$RC | ${OUT: -200}" >> $LOG
fi
# keep the log from growing without bound
tail -n 400 $LOG > $LOG.tmp && mv $LOG.tmp $LOG
