#!/bin/bash
# Session-independent watchdog for the week queue (system crontab, not
# Claude-session cron — survives session death, reboots, everything).
#
# Policy (2026-08-20, the autonomous-week contract):
#   - queue alive  -> silent exit.
#   - queue dead after a DELIBERATE stop (CIRCUIT-BREAKER, ABORT-DISK,
#     WEEK-DONE in the last marks) -> DO NOT restart; those pauses exist
#     precisely because a human/Claude must diagnose first. Log once.
#   - queue dead with NO deliberate stop (crash, reboot, killed session's
#     collateral) -> relaunch detached. Safe by design: pointers are
#     written only at slot outcomes, so a mid-slot death resumes cleanly.
set -u
# cron runs with PATH=/usr/bin:/bin. pixi (and with it every pipeline stage)
# lives in ~/.pixi/bin, so a cron-launched relaunch would fast-fail every
# slot in ~1 s (no K-domain check), burn four pointers and trip the breaker
# — the 2026-08-20 failure class. Found 2026-08-21 (first real reboot).
export PATH=/home/paperspace/.local/bin:/home/paperspace/.pixi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Q=/home/paperspace/code/automation/week_prod_queue_20260814.sh
LOG=/home/paperspace/logs/week_prod_20260814.log
WLOG=/home/paperspace/logs/queue_watchdog.log
STAMP=/home/paperspace/logs/week_prod_20260814_state/watchdog_notified

alive=$(ps -eo args | awk -v q="$Q" '$1 == "bash" && $2 == q' | wc -l)
if [ "$alive" -gt 0 ]; then
    rm -f "$STAMP" 2>/dev/null
    exit 0
fi

recent=$(tail -40 "$LOG" 2>/dev/null)
if printf '%s' "$recent" | grep -qE "CIRCUIT-BREAKER|ABORT-DISK|WEEK-DONE"; then
    if [ ! -f "$STAMP" ]; then
        echo "$(date -Is) queue stopped DELIBERATELY (breaker/disk/done) — NOT restarting; needs diagnosis" >> "$WLOG"
        touch "$STAMP"
    fi
    exit 0
fi

echo "$(date -Is) queue dead with no deliberate stop — relaunching" >> "$WLOG"
setsid bash "$Q" >> "$LOG" 2>&1 < /dev/null &
disown
