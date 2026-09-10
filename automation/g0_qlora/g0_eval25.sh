#!/bin/bash
# G0 25% eval orchestrator: wait for checkpoint-1200, pause training, run the
# paired eval, resume training. Detached-safe; every step logged.
set -u
RUN=/home/paperspace/data/g0_run
LOG=~/logs/g0_qlora.log
log() { echo "$(date -Is) [eval25] $*" >> "$LOG"; }

log "waiting for checkpoint-1200"
until [ -d "$RUN/checkpoint-1200" ]; do sleep 120; done
# let the checkpoint finish writing
sleep 60

log "pausing training (driver + trainer)"
for p in $(pgrep -f "run_g0.sh"); do kill "$p" 2>/dev/null; done
for p in $(pgrep -f "train_qlora.py"); do kill "$p" 2>/dev/null; done
sleep 20
pgrep -f "train_qlora.py" >/dev/null && { for p in $(pgrep -f train_qlora.py); do kill -9 "$p"; done; sleep 5; }

log "running paired eval on checkpoint-1200"
~/envs/hfeval/bin/python /home/paperspace/code/automation/g0_qlora/eval_ckpt.py \
  --ckpt "$RUN/checkpoint-1200" --out "$RUN/eval25.json" >> "$LOG" 2>&1 \
  || log "EVAL FAILED — resuming training regardless"

log "resuming training"
setsid nohup env G0_PY=/home/paperspace/envs/hfeval_ft/bin/python \
  bash /home/paperspace/code/automation/g0_qlora/run_g0.sh train > /dev/null 2>&1 &
sleep 15
pgrep -f "train_qlora.py" >/dev/null && log "G0_EVAL25_DONE — training resumed" \
  || log "G0_EVAL25_DONE — WARNING trainer not yet up (driver will retry)"
