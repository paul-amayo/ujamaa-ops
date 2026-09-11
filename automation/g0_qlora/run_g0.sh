#!/bin/bash
# G0 driver (plans/gemma4_finetune_memo.md): prep -> train with auto-resume.
# Detached-safe; every step appends to $LOG. The train loop relaunches with
# --resume after crashes/preemption (fleet grabbing the card aborts loudly
# via the VRAM guard; the loop waits and retries).
set -u
# gemma vocab (262k) makes the CE logits the VRAM hog: micro_bs=1 +
# expandable segments, or the loss alone allocates 8 GiB (smoke, 2026-09-04).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${G0_PY:-~/envs/hfeval_ft/bin/python}   # hfeval_ft: torch>=2.6, required for checkpoint resume
# (transformers' CVE-2025-32434 torch.load guard). Plain hfeval (torch 2.5.1) as the
# default cost 139 failed retries / 23.6 h on 2026-09-09 when an eval pause relaunched
# without G0_PY. Override with G0_PY=... only for a non-resume cold start.
D=/home/paperspace/code/automation/g0_qlora
TOK=/home/paperspace/data/g0_tokens
RUN=/home/paperspace/data/g0_run
LOG=~/logs/g0_qlora.log
log() { echo "$(date -Is) $*" >> "$LOG"; }

mkdir -p "$TOK" "$RUN"
PHASE=${1:-all}

if [ "$PHASE" = "prep" ] || [ "$PHASE" = "all" ]; then
  log "PREP start"
  $PY "$D/prep_data.py" --out "$TOK" >> "$LOG" 2>&1 \
    || { log "FATAL prep"; exit 1; }
  log "PREP done"
fi

if [ "$PHASE" = "train" ] || [ "$PHASE" = "all" ]; then
  TRIES=0
  until [ -f "$RUN/TRAIN_DONE" ]; do
    TRIES=$((TRIES+1))
    [ $TRIES -gt 200 ] && { log "FATAL too many restarts"; exit 1; }
    RESUME=""
    ls "$RUN"/checkpoint-* >/dev/null 2>&1 && RESUME="--resume"
    log "TRAIN attempt $TRIES $RESUME"
    $PY "$D/train_qlora.py" --data "$TOK" --out "$RUN" --micro-bs 2 --accum 64 $RESUME >> "$LOG" 2>&1
    [ -f "$RUN/TRAIN_DONE" ] || { log "train exited; wait 10m then retry"; sleep 600; }
  done
  log "G0 TRAIN COMPLETE"
fi
