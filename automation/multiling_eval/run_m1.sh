#!/bin/bash
# M1 anchor driver (plans/multilingual_model_track.md). A100, detached-safe.
# Outer loop = model (one ollama load each); inner = tasks. Resumable.
set -u
DIR=/home/paperspace/code/automation/multiling_eval
OUT=${MULTILING_OUT:-/home/paperspace/data/experimental_multiling_m1}
LOG=$OUT/driver.log
MODELS=(
  "hf.co/mradermacher/gemma-3-12b-pt-GGUF:Q4_K_M"
  "hf.co/mradermacher/AfriqueGemma-12B-GGUF:Q4_K_M"
  "hf.co/unsloth/gemma-4-12b-it-GGUF:Q4_K_M"
)
mkdir -p "$OUT"
log() { echo "$(date -Is) $*" >> "$LOG"; }

ensure_ollama() {
  curl -sf -m 4 http://localhost:11434/api/version >/dev/null && return 0
  log "ollama down — starting it (evening fleet policy stops it; drivers restart it)"
  sudo -n systemctl start ollama 2>/dev/null || systemctl start ollama 2>/dev/null
  for i in $(seq 1 15); do
    curl -sf -m 4 http://localhost:11434/api/version >/dev/null \
      && { log "ollama up"; return 0; }
    sleep 2
  done
  log "FATAL ollama would not start"
  return 1
}

log "M1 start"
ensure_ollama || exit 1
python3 "$DIR/fetch_data.py" >> "$LOG" 2>&1 || { log "FATAL fetch"; exit 1; }
for M in "${MODELS[@]}"; do
  TAG=$(basename "${M%%:*}" | sed s/-GGUF//)
  for TASK in belebele flores; do
    log "run $TAG $TASK"
    python3 "$DIR/anchor_run.py" --model "$M" --task "$TASK" \
      --out "$OUT/${TAG}_${TASK}.jsonl" >> "$LOG" 2>&1 \
      || log "WARN $TAG $TASK exited nonzero (resumable)"
  done
done
python3 "$DIR/score.py" "$OUT" > "$OUT/summary.txt" 2>&1
log "M1 DONE — summary.txt written"
