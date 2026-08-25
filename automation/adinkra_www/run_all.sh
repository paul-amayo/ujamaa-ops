#!/bin/bash
# Adinkra x WWW eval driver (plans/adinkra_www_eval.md). A100 only.
# Runs detached-safe: launch under setsid nohup; every step appends to $LOG.
# Sequence: freeze record -> citrus (pass A, B x3, C) -> restart server with
# klapmuts grounding -> klap (pass A, B x3, C) -> restore citrus server.
set -u
DIR=/home/paperspace/code/automation/adinkra_www
OUT=${ADINKRA_OUT:-/home/paperspace/data/experimental_adinkra_www_20260827}
LOG=$OUT/driver.log
MODEL="hf.co/unsloth/gemma-4-12b-it-GGUF:Q4_K_M"
ADK=/home/paperspace/code/ujamaa/adinkra

CITRUS_H=/home/paperspace/data/citrus_all/04_13D_Jackal/prod/bateleur/scene_graph/marker_hierarchy.json
CITRUS_S=/home/paperspace/data/citrus_all/04_13D_Jackal/prod/tassili/blocks_ns/lio_row100/verdicts_censusinit_fw2.json
LEDGER=/home/paperspace/data/citrus_all/sankofa_substrate/ledger_v2.json
KLAP_H=/home/paperspace/data/klapmuts/apr_2026_zed/experimental/gen2_m0/apr_mv10_band06/marker_hierarchy.json
KLAP_S=/home/paperspace/data/klapmuts/apr_2026_zed/prod/tassili/blocks_ns/lio_row100/verdicts_censusinit_fw2.json

mkdir -p "$OUT"
log() { echo "$(date -Is) $*" >> "$LOG"; }

start_server() { # $1 hierarchy $2 scores $3 label
  pkill -f "uvicorn server:app.*8003" 2>/dev/null; sleep 2
  cd "$ADK"
  setsid nohup env HIGH_LLM_URL=http://localhost:11434 \
    HIGH_LLM_MODEL_ALIAS="$MODEL" \
    ADINKRA_HIERARCHY="$1" ADINKRA_SCORES="$2" ADINKRA_LEDGER="$LEDGER" \
    python3 -m uvicorn server:app --host 0.0.0.0 --port 8003 \
    > "$OUT/server_$3.log" 2>&1 &
  for i in $(seq 1 20); do
    curl -sf -m 3 http://localhost:8003/openapi.json > /dev/null && break
    sleep 2
  done
  curl -sf -m 3 http://localhost:8003/openapi.json > /dev/null \
    || { log "FATAL server $3 did not come up"; exit 1; }
  log "server up: $3 hierarchy=$1"
}

# ---- freeze record ----
{
  echo "{"
  echo "  \"frozen\": \"$(date -Is)\","
  echo "  \"model\": \"$MODEL\","
  echo "  \"model_digest\": \"$(ollama list | awk '/gemma-4/{print $2}')\","
  echo "  \"ollama\": \"$(curl -sf -m 3 http://localhost:11434/api/version | tr -d '{}\"')\","
  echo "  \"box\": \"$(hostname)\", \"gpu\": \"$(nvidia-smi --query-gpu=name --format=csv,noheader)\","
  echo "  \"repo\": \"$(git -C /home/paperspace/code rev-parse --short HEAD)\","
  echo "  \"questions_sha\": \"$(shasum -a 256 $DIR/questions.json | cut -c1-16)\","
  echo "  \"citrus_hierarchy_sha\": \"$(shasum -a 256 $CITRUS_H | cut -c1-16)\","
  echo "  \"klap_hierarchy_sha\": \"$(shasum -a 256 $KLAP_H | cut -c1-16)\","
  echo "  \"ledger_sha\": \"$(shasum -a 256 $LEDGER | cut -c1-16)\""
  echo "}"
} > "$OUT/freeze.json"
log "freeze written"

# ---- citrus ----
start_server "$CITRUS_H" "$CITRUS_S" citrus
log "PASS A citrus start"
python3 "$DIR/runner.py" --questions "$DIR/questions.json" --site citrus \
  --tag passA --out "$OUT/passA_citrus.jsonl" >> "$LOG" 2>&1
for r in 1 2 3; do
  log "PASS B citrus r$r"
  python3 "$DIR/runner.py" --questions "$DIR/questions.json" --site citrus \
    --tag "passB_r$r" --subset B2,T1,S2,A1 \
    --out "$OUT/passB_citrus.jsonl" >> "$LOG" 2>&1
done
log "PASS C citrus"
python3 "$DIR/runner.py" --questions "$DIR/questions.json" --site citrus \
  --tag passC --followups --out "$OUT/passC_citrus.jsonl" >> "$LOG" 2>&1

# ---- klapmuts ----
start_server "$KLAP_H" "$KLAP_S" klap
log "PASS A klap start"
python3 "$DIR/runner.py" --questions "$DIR/questions.json" --site klap \
  --tag passA --out "$OUT/passA_klap.jsonl" >> "$LOG" 2>&1
for r in 1 2 3; do
  log "PASS B klap r$r"
  python3 "$DIR/runner.py" --questions "$DIR/questions.json" --site klap \
    --tag "passB_r$r" --subset B2,T1 \
    --out "$OUT/passB_klap.jsonl" >> "$LOG" 2>&1
done
log "PASS C klap"
python3 "$DIR/runner.py" --questions "$DIR/questions.json" --site klap \
  --tag passC --followups --out "$OUT/passC_klap.jsonl" >> "$LOG" 2>&1

# ---- leave the server grounded on citrus, as found ----
start_server "$CITRUS_H" "$CITRUS_S" citrus
log "ALL PASSES DONE"
