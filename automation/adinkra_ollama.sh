#!/bin/bash
# Adinkra stack: user-space ollama (Gemma 4) + per-panel query server on 8002.
#
# ollama lives in /home/paperspace/ollama (tarball install, no root, v0.32.5).
# GPU: the V100 (cc 7.0) is served by ollama's cuda_v12 runtime — the newer
# cuda_v13 build dropped cc 7.0, the log line about it is expected and benign.
# VRAM is shared with training/viewer: 12B Q4_K_M wants ~8 GiB, so expect
# CPU/GPU split (slow) while a training run holds the card.
#
# Panels (panel_agents.py): azalai = camera navigation (legacy get_plan),
# tassili = reconstruction QA, bateleur = farm state, sankofa = change/epochs.
# POST /query {"query": ..., "panel": "tassili"} on :8002.
set -e
OLLAMA=/home/paperspace/ollama
export OLLAMA_MODELS=$OLLAMA/models
export LD_LIBRARY_PATH=$OLLAMA/lib/ollama
MODEL="${ADINKRA_MODEL:-hf.co/unsloth/gemma-4-12b-it-GGUF:Q4_K_M}"

if ! curl -sf http://localhost:11434/api/tags > /dev/null; then
  echo "[adinkra] starting ollama serve"
  setsid nohup $OLLAMA/bin/ollama serve > /home/paperspace/logs/ollama.log 2>&1 &
  sleep 5
fi
$OLLAMA/bin/ollama list | grep -q gemma-4 || $OLLAMA/bin/ollama pull "$MODEL"

# Agents + server live in the UJAMAA repo (product); aru web_viewer was
# only the template. Context files are env-driven; default to citrus 04.
D=/home/paperspace/data/citrus_all/04_13D_Jackal
echo "[adinkra] query server :8002, model $MODEL"
cd /home/paperspace/code/ujamaa/adinkra
HIGH_LLM_URL=http://localhost:11434 \
HIGH_LLM_MODEL_ALIAS="$MODEL" \
ADINKRA_HIERARCHY="${ADINKRA_HIERARCHY:-$D/scene_graph_v4/marker_hierarchy_fruit5.json}" \
ADINKRA_SCORES="${ADINKRA_SCORES:-}" \
ADINKRA_LEDGER="${ADINKRA_LEDGER:-/home/paperspace/data/citrus_all/sankofa_substrate/ledger_v1.json}" \
python3 -m uvicorn server:app --host 0.0.0.0 --port 8002
