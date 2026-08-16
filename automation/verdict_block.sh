#!/bin/bash
# Machine-readable containment verdict for one two-stage block.
#   usage: verdict_block.sh <block_dir> <supervision_dir>
# Runs containment_eval on the block's top-tree supervision frame against the
# stage2_censusinit_fw2 config, then MERGES the parsed verdict into the
# config-level verdicts_censusinit_fw2.json (the prod-checklist record).
# Env (defaults = 05/klap-agnostic canon): CENSUS_EMBEDDER, CENSUS_HIERARCHY.
set -uo pipefail
BD=$(readlink -f "$1"); SUP=$(readlink -f "$2"); N=$(basename "$BD")
CFGDIR=$(dirname "$BD")
# PROVENANCE FIRST: score with exactly what stage2 built the features from
# (env/defaults are a fallback — a hand-kept map scored apr with the wrong
# survey's embedder and 04 with a retired vocabulary, yielding IoU 0.00)
PROV=$BD/splat_runs_FEATFIX/stage2_provenance.json
if [ -f "$PROV" ]; then
    EMB=$(python3 -c "import json;print(json.load(open('$PROV'))['embedder'])" 2>/dev/null)
    HJ=$(python3 -c "import json;print(json.load(open('$PROV'))['hierarchy'])" 2>/dev/null)
fi
EMB=${EMB:-${CENSUS_EMBEDDER:-/home/paperspace/data/high/nerf/04_13D_v3vocab1k/ckpts/model_best.pth}}
HJ=${HJ:-${CENSUS_HIERARCHY:-/home/paperspace/data/citrus_all/04_13D_Jackal/scene_graph_v4/marker_hierarchy_fruit5.json}}
ARU=/home/paperspace/code/aru_sil_core/src/scripts
FIG=/home/paperspace/code/lab_notebook/figs
OUT_JSON=$CFGDIR/verdicts_censusinit_fw2.json
TMPLOG=$(mktemp /tmp/claude-1000/-home-paperspace-code/c47d8606-c5fe-4343-ae48-8faa25cdc994/scratchpad/verdict_${N}_XXXX.log 2>/dev/null || mktemp)
cd /home/paperspace/code/nerf_new

CFG=$(ls -t "$BD"/splat_runs_FEATFIX/stage2_censusinit_fw2/high/*/config.yml 2>/dev/null | head -1)
[ -n "$CFG" ] || { echo "VERDICT-SKIP $N: no stage2 config"; exit 0; }

# top-coverage supervision frame — TREE MODE: most painted (non-void) px.
# The censusinit step-5 criterion (ids >= 200) is fruit-only and returns
# EMPTY on fruitless surveys (05, klap) — the known fruit-centric-battery
# gap. Any painted id counts here.
FR=$(pixi run python - "$SUP" << 'PY'
import sys
import numpy as np
from PIL import Image
from pathlib import Path
best = (0, None)
for f in sorted(Path(sys.argv[1]).glob('kf_*.png')):
    a = np.array(Image.open(f), np.uint16)
    n = int((a != 65535).sum())
    if n > best[0]:
        best = (n, f.name)
print(best[1] or '')
PY
)
[ -n "$FR" ] || { echo "VERDICT-SKIP $N: no supervision frames"; exit 0; }

ROOT_DIR=$(dirname "$(dirname "$CFGDIR")")
# kf_images sits at the survey root (shimmed into prod/tassili). stderr goes
# INTO the tmplog — a swallowed traceback made VERDICT-EMPTY undiagnosable.
HIGH_EMBEDDER_CKPT=$EMB pixi run python "$ARU/containment_eval.py" \
  --config "$CFG" --hyper-ckpt "$EMB" --hierarchy-json "$HJ" \
  --supervision-dir "$SUP" --frame "$FR" \
  --kf-images "$ROOT_DIR/kf_images" \
  --out "$FIG/week_${N}_containment.png" 2>> "$TMPLOG" \
  | grep -aE "TREE|ROW|FRUIT|SAVED" | tee -a "$TMPLOG"

# One retry after a drain pause: a just-exited training forest can still
# hold VRAM for a beat, and an OOM here surfaces as an NVML INTERNAL ASSERT
# (broken NVML masks the real message — 2026-08-15 diagnosis, V100-32G).
if ! grep -qaE "TREE" "$TMPLOG"; then
  echo "VERDICT-RETRY $N in 90s (first pass empty — likely VRAM not drained)"
  sleep 90
  HIGH_EMBEDDER_CKPT=$EMB pixi run python "$ARU/containment_eval.py" \
    --config "$CFG" --hyper-ckpt "$EMB" --hierarchy-json "$HJ" \
    --supervision-dir "$SUP" --frame "$FR" \
    --kf-images "$ROOT_DIR/kf_images" \
    --out "$FIG/week_${N}_containment.png" 2>> "$TMPLOG" \
    | grep -aE "TREE|ROW|FRUIT|SAVED" | tee -a "$TMPLOG"
fi

if grep -qaE "TREE" "$TMPLOG"; then
  python3 /home/paperspace/code/automation/distill_containment_verdicts.py \
    --log "$TMPLOG" --out "$OUT_JSON" --merge \
    --block-id "${N#block_}" --frame "$FR" \
    && echo "VERDICT-DONE $N -> $OUT_JSON" \
    || echo "VERDICT-PARSE-FAIL $N (TREE lines present but distiller matched none — $TMPLOG)"
else
  echo "VERDICT-EMPTY $N (no TREE lines after retry — check $TMPLOG)"
fi
