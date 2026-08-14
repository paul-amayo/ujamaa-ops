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
EMB=${CENSUS_EMBEDDER:-/home/paperspace/data/high/nerf/04_13D_v3vocab1k/ckpts/model_best.pth}
HJ=${CENSUS_HIERARCHY:-/home/paperspace/data/citrus_all/04_13D_Jackal/scene_graph_v4/marker_hierarchy_fruit5.json}
ARU=/home/paperspace/code/aru_sil_core/src/scripts
FIG=/home/paperspace/code/lab_notebook/figs
OUT_JSON=$CFGDIR/verdicts_censusinit_fw2.json
TMPLOG=$(mktemp /tmp/claude-1000/-home-paperspace-code/c47d8606-c5fe-4343-ae48-8faa25cdc994/scratchpad/verdict_${N}_XXXX.log 2>/dev/null || mktemp)
cd /home/paperspace/code/nerf_new

CFG=$(ls -t "$BD"/splat_runs_FEATFIX/stage2_censusinit_fw2/high/*/config.yml 2>/dev/null | head -1)
[ -n "$CFG" ] || { echo "VERDICT-SKIP $N: no stage2 config"; exit 0; }

# top-tree supervision frame (most >=200 non-void semantic px)
FR=$(pixi run python - "$SUP" << 'PY'
import sys
import numpy as np
from PIL import Image
from pathlib import Path
best = (0, None)
for f in sorted(Path(sys.argv[1]).glob('kf_*.png')):
    a = np.array(Image.open(f), np.uint16)
    n = int(((a >= 200) & (a != 65535)).sum())
    if n > best[0]:
        best = (n, f.name)
print(best[1] or '')
PY
)
[ -n "$FR" ] || { echo "VERDICT-SKIP $N: no supervision frames"; exit 0; }

ROOT_DIR=$(dirname "$(dirname "$CFGDIR")")
# kf_images sits at the survey root (shimmed into prod/tassili)
HIGH_EMBEDDER_CKPT=$EMB pixi run python "$ARU/containment_eval.py" \
  --config "$CFG" --hyper-ckpt "$EMB" --hierarchy-json "$HJ" \
  --supervision-dir "$SUP" --frame "$FR" \
  --kf-images "$ROOT_DIR/kf_images" \
  --out "$FIG/week_${N}_containment.png" 2>/dev/null \
  | grep -aE "TREE|ROW|FRUIT|SAVED" | tee "$TMPLOG"

# block id inside the log lines is containment_eval's [NNN frame] prefix;
# fall back to labelling by dir if the parse finds nothing
if grep -qaE "TREE" "$TMPLOG"; then
  python3 /home/paperspace/code/automation/distill_containment_verdicts.py \
    --log "$TMPLOG" --out "$OUT_JSON" --merge \
    && echo "VERDICT-DONE $N -> $OUT_JSON"
else
  echo "VERDICT-EMPTY $N (no TREE lines — check $TMPLOG)"
fi
