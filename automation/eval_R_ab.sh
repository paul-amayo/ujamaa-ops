#!/usr/bin/env bash
# A/B relevancy eval on 01_13B lio_row blocks 000-002: old C_20k arm vs new
# R_20k arm, both scored with 01_13B_v1 (production decoder). Plus a diagnostic
# third eval on block_000 old arm scored with 02_block000_v1 (its build-time
# embedder) to separate "decoder mismatch" from "bad features".
# transforms.json must be GL during rendering -> swap per block, restore after.
set -e
ROOT=/home/paperspace/data/citrus_all/01_13B_Jackal
HIER=$ROOT/scene_graph/marker_hierarchy.json
V1=/home/paperspace/data/high/nerf/01_13B_v1/ckpts/model_best.pth
V02=/home/paperspace/data/high/nerf/02_block000_v1/ckpts/model_best.pth
NS_PIXI=/home/paperspace/code/nerf_new/pixi.toml
ARU=/home/paperspace/code/aru_sil_core
OUTBASE=${1:?usage: eval_R_ab.sh <out_base_dir>}
NF=60

swap() { python3 - "$1" "$ROOT" <<'PY'
import json, numpy as np, sys
from pathlib import Path
B=Path(sys.argv[1]); ROOT=Path(sys.argv[2])
d=json.loads((B/"transforms.json").read_text())
lio={p["image_name"]:p["transform"] for p in json.loads((ROOT/"lio_image_poses_kf20cm.json").read_text())}
CV2GL=np.diag([1.,-1.,-1.,1.])
for f in d["frames"]:
    n=Path(f["file_path"]).name
    if n in lio: f["transform_matrix"]=(np.asarray(lio[n])@CV2GL).tolist()
(B/"transforms.json").write_text(json.dumps(d,indent=2))
PY
}

cd "$ARU/src"
for N in 000 001 002; do
  B=$ROOT/blocks_ns/lio_row/block_$N
  cp "$B/transforms.json" "$B/transforms.json.evalbak"
  swap "$B"
  for ARM in abC_20k R_20k; do
    CFG=$(find "$B/splat_runs_$ARM" -name config.yml | sort | tail -1)
    [ -z "$CFG" ] && { echo "SKIP $N $ARM (no config)"; continue; }
    echo "=== EVAL block_$N arm=$ARM decoder=v1 ==="
    pixi run --manifest-path "$NS_PIXI" python scripts/full_relevancy_eval.py \
      "$CFG" "$V1" "$HIER" "$B" "$OUTBASE/b${N}_${ARM}_v1" $NF
  done
  if [ "$N" = "000" ] && [ -f "$V02" ]; then
    CFG=$(find "$B/splat_runs_abC_20k" -name config.yml | sort | tail -1)
    if [ -n "$CFG" ]; then
      echo "=== EVAL block_000 arm=abC_20k decoder=v02 (diagnostic) ==="
      pixi run --manifest-path "$NS_PIXI" python scripts/full_relevancy_eval.py \
        "$CFG" "$V02" "$HIER" "$B" "$OUTBASE/b000_abC_20k_v02" $NF || echo "DIAG FAILED (non-fatal)"
    fi
  fi
  mv -f "$B/transforms.json.evalbak" "$B/transforms.json"
done
echo "AB-EVAL ALL DONE"
