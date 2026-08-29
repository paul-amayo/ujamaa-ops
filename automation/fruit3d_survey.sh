#!/bin/bash
# 10 fps 3D fruit dedup for every fruit-bearing tree of ONE survey.
#   usage: fruit3d_survey.sh <survey_root> [logdir]
# prep (CPU, serial) -> one SAM3 process over all workdirs -> cluster (CPU).
set -u
R=$(readlink -f "$1"); LOGD=${2:-/home/paperspace/logs/fruit3d_$(basename "$R")}
mkdir -p "$LOGD"
CODE=/home/paperspace/code
PY=/home/paperspace/miniconda3/bin/python3   # cv2+scipy+mpl (validated 08-28)
note() { echo "[$(date +%F_%T)] $*" | tee -a "$LOGD/STATUS"; }

TIDS=$($PY -c "
import json
h = json.loads(open('$R/prod/bateleur/scene_graph/marker_hierarchy.json').read())
print(' '.join(str(f['tree_id']) for f in h.get('fruits', [])))")
note "START $(basename "$R"): $(echo $TIDS | wc -w) fruit-bearing trees"

WORKDIRS=()
for TID in $TIDS; do
  W="$R/prod/scratch_sam3/fruit3d_t${TID}"
  if nice -n 15 $PY "$CODE/automation/fruit3d_prep.py" \
      --root "$R" --tree "$TID" --out "$W" > "$LOGD/prep_t${TID}.log" 2>&1; then
    WORKDIRS+=("$W")
  else
    note "prep t$TID FAILED (see $LOGD/prep_t${TID}.log)"
  fi
done
note "prep done: ${#WORKDIRS[@]} workdirs"
[ "${#WORKDIRS[@]}" -gt 0 ] || { note "NOTHING TO DETECT"; exit 0; }

(cd "$CODE/aru_sil_core" && pixi run --manifest-path "$CODE/sam3/pixi.toml" \
  python "$CODE/automation/fruit3d_detect.py" "${WORKDIRS[@]}") \
  > "$LOGD/detect.log" 2>&1 || { note "DETECT FAILED"; exit 1; }
note "detect done"

for W in "${WORKDIRS[@]}"; do
  nice -n 15 $PY "$CODE/automation/fruit3d_cluster.py" "$W" \
    > "$LOGD/cluster_$(basename "$W").log" 2>&1 \
    || note "cluster $(basename "$W") FAILED"
done
note "SURVEY-DEDUP-DONE $(basename "$R")"
