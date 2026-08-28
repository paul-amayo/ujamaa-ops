#!/bin/bash
# Weekend catch-up queue — 2026-08-28 (Paul: "queue non GPU roadmap items as
# well so we catch up over the weekend on long tasks").
#
# Items (sequential, niced, per-item logs, disk floor 40G, failures isolated):
#   1 datasets      DATASETS.md regen (post-gen2 milestone hygiene)
#   2 harness_a     regression harness Tier A (CPU invariants)
#   3 topdown_*     Bateleur top-down export refresh, every survey (ledger P4b)
#   4 fruit3d_*     10 fps 3D fruit dedup fleet: every fruit-bearing tree on
#                   04+05 — prep (CPU) -> one SAM3 pass (GPU, model loaded
#                   once) -> cluster (CPU). Validated on 05 tree 72 today.
#   5 xepoch_v3     cross-epoch join on corroborated 3D counts (the 158-vs-2
#                   comparability fix v1/v2 could not deliver)
#
# NOT queued: tree_struct_metrics (P2) — smoke run showed census argmax
# ownership smears along ray corridors (t72/b013: owned cloud spans 10x23 m,
# 0 gaussians within 3 m of anchor); needs a diagnosis session, not a queue.
set -u
export PATH=/home/paperspace/.local/bin:/home/paperspace/.pixi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
CODE=/home/paperspace/code
DATA=/home/paperspace/data/citrus_all
LOGD=/home/paperspace/logs/weekend_cpu_20260828
# fruit3d/v3 validated end-to-end under this interpreter (cv2+scipy+mpl);
# the bare /usr/bin/python3 has none of them — pin explicitly, no PATH roulette
PY=/home/paperspace/miniconda3/bin/python3
mkdir -p "$LOGD"
ST="$LOGD/STATUS"
note() { echo "[$(date +%F_%T)] $*" | tee -a "$ST"; }

run_item() { # run_item <name> <cmd...>
  local name=$1; shift
  local free
  free=$(df --output=avail -BG /home/paperspace | tail -1 | tr -dc 0-9)
  if [ "$free" -lt 40 ]; then note "SKIP $name (disk ${free}G < 40G)"; return 1; fi
  note "START $name"
  if nice -n 15 "$@" > "$LOGD/$name.log" 2>&1; then
    note "DONE  $name"
  else
    note "FAILED $name (see $LOGD/$name.log)"
    return 1
  fi
}

note "WEEKEND QUEUE START (pid $$)"

run_item datasets python3 "$CODE/automation/build_dataset_registry.py"
run_item harness_a bash "$CODE/automation/regression_harness.sh" --cpu-only

for R in "$DATA"/0*_Jackal; do
  s=$(basename "$R" | cut -c1-2)
  run_item "topdown_$s" python3 "$CODE/ujamaa/project/export_bateleur_topdown.py" "$R"
done

# ---- 10 fps 3D fruit dedup fleet (04 + 05) ----
WORKDIRS=()
for s in 04 05; do
  R="$DATA/${s}_13D_Jackal"
  for TID in $(python3 -c "
import json
h = json.loads(open('$R/prod/bateleur/scene_graph/marker_hierarchy.json').read())
print(' '.join(str(f['tree_id']) for f in h.get('fruits', [])))"); do
    W="$R/prod/scratch_sam3/fruit3d_t${TID}"
    run_item "prep_${s}_t${TID}" "$PY" "$CODE/automation/fruit3d_prep.py" \
      --root "$R" --tree "$TID" --out "$W" \
      && WORKDIRS+=("$W")
  done
done
note "PREP PHASE DONE: ${#WORKDIRS[@]} workdirs"

if [ "${#WORKDIRS[@]}" -gt 0 ]; then
  run_item detect_fleet bash -c "cd $CODE/aru_sil_core && pixi run \
    --manifest-path $CODE/sam3/pixi.toml python \
    $CODE/automation/fruit3d_detect.py ${WORKDIRS[*]}"
  for W in "${WORKDIRS[@]}"; do
    rel=${W#"$DATA"/}                      # 04_13D_Jackal/prod/.../fruit3d_tN
    run_item "cluster_${rel:0:2}_$(basename "$W")" \
      "$PY" "$CODE/automation/fruit3d_cluster.py" "$W"
  done
fi

run_item xepoch_v3 "$PY" "$CODE/automation/fruit_cross_epoch_v3.py"

note "WEEKEND-QUEUE-DONE"
