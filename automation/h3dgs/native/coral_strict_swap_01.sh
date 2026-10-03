#!/bin/bash
# 01 row swap to CORAL strict (Paul 2026-10-03: "run 01 on coral with the strict solution" -> "yeah retrain reseed").
#  1. quarantine (COPY) the census-0.7 hierarchy + the embedder trained on it -> experimental/era_census07_20260822/
#  2. install the CORAL dir-0.985 hierarchy (experimental/coral_rows_20261003/strict_0985) as prod marker_hierarchy.json
#  3. retrain the survey embedder IN PLACE with the canonical recipe (run_unified_pipeline step 5b: graph c20cos20, 1500 ep)
#  4. re-seed chunk 3_1 native identity (seeds A and B from the existing census W.npz) + rebuild its text bank; the old
#     seeds/bank move to <cell>/era_census07_20260822/ (they pair with the quarantined embedder)
set -euo pipefail
S=/home/paperspace/data/citrus_all/01_13B_Jackal; B=$S/prod/bateleur; HJ=$B/scene_graph/marker_hierarchy.json
Q=$S/experimental/era_census07_20260822; NEWH=$S/experimental/coral_rows_20261003/strict_0985/marker_hierarchy.json
ARU=/home/paperspace/code/aru_sil_core/src/scripts; HIGH=/home/paperspace/code/aru_sil_core/src/interfaces/rerun/HiGH
NAT=/home/paperspace/code/automation/h3dgs/native; N=$S/experimental/h3dgs_native/chunk_3_1_sam3; L=/home/paperspace/logs/coral_strict_swap_01
EMB=$B/embedder/01_13B_v1g/ckpts/model_best.pth
say() { echo "[$(date '+%H:%M:%S')] $*"; }
mkdir -p "$Q" "$N/era_census07_20260822"
say "1 quarantine copies -> $Q"
[ -f "$Q/marker_hierarchy.json" ] || cp -a "$HJ" "$Q/marker_hierarchy.json"
[ -d "$Q/embedder_01_13B_v1g" ] || cp -a "$B/embedder/01_13B_v1g" "$Q/embedder_01_13B_v1g"
say "2 install CORAL strict hierarchy"
cp "$NEWH" "$HJ"
python3 -c "import json; p=json.load(open('$HJ'))['_provenance']; print('   prod hierarchy now:', p['row_solver'], p['dominant_direction_threshold'], p['n_rows'], 'rows')"
say "3 retrain embedder 01_13B_v1g (graph c20cos20, 1500 epochs)"
cd /home/paperspace/code/nerf_new
pixi run python "$HIGH/train_hyperembedder_graph.py" --hierarchy-json "$HJ" --experiment-name 01_13B_v1g --output-dir "$B/embedder" \
    --contrastive-weight 2.0 --cosine-reconstruction-weight 2.0 --reconstruction-weight 1.0 --temperature 0.2 --keep-super-row \
    --epochs 1500 --no-level-norms > "${L}_embedder.log" 2>&1
ls -la --time-style=+%m-%d_%H:%M "$EMB"
say "4 re-seed chunk 3_1 (seeds A, B) + text bank"
for f in features_A_block.bin features_A_block.bin.json features_B_bg2share.bin features_B_bg2share.bin.json text_bank.npz; do
    [ -e "$N/$f" ] && [ ! -e "$N/era_census07_20260822/$f" ] && mv "$N/$f" "$N/era_census07_20260822/$f"
done
HIGH_EMBEDDER_CKPT=$EMB pixi run python $ARU/build_census_init.py --w-npz $N/W.npz --embedder $EMB --out-features $N/features_A_block.bin --tree-floor 1.0 2>&1 | grep -aE 'floors|assigned|node features|Error' | sed 's/^/   [seed A] /'
HIGH_EMBEDDER_CKPT=$EMB pixi run python $ARU/build_census_init.py --w-npz $N/W.npz --embedder $EMB --out-features $N/features_B_bg2share.bin --tree-floor 1.0 --bg-competes --bg-ratio 2 --fruit-share-assign 0.1 2>&1 | grep -aE 'background|floors|share|assigned|node features|Error' | sed 's/^/   [seed B] /'
pixi run python $NAT/build_text_bank.py --manifest $N/supervision/trees_only/manifest.json --hierarchy-json "$HJ" --out $N/text_bank.npz 2>&1 | grep -a "text-bank\|Error"
say "done"
