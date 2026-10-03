#!/bin/bash
# disk_cleanup_20261003.sh — deletions Paul authorised 2026-10-03 13:5x box: "delete 02 raw rosbag g0 and 05 h3dgs
# ablations and cabbage stage 2 refine negatives" (the audit's candidate list, lab_notebook/2026-10.md).
# Pre-checks done before writing this: no process had any target open, no symlink outside the cabbage eval stubs
# pointed into a target, 02's monolithic set equals the bagless surveys' (01/05), G0 models read only by
# automation/g0_qlora + automation/multiling_eval. Explicit paths only; refuses anything outside the two roots.
set -u
LOG=/home/paperspace/logs/disk_cleanup_20261003.log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $LOG; }
free(){ df --output=avail -B1 / | tail -1 | awk '{printf "%.1f GB", $1/1e9}'; }
D=/home/paperspace/data; HF=/home/paperspace/.cache/huggingface/hub; E05=$D/citrus_all/05_13D_Jackal/experimental
L=$D/image_farm/gwakungu/2026-05-16/IMG_7993_s0_cabbage/demo_root/experimental/h3dgs_sidecar_chunks/chunk_lane/splat_runs_FEATFIX
TARGETS=(
  "02 raw rosbag (extracted to monolithics)|$D/citrus_all/02_13B_Jackal/prod/monos/combined.bag"
  "G0 fine-tune run (negative result 09-20)|$D/g0_run"
  "G0 tokenised corpus|$D/g0_tokens"
  "HF base unsloth/gemma-4-12b (G0 + multiling HF arm)|$HF/models--unsloth--gemma-4-12b"
  "HF lock dir for it|$HF/.locks/models--unsloth--gemma-4-12b"
  "05 H3DGS ablation halfview|$E05/h3dgs_halfview"
  "05 H3DGS ablation budget|$E05/h3dgs_budget"
  "05 V100 parity reference (08-21 check: 9/9 MATCH)|$E05/parity_ref_v100_20260821"
  "cabbage refine lr1e-3 2k|$L/stage2_train_lane_v2_live2k"
  "cabbage refine lr1e-3 2k eval stubs|$L/stage2_train_lane_v2_live2k_eval_step2000"
  "cabbage refine lr1e-2 2k|$L/stage2_train_lane_v2_lr1e2_2k"
  "cabbage refine lr1e-2 2k eval stubs|$L/stage2_train_lane_v2_lr1e2_2k_eval_step2000"
  "cabbage refine lr1e-2 7.5k|$L/stage2_train_lane_v2_lr1e2_7k5"
  "cabbage refine lr1e-2 7.5k eval stubs 2250|$L/stage2_train_lane_v2_lr1e2_7k5_eval_step2250"
  "cabbage refine lr1e-2 7.5k eval stubs 5500|$L/stage2_train_lane_v2_lr1e2_7k5_eval_step5500"
  "cabbage refine lr1e-2 bg0.1 2k|$L/stage2_train_lane_v2_lr1e2_bg01_2k"
  "cabbage refine lr1e-2 bg0.1 2k eval stubs|$L/stage2_train_lane_v2_lr1e2_bg01_2k_eval_step2000"
)
say "=== start; free $(free)"
TOT=0
for t in "${TARGETS[@]}"; do
  name=${t%%|*}; p=${t#*|}
  case "$p" in /home/paperspace/data/?*|/home/paperspace/.cache/huggingface/hub/?*) ;; *) say "REFUSED (outside allowed roots): $p"; continue;; esac
  if [ ! -e "$p" ] && [ ! -L "$p" ]; then say "absent: $name  $p"; continue; fi
  b=$(du -sxB1 "$p" 2>/dev/null | cut -f1)
  if rm -rf -- "$p"; then TOT=$((TOT+b)); say "deleted $(awk -v b=$b 'BEGIN{printf "%7.2f GB",b/1e9}')  $name  $p"; else say "FAILED rm: $name  $p"; fi
done
say "files deleted: $(awk -v b=$TOT 'BEGIN{printf "%.2f GB",b/1e9}')"
for m in hf.co/mradermacher/gemma-3-12b-pt-GGUF:Q4_K_M hf.co/mradermacher/AfriqueGemma-12B-GGUF:Q4_K_M; do
  if ollama rm "$m" >> $LOG 2>&1; then say "ollama rm $m (multiling eval only)"; else say "FAILED ollama rm $m"; fi
done
say "ollama models left: $(ollama list | awk 'NR>1{print $1}' | tr '\n' ' ')"
say "=== done; free $(free)"
