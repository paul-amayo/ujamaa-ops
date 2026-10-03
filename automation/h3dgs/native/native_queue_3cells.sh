#!/bin/bash
# native_queue_3cells.sh — native census v2 on 05 chunk 0_0, 01 chunk 3_1, Gwakungu cabbage lane, one after another (2026-10-03)
NT=/home/paperspace/code/automation/h3dgs/native; C=/home/paperspace/data/citrus_all; G=/home/paperspace/data/image_farm/gwakungu/2026-05-16
S=$C/05_13D_Jackal; SV=$S PROJ=$S/experimental/h3dgs_expo CN=0_0 SC=$S/experimental/h3dgs_sidecar_chunks/chunk_0_0_expo \
  EMB=$S/prod/bateleur/embedder/05_13D_v1g/ckpts/model_best.pth HJ=$S/prod/bateleur/scene_graph/marker_hierarchy.json KF=$S/prod/scratch_sam3 \
  OUT=$S/experimental/h3dgs_native/chunk_0_0_expo MUST=kf_001431.png bash $NT/native_cell_test_v2.sh
S=$C/01_13B_Jackal; SV=$S PROJ=$S/experimental/h3dgs CN=3_1 SC=$S/experimental/h3dgs_sidecar_chunks/chunk_3_1 \
  EMB=$S/prod/bateleur/embedder/01_13B_v1g/ckpts/model_best.pth HJ=$S/prod/bateleur/scene_graph/marker_hierarchy.json KF=$S/prod/scratch_sam3 \
  OUT=$S/experimental/h3dgs_native/chunk_3_1 MUST=kf_003426.png bash $NT/native_cell_test_v2.sh
D=$G/IMG_7993_s0_cabbage/demo_root; SC=$D/experimental/h3dgs_sidecar_chunks/chunk_lane
SV=$D PROJ=$G/IMG_7993_s0/h3dgs_8m CN=lane SC=$SC SIDECAR_CFG=$(ls $SC/splat_runs_FEATFIX/stage2_censusinit_glref_bg_f1.0_r2_IMG_7993_s0_cabbage_v2_e400/high/*/config.yml | head -1) \
  EMB=$D/prod/bateleur/embedder/IMG_7993_s0_cabbage_v2_e400/ckpts/model_best.pth HJ=$D/prod/bateleur/scene_graph/marker_hierarchy.json KF=$D/prod/scratch_sam3 \
  OUT=$D/experimental/h3dgs_native/chunk_lane MUST="image_33.png image_40.png image_50.png image_52.png image_64.png" NTR=6 NEV=3 bash $NT/native_cell_test_v2.sh
echo "[queue] three cells done $(date '+%H:%M')"
