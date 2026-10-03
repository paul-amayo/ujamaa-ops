#!/bin/bash
# native_sam3_queue.sh — chunk-specific supervision chain on 05 chunk 0_0 then 01 chunk 3_1 (UJAMAA, 2026-10-03)
NT=/home/paperspace/code/automation/h3dgs/native; C=/home/paperspace/data/citrus_all
S=$C/05_13D_Jackal; SVN=05_13D_Jackal PROJ=$S/experimental/h3dgs_expo CN=0_0 OUT=$S/experimental/h3dgs_native/chunk_0_0_sam3 MUST=kf_001431.png \
  EARLIER=$S/experimental/h3dgs_native/chunk_0_0_expo/features_census.bin bash $NT/native_sam3_cell.sh
S=$C/01_13B_Jackal; SVN=01_13B_Jackal PROJ=$S/experimental/h3dgs CN=3_1 OUT=$S/experimental/h3dgs_native/chunk_3_1_sam3 MUST=kf_003426.png \
  EARLIER=$S/experimental/h3dgs_native/chunk_3_1/features_census.bin bash $NT/native_sam3_cell.sh
echo "[queue] sam3 cells done $(date '+%H:%M')"
