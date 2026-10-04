#!/bin/bash
# Runs fruit_fp_trace.py once the GPU queue (cabbage whole-clip H3DGS) is done. Same scene as cell chunk_1_0_sam3's census.
E=/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs_expo; OUT=/home/paperspace/data/demo_video_v2/citrus_b_cut_trained/fp_trace
L=/home/paperspace/logs/fruit_fp_trace.log
until grep -q "queue done" /home/paperspace/logs/gpu_queue_chilli_cabbage.log 2>/dev/null; do sleep 30; done
while [ "$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '$1 > 2048' | wc -l)" -gt 0 ]; do sleep 30; done
echo "[$(date '+%H:%M:%S')] GPU free, tracing" > $L
cd /home/paperspace/code/hierarchical-3d-gaussians && env CUDA_HOME=/home/paperspace/code/_cuda12 PATH=/home/paperspace/code/_cuda12/bin:$PATH \
  /home/paperspace/miniconda3/envs/h3dgs/bin/python /home/paperspace/code/automation/h3dgs/native/fruit_fp_trace.py \
  -s $E/camera_calibration/chunks/1_0 -m $E/output/trained_chunks/1_0 --hierarchy $E/output/trained_chunks/1_0/hierarchy.hier_opt \
  -i ../../rectified/images --eval --data_device cpu --out $OUT >> $L 2>&1
echo "[$(date '+%H:%M:%S')] trace rc=$?" >> $L
